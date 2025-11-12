import importlib
import io
import logging
import logging.handlers
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import unquote_plus

import psutil
import numpy as np
import torch

try:
    from dotenv import load_dotenv  # type: ignore[import]
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> None:
        """本地未安装 python-dotenv 时的兜底实现。"""

        return None


from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from kokoro import KModel, KPipeline
import soundfile as sf


logger = logging.getLogger("kokoro_tts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

load_dotenv()

# 配置项
MONITOR_INTERVAL_SECONDS = float(os.getenv("KOKORO_MONITOR_INTERVAL", "10"))
LOG_MAX_BYTES = int(os.getenv("KOKORO_LOG_MAX_BYTES", "5000000"))
LOG_BACKUP_COUNT = int(os.getenv("KOKORO_LOG_BACKUP_COUNT", "3"))
DEFAULT_DEVICE_PREFERENCE = os.getenv("KOKORO_DEVICE", "auto")
TOKEN_HEADER_NAME = os.getenv("KOKORO_TOKEN_HEADER", "X-API-Token")
AUTH_TOKENS = {
    token.strip()
    for token in os.getenv("KOKORO_API_TOKEN", "").split(",")
    if token.strip()
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "tts_service.log",
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

if not any(
    isinstance(handler, logging.handlers.RotatingFileHandler)
    for handler in logger.handlers
):
    logger.addHandler(file_handler)
    logger.propagate = False

if not AUTH_TOKENS:
    logger.warning(
        "API token set is empty. Configure KOKORO_API_TOKEN to enable authenticated access."
    )


class ResourceMonitor:
    """后台采样线程，用于周期性记录 CPU 和 GPU 的资源占用。

    Parameters
    ----------
    interval : float, default=MONITOR_INTERVAL_SECONDS
        采样间隔（秒）。当传入值小于 1 时会自动提升到 1 秒，以避免
        过于频繁的系统调用。

    Attributes
    ----------
    interval : float
        当前有效采样间隔。
    _thread : threading.Thread or None
        后台采样线程句柄。
    _stop_event : threading.Event
        用于通知线程停止的事件对象。
    _process : psutil.Process
        当前进程对象，负责读取 CPU/内存指标。
    _logger : logging.Logger
        专用的监控日志记录器。
    _gpu_available : bool
        是否检测到 CUDA 设备。
    _gpu_devices : list of int
        可用 GPU 索引列表，用于枚举显存指标。
    """

    def __init__(self, interval: float = MONITOR_INTERVAL_SECONDS) -> None:
        self.interval = max(1.0, interval)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # 预热采样，避免首个数据偏差
        self._logger = logger.getChild("monitor")
        self._gpu_available = torch.cuda.is_available()
        self._gpu_devices = (
            list(range(torch.cuda.device_count())) if self._gpu_available else []
        )

    def start(self) -> None:
        """启动监控线程。

        若线程已启动则直接返回，实现幂等的生命周期管理。
        """

        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ResourceMonitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止监控线程。

        在线程对象不存在的情况下调用同样安全，方法会被忽略。
        """

        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        """线程主体逻辑，按设定间隔刷新资源统计。"""

        while not self._stop_event.wait(self.interval):
            try:
                self._log_stats()
            except Exception:  # pragma: no cover - 监控线程自身不能导致服务崩溃
                self._logger.exception("Resource monitor failed")

    def _log_stats(self) -> None:
        """收集 CPU、内存及 GPU 指标并写入日志。"""

        cpu_percent = self._process.cpu_percent(interval=None)
        memory_info = self._process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)

        gpu_stats = []
        if self._gpu_available:
            for device_index in self._gpu_devices:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                used_mb = (total_bytes - free_bytes) / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved(device_index) / (1024 * 1024)
                allocated_mb = torch.cuda.memory_allocated(device_index) / (1024 * 1024)
                gpu_stats.append(
                    {
                        "gpu": device_index,
                        "mem_used_mb": round(used_mb, 2),
                        "mem_reserved_mb": round(reserved_mb, 2),
                        "mem_allocated_mb": round(allocated_mb, 2),
                        "mem_total_mb": round(total_bytes / (1024 * 1024), 2),
                    }
                )

        self._logger.info(
            "cpu=%.2f%% rss=%.2fMB vms=%.2fMB gpu=%s",
            cpu_percent,
            rss_mb,
            vms_mb,
            gpu_stats if gpu_stats else "none",
        )


class KokoroService:
    """封装 Kokoro 模型的加载、声线缓存与推理逻辑。

    Parameters
    ----------
    repo_id : str
        Hugging Face 上的仓库标识，用于同步配置。
    model_path : pathlib.Path
        本地模型权重路径。
    config_path : pathlib.Path
        本地模型配置文件路径。
    voices_dir : pathlib.Path
        声线向量目录，目录下文件名对应声线标识。
    default_voice : str, default="zf_001"
        当请求未指定声线时使用的默认声线。
    sample_rate : int, default=24000
        输出音频采样率（Hz）。
    device_preference : str or None, default=auto
        用户期望的计算设备标识，支持 "cpu"、"cuda"、"auto" 或具体
        的 CUDA 设备名称。

    Attributes
    ----------
    repo_id, model_path, config_path, voices_dir, default_voice, sample_rate
        对应构造参数，保持原样存储。
    device_preference : str
        正常化后的设备偏好，默认取环境变量 ``KOKORO_DEVICE``。
    _model : kokoro.KModel or None
        模型实例，只加载一次。
    _zh_pipeline, _en_pipeline : kokoro.KPipeline or None
        中文/英文管线，按需延迟构建。
    _voice_cache : dict[str, torch.Tensor]
        声线缓存，避免重复读取磁盘。
    _model_device : torch.device or None
        当前模型所在设备，用于避免重复迁移。
    """

    def __init__(
        self,
        repo_id: str,
        model_path: Path,
        config_path: Path,
        voices_dir: Path,
        default_voice: str = "zf_001",
        sample_rate: int = 24000,
        device_preference: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.model_path = model_path
        self.config_path = config_path
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.device_preference = (
            device_preference or DEFAULT_DEVICE_PREFERENCE
        ).lower()

        self._model: Optional[KModel] = None
        self._zh_pipeline: Optional[KPipeline] = None
        self._en_pipeline: Optional[KPipeline] = None
        self._voice_cache: Dict[str, torch.Tensor] = {}
        self._model_device: Optional[torch.device] = None

    def initialize(self) -> None:
        """懒加载模型和管线，避免重复初始化。

        Raises
        ------
        FileNotFoundError
            当模型权重或配置文件缺失时触发。
        ValueError
            当设置了不支持的设备偏好时由内部调用抛出。
        """

        if self._model is not None:
            return

        if not self.model_path.exists() or not self.config_path.exists():
            raise FileNotFoundError("Kokoro model or config file not found")

        logger.info("Loading Kokoro model from %s", self.model_path)
        self._model = KModel(
            model=str(self.model_path),
            config=str(self.config_path),
            repo_id=self.repo_id,
        ).eval()

        target_device, fallback = self._resolve_device(None)
        if fallback:
            logger.warning(
                "Preferred device '%s' unavailable. Falling back to %s.",
                self.device_preference,
                target_device,
            )

        self._ensure_model_device(target_device)

        self._en_pipeline = KPipeline(lang_code="a", repo_id=self.repo_id, model=False)
        self._zh_pipeline = KPipeline(
            lang_code="z",
            repo_id=self.repo_id,
            model=self._model,
            en_callable=self._en_callable,
        )

        logger.info("Kokoro model loaded on %s", self.current_device)

    def _en_callable(self, text: str) -> str:
        """针对混入中文的特定英文词汇做音素补丁。

        Parameters
        ----------
        text : str
            原始英文词汇。

        Returns
        -------
        str
            经过规则覆盖后的音素字符串。
        """

        special_phonemes = {
            "Kokoro": "kˈOkəɹO",
            "Sol": "sˈOl",
        }
        if text in special_phonemes:
            return special_phonemes[text]
        assert self._en_pipeline is not None
        return next(self._en_pipeline(text)).phonemes

    def _load_voice(self, voice_name: str) -> torch.Tensor:
        """从磁盘加载声线向量并缓存，减少重复 I/O。

        Parameters
        ----------
        voice_name : str
            声线标识，需与 ``voices_dir`` 中的文件名匹配。

        Returns
        -------
        torch.Tensor
            对应声线的向量表示。

        Raises
        ------
        FileNotFoundError
            当指定声线文件不存在时抛出。
        """

        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        voice_file = self.voices_dir / f"{voice_name}.pt"
        if not voice_file.exists():
            raise FileNotFoundError(f"Voice profile '{voice_name}' not found")

        logger.info("Loading voice profile %s", voice_name)
        voice_tensor = torch.load(voice_file, map_location="cpu", weights_only=True)
        self._voice_cache[voice_name] = voice_tensor
        return voice_tensor

    def _resolve_device(
        self, device_request: Optional[str]
    ) -> Tuple[torch.device, bool]:
        """解析目标计算设备，必要时返回是否退回 CPU。

        Parameters
        ----------
        device_request : str or None
            请求指定的设备标识；若为空则回退到全局偏好或自动检测。

        Returns
        -------
        device : torch.device
            解析出的目标设备。
        fallback : bool
            若原始请求无法满足而退回到 CPU，则返回 ``True``。

        Raises
        ------
        ValueError
            当传入无法解析的设备字符串时抛出。
        """

        preference = (
            (device_request or self.device_preference or "auto").strip().lower()
        )
        fallback = False

        if preference in {"", "auto"}:
            if torch.cuda.is_available():
                return torch.device("cuda"), fallback
            return torch.device("cpu"), fallback

        if preference in {"gpu", "cuda"} or preference.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(
                    preference if preference != "gpu" else "cuda"
                ), fallback
            fallback = True
            return torch.device("cpu"), fallback

        if preference == "cpu":
            return torch.device("cpu"), fallback

        raise ValueError(f"Unsupported device preference '{preference}'")

    def _ensure_model_device(self, device: torch.device) -> None:
        """按需将模型权重迁移到目标设备。

        Parameters
        ----------
        device : torch.device
            目标计算设备。

        Raises
        ------
        RuntimeError
            当模型尚未初始化时调用该方法会触发。
        """

        if self._model is None:
            raise RuntimeError("Model must be initialized before ensuring device")

        if self._model_device == device:
            return

        logger.info("Moving Kokoro model to %s", device)
        self._model.to(device)
        if device.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_device = device

    @property
    def current_device(self) -> str:
        """返回当前模型所驻留的设备名称。

        Returns
        -------
        str
            若模型已加载则返回设备字符串，否则返回 ``"uninitialized"``。
        """

        return str(self._model_device) if self._model_device else "uninitialized"

    def _base_speed_callable(self) -> Callable[[int], float]:
        """提供 Kokoro 官方中文语速曲线。

        Returns
        -------
        callable
            接受音素长度并返回基础语速倍率的回调函数。
        """

        def speed_callable(len_ps: int) -> float:
            speed = 0.8
            if len_ps <= 83:
                speed = 1.0
            elif len_ps < 183:
                speed = 1.0 - (len_ps - 83) / 500
            return speed * 1.1

        return speed_callable

    def _make_speed_callable(self, speed_scale: float) -> Callable[[int], float]:
        """在默认曲线基础上叠加外部倍率。

        Parameters
        ----------
        speed_scale : float
            用户指定的语速倍率。

        Returns
        -------
        callable
            组合后的语速回调，可直接传入 Kokoro 管线。
        """

        base_callable = self._base_speed_callable()

        def scaled_speed(len_ps: int) -> float:
            return base_callable(len_ps) * speed_scale

        return scaled_speed

    def synthesize(
        self,
        text: str,
        speak_speed: float,
        voice_name: Optional[str],
        device_request: Optional[str] = None,
    ) -> tuple[bytes, float, str]:
        """执行语音合成，返回音频字节、RTF 与实际使用的设备。

        Parameters
        ----------
        text : str
            待合成的文本。
        speak_speed : float
            期望语速，范围为 5-50。
        voice_name : str or None
            使用的声线标识，缺省时采用 ``default_voice``。
        device_request : str or None, default=None
            本次推理的设备偏好，优先级高于全局设置。

        Returns
        -------
        audio_bytes : bytes
            生成的 WAV 音频数据。
        rtf : float
            实际推理的实时因子（越小越快）。
        device : str
            推理过程中最终使用的设备。

        Raises
        ------
        ValueError
            当输入文本为空白时抛出。
        FileNotFoundError
            声线文件缺失时由内部调用抛出。
        RuntimeError
            当 CUDA 推理失败且 CPU 回退同样失败时会传播原始异常。
        """

        if not text.strip():
            raise ValueError("Text for synthesis cannot be empty")

        self.initialize()
        assert self._zh_pipeline is not None

        selected_voice = voice_name or self.default_voice
        voice_tensor = self._load_voice(selected_voice)

        speed_scale = self._resolve_speed_scale(speak_speed)
        speed_callable = self._make_speed_callable(speed_scale)

        target_device, fallback = self._resolve_device(device_request)
        if fallback and device_request:
            logger.warning(
                "Requested device '%s' unavailable. Falling back to %s.",
                device_request,
                target_device,
            )

        def run_on(device: torch.device) -> list[np.ndarray]:
            self._ensure_model_device(device)
            return self._run_pipeline(
                self._zh_pipeline(
                    text,
                    voice=voice_tensor,  # type: ignore[arg-type]
                    speed=speed_callable,
                )
            )

        start_time = time.perf_counter()
        used_device = target_device

        try:
            audio_chunks = run_on(target_device)
        except RuntimeError as exc:
            if target_device.type == "cuda" and "cuda" in str(exc).lower():
                logger.warning(
                    "CUDA inference failed (%s). Retrying on CPU.",
                    exc,
                )
                cpu_device, _ = self._resolve_device("cpu")
                audio_chunks = run_on(cpu_device)
                used_device = cpu_device
            else:
                raise

        elapsed = time.perf_counter() - start_time

        waveform = (
            np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        )
        speech_duration = len(waveform) / self.sample_rate
        rtf = elapsed / speech_duration if speech_duration else 0.0

        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.sample_rate, format="WAV")
        buffer.seek(0)

        logger.info(
            "Synthesized %d chars with voice=%s speed=%.2f rtf=%.3f duration=%.2fs device=%s",
            len(text),
            selected_voice,
            speak_speed,
            rtf,
            speech_duration,
            used_device,
        )

        return buffer.read(), rtf, str(used_device)

    def _run_pipeline(self, generator: Iterable) -> list[np.ndarray]:
        """收集推理产出的音频块，并转换为 numpy 数组。

        Parameters
        ----------
        generator : Iterable
            Kokoro 管线返回的生成器对象。

        Returns
        -------
        list of numpy.ndarray
            连续的音频块序列。

        Raises
        ------
        RuntimeError
            当生成器未产生任何音频数据时抛出。
        """

        audio_chunks: list[np.ndarray] = []
        for chunk in generator:
            audio = chunk.audio
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio_chunks.append(audio)
        if not audio_chunks:
            raise RuntimeError("Pipeline did not produce any audio chunks")
        return audio_chunks

    def _resolve_speed_scale(self, speak_speed: float) -> float:
        """将 5-50 的 UI 输入映射到语速倍率区间。

        Parameters
        ----------
        speak_speed : float
            语速控制值。

        Returns
        -------
        float
            对应的语速倍率。
        """

        clamped = max(5.0, min(50.0, speak_speed))
        min_scale, max_scale = 0.6, 1.4
        return min_scale + (clamped - 5.0) * (max_scale - min_scale) / (50.0 - 5.0)


def decode_text(value: Optional[str]) -> Optional[str]:
    """最多执行两次 URL 解码，并裁剪首尾空白。

    Parameters
    ----------
    value : str or None
        可能经过 URL 编码的文本。

    Returns
    -------
    str or None
        解码后的文本；若入参为 ``None`` 则返回 ``None``。
    """

    if value is None:
        return None
    decoded = value
    for _ in range(2):
        new_value = unquote_plus(decoded)
        if new_value == decoded:
            break
        decoded = new_value
    return decoded.strip()


def resolve_speak_speed(speak_speed: Optional[float], spd: Optional[float]) -> float:
    """将速度参数换算为 5-50 范围。

    Parameters
    ----------
    speak_speed : float or None
        直接指定的语速。
    spd : float or None
        接口中的 ``spd`` 参数，将转换为 ``speak_speed``。

    Returns
    -------
    float
        归一化后的语速值。
    """

    if speak_speed is not None:
        return float(speak_speed)
    if spd is not None:
        return float((spd - 4.0) * 10.0 - 5.0)
    return 25.0


def ensure_authenticated(
    request: Request,
    token: Optional[str] = Header(None, alias=TOKEN_HEADER_NAME),
    authorization: Optional[str] = Header(None),
) -> str:
    """校验请求头中的访问令牌。

    Parameters
    ----------
    request : fastapi.Request
        当前 HTTP 请求对象，用于读取查询参数等附加信息。
    token : str or None
        来自 ``TOKEN_HEADER_NAME`` 指定请求头中的令牌。
    authorization : str or None
        可选的 ``Authorization`` 头部，支持 ``Bearer`` 格式。

    Returns
    -------
    str
        通过认证的令牌。

    Raises
    ------
    HTTPException
        - 503: 未配置服务端令牌时抛出。
        - 401: 请求未提供有效令牌时抛出（包括格式错误、缺失或值不匹配）。
    """

    if not AUTH_TOKENS:
        raise HTTPException(status_code=503, detail="API token not configured")

    candidates: list[str] = []

    if token:
        candidates.append(token.strip())

    if authorization:
        auth_value = authorization.strip()
        if auth_value.lower().startswith("bearer "):
            candidates.append(auth_value[7:].strip())
        elif auth_value:
            candidates.append(auth_value)

    for query_key in ("token", "api_token", "apiToken", "key", "auth"):
        value = request.query_params.get(query_key)
        if value:
            candidates.append(value.strip())

    for candidate in candidates:
        if candidate and candidate in AUTH_TOKENS:
            return candidate

    raise HTTPException(status_code=401, detail="Missing API token")


repo_id = "hexgrad/Kokoro-82M-v1.1-zh"
model_path = Path("model/kokoro_V1.1_zh/kokoro-v1_1-zh.pth")
config_path = Path("model/kokoro_V1.1_zh/config.json")
voices_dir = Path("model/kokoro_V1.1_zh/voices")

service = KokoroService(
    repo_id=repo_id,
    model_path=model_path,
    config_path=config_path,
    voices_dir=voices_dir,
)

monitor = ResourceMonitor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI 生命周期钩子：启动时初始化模型，退出时回收监控线程。"""

    service.initialize()
    monitor.start()
    try:
        yield
    finally:
        monitor.stop()


app = FastAPI(title="Kokoro TTS Service", version="0.1.0", lifespan=lifespan)


async def generate_audio_response(
    tex: Optional[str],
    text: Optional[str],
    speak_speed: Optional[float],
    spd: Optional[float],
    voice: Optional[str],
    device_request: Optional[str],
):
    """合并 GET/POST 逻辑，负责统一的语音合成流程。

    Parameters
    ----------
    tex, text : str or None
        需要合成的文本字段，优先使用 ``tex``。
    speak_speed : float or None
        直接传入的语速值。
    spd : float or None
    接口对应语速参数，会换算为 ``speak_speed``。
    voice : str or None
        可选声线标识。
    device_request : str or None
        当前请求希望使用的运行设备。

    Returns
    -------
    fastapi.responses.StreamingResponse
        包含 WAV 音频流以及性能头部的响应对象。

    Raises
    ------
    HTTPException
        当文本缺失、声线不存在或底层推理失败时抛出。

    Notes
    -----
    安全校验由路由依赖 ``ensure_authenticated`` 负责，在进入该函数之前
    请求已通过令牌验证。
    """

    decoded_text = decode_text(tex) or decode_text(text)
    if not decoded_text:
        raise HTTPException(
            status_code=400, detail="Missing text content (tex or text)"
        )

    resolved_speed = resolve_speak_speed(speak_speed, spd)

    try:
        audio_bytes, rtf, device_used = await run_in_threadpool(
            service.synthesize,
            decoded_text,
            resolved_speed,
            voice,
            device_request,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected errors
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed") from exc

    headers = {
        "X-RTF": f"{rtf:.4f}",
        "X-Sample-Rate": str(service.sample_rate),
        "X-Device": device_used,
    }

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/text2audio")
async def text2audio_post(
    tex: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    speakSpeed: Optional[float] = Form(None),
    spd: Optional[float] = Form(None),
    voice: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    _: str = Depends(ensure_authenticated),
):
    """POST 接口。

    直接读取表单字段，复用通用的合成逻辑。调用者需在请求头中携带
    默认名为 ``X-API-Token`` 的令牌，或在运行时通过环境变量覆盖。
    """

    return await generate_audio_response(tex, text, speakSpeed, spd, voice, device)


@app.get("/text2audio")
async def text2audio_get(
    tex: Optional[str] = Query(None),
    text: Optional[str] = Query(None),
    speakSpeed: Optional[float] = Query(None),
    spd: Optional[float] = Query(None),
    voice: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
    _: str = Depends(ensure_authenticated),
):
    """GET 接口。

    支持通过查询字符串直接发起朗读请求。同样要求请求头包含有效令牌。
    """

    return await generate_audio_response(tex, text, speakSpeed, spd, voice, device)


@app.get("/health")
async def health() -> JSONResponse:
    """简易健康检查，返回设备状态与模型加载情况。

    Returns
    -------
    fastapi.responses.JSONResponse
        包含 ``status``、``device`` 与 ``model_loaded`` 字段的响应。
    """

    return JSONResponse(
        {
            "status": "ok",
            "device": service.current_device,
            "model_loaded": service._model is not None,
        }
    )


if __name__ == "__main__":
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:  # pragma: no cover - executed only in CLI mode
        raise RuntimeError(
            "uvicorn is required to run the server from the CLI"
        ) from exc

    uvicorn.run(
        "tts_service:app",
        host="0.0.0.0",
        port=3236,
        reload=False,
        workers=1,
    )
