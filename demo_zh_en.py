import torch
import time
from kokoro import KPipeline, KModel
import soundfile as sf

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型配置
repo_id = "hexgrad/Kokoro-82M-v1.1-zh"
model_path = "model/kokoro_V1.1_zh/kokoro-v1_1-zh.pth"
config_path = "model/kokoro_V1.1_zh/config.json"

# 加载模型
model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()

# 速度回调函数（官方默认实现）
def speed_callable(len_ps):
    """
    根据音素长度动态调整语速:
    - 83音素以内: 1.1倍速
    - 83-183音素: 线性减速
    - 183音素以上: 0.88倍速
    """
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

# 公共音色配置 (两个任务共用同一音色)
voice = "zf_001"
voice_tensor = torch.load(f"model/kokoro_V1.1_zh/voices/{voice}.pt", weights_only=True)


# ======================
# 中英混合TTS
# ======================
sentence_mix = "卧槽，小米怎么那么坏啊! What Fuck, Xiaomi is so bad!"

# 英文处理专用pipeline (不加载模型)
en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

# 英文回调函数（官方默认实现）
def en_callable(text):
    """特殊英文词汇发音覆盖"""
    if text == "Kokoro":
        return "kˈOkəɹO"
    elif text == "Sol":
        return "sˈOl" 
    return next(en_pipeline(text)).phonemes

# 中英混合pipeline
zh_pipeline_mix = KPipeline(
    lang_code="z", 
    repo_id=repo_id, 
    model=model, 
    en_callable=en_callable
)

# 生成音频
start_time = time.time()
generator_mix = zh_pipeline_mix(sentence_mix, voice=voice_tensor, speed=speed_callable)
result_mix = next(generator_mix)
wav_mix = result_mix.audio
speech_len_mix = len(wav_mix) / 24000

# 性能统计
rtf_mix = (time.time() - start_time) / speech_len_mix
print(f"中英混合音频长度: {speech_len_mix:.2f}秒, RTF: {rtf_mix:.4f}")

sf.write("demo_zh_en.wav", wav_mix, 24000)


# ======================
# 纯中文TTS
# ======================
sentence_zh = "卧槽，小米怎么那么坏啊!"

# 纯中文pipeline 
zh_pipeline_zh = KPipeline(lang_code="z", repo_id=repo_id, model=model)

# 生成音频
start_time = time.time()
generator_zh = zh_pipeline_zh(sentence_zh, voice=voice_tensor, speed=speed_callable)
result_zh = next(generator_zh)
wav_zh = result_zh.audio
speech_len_zh = len(wav_zh) / 24000

# 性能统计
rtf_zh = (time.time() - start_time) / speech_len_zh
print(f"纯中文音频长度: {speech_len_zh:.2f}秒, RTF: {rtf_zh:.4f}")

sf.write("demo_zh.wav", wav_zh, 24000)