<div style="text-align: center;">
<h1>Kokoro TTS Service</h1>
<p><a href="README.md">中文</a> | <a href="README_en.md">English</a></p>
</div>

本项目将 `hexgrad/Kokoro-82M-v1.1-zh` 模型推理包装在一个轻量级的 FastAPI 服务中，提供 `/text2audio` 接口以供[阅读3.0](https://github.com/gedoor/legado)的 TTS 使用。

## 先决条件

- Python 3.12+
- 放置在 `model/` 目录下的 Kokoro 1.1 中文权重和声音文件
- 通过 `KOKORO_API_TOKEN` 环境变量配置的访问令牌（调用接口必需）

可选但推荐：

- 运行 `device=cuda` 时需要支持 CUDA 的 GPU

## 环境设置

```bash
# 创建虚拟环境并安装依赖
uv lock
uv sync
```

## 运行服务

```powershell
# 从项目根目录执行
uv run uvicorn tts_service:app --host 0.0.0.0 --port 3236

# 终端输出示例
# INFO kokoro_tts: Kokoro model loaded on cuda
# INFO kokoro_tts.monitor: cpu=5.12% rss=1024.00MB ...
```



启动成功后可以访问：

- `http://127.0.0.1:3236/health`   —— 查看模型是否已加载以及当前设备
- `http://127.0.0.1:3236/text2audio?token=<令牌>&tex=卧槽小米怎么这么坏啊` —— 快速检验语音输出

主要端点：

- `GET /text2audio`
- `POST /text2audio`
- `GET /health`

重要的查询/表单参数：

| 名称         | 类型    | 描述                                                                       |
|--------------|---------|----------------------------------------------------------------------------|
| `tex`/`text` | string  | 待合成的文本。双重URL编码的值会被自动解码。                               |
| `speakSpeed` | float   | 偏好的语速，范围为5-50。                                                  |
| `spd`        | float   | 语速变体。转换为相同的5-50范围。                                      |
| `voice`      | string  | 可选的Kokoro声音配置文件（例如 `zf_001`）。                               |
| `device`     | string  | `cpu`、`cuda`或`auto`之一。默认为`KOKORO_DEVICE`环境变量或`auto`。        |
| `token`      | string  | 当客户端无法设置请求头时，可通过查询参数携带访问令牌。                   |

响应以 `audio/wav` 流形式返回，包含以下头部信息：

- `X-RTF`：合成过程的实时因子
- `X-Sample-Rate`：音频采样率（24 kHz）
- `X-Device`：实际用于推理的设备
- 调用接口必须携带有效的访问令牌，可通过请求头（默认 `X-API-Token`）
  或在 URL 末尾追加 `?token=<令牌>` 的方式提供。

如果请求 CUDA 但不可用，或者推理过程中发生 CUDA 运行时错误，服务会自动重试 CPU。

## 监控和日志

- 运行时日志写入 `logs/tts_service.log`。文件在达到 5MB 时轮换，保留三个备份，防止无限增长。可通过环境变量调整（见下文）。
- 轻量级资源监控器每 10 秒记录 CPU 使用情况和（当 CUDA 可用时）GPU 内存。可通过 `KOKORO_MONITOR_INTERVAL` 更改采样频率。

## 配置环境变量

| 变量名                       | 默认值     | 用途                                                         |
|-----------------------------|-----------|--------------------------------------------------------------|
| `KOKORO_DEVICE`             | `auto`    | 启动时对 `cpu`、`cuda` 或 `auto` 的偏好设置|
| `KOKORO_MONITOR_INTERVAL`   | `10`      | 资源监控器采样间隔（秒|
| `KOKORO_LOG_MAX_BYTES`      | `5000000` | 日志文件轮换前的最大大小|
| `KOKORO_LOG_BACKUP_COUNT`   | `3`       | 保留的轮换日志文件数量|
| `KOKORO_API_TOKEN`          | _(未设置)_ | 允许访问服务的令牌列表|
| `KOKORO_TOKEN_HEADER`       | `X-API-Token` | 携带令牌的请求头名称|

## 示例请求

### PowerShell

```powershell
Invoke-WebRequest -Method POST `
  -Uri "http://127.0.0.1:3236/text2audio" `
  -Headers @{ "X-API-Token" = "<你的令牌>" } `
  -Body "tex=卧槽小米怎么这么坏啊&voice=zf_001&device=cuda" `
  -ContentType "application/x-www-form-urlencoded" `
  -OutFile "demo.wav"
```

### Bash/zsh

```bash
# 使用 curl 通过 HTTP POST 调用 /text2audio 并把输出保存为 demo.wav
curl -X POST 'http://127.0.0.1:3236/text2audio' \
  -H 'X-API-Token: <你的令牌>' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'tex=卧槽小米怎么这么坏啊' \
  --data 'voice=zf_001&device=cuda' \
  --output demo.wav
```

### URL

```text
http://127.0.0.1:3236/text2audio?token=<你的令牌>&tex=卧槽小米怎么这么坏啊
```

检查响应头来查看 `X-RTF` 和 `X-Device` 的值。

## Legado（阅读 3.0）配置示例

在阅读 3.0 App 的“朗读引擎”中新增自定义配置，可参考以下 JSON 并粘贴源：

```json
{
  "concurrentRate": "",
  "contentType": "audio/wav",
  "enabledCookieJar": false,
  "header": "Content-Type: application/x-www-form-urlencoded",
  "id": -100,
  "lastUpdateTime": 1762950914174,
  "loginCheckJs": "",
  "loginUi": "",
  "loginUrl": "",
  "name": "Kokoro",
  "url": "http://<服务器IP或域名>:3236/text2audio?token=<你的令牌>,{\"method\":\"POST\",\"body\":\"tex={{java.encodeURI(java.encodeURI(speakText))}}&speakSpeed={{speakSpeed}}&voice=zf_001\"}"
}
```

说明：

- `speakSpeed` 对应阅读的“语速”滑条，服务端会自动映射到 5-50 的范围
- `voice` 参数可以替换为 `voices/` 目录下其他可用声线的名称（需要对应语言）
