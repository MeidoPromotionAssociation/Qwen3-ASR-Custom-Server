# Qwen3-ASR HTTP Service

为 [COM3D2 Translate Tool](https://github.com/MeidoPromotionAssociation/COM3D2_TRANSLATE_TOOL) 准备的 ASR 服务器。

本项目 99% 由 GPT5.4 编写。

<br>
<br>
<br>

一个最小可运行的 `Qwen/Qwen3-ASR-1.7B` HTTP 服务示例，使用：

- `transformers` 后端
- `uv` 管理 Python 环境
- `FastAPI` 提供 HTTP 接口
- 单文件转写
- 批量转写
- 可选启用 `Qwen/Qwen3-ForcedAligner-0.6B` 返回时间戳

## 接口概览

- `GET /healthz`
  - 健康检查
- `POST /v1/audio/transcriptions`
  - 单文件转写
  - 可选 `return_timestamps=true`
- `POST /v1/audio/transcriptions/batch`
  - 批量转写
  - 可选 `return_timestamps=true`
- `POST /v1/audio/alignments`
  - 使用 `ForcedAligner` 对“已有文本”和音频做强制对齐，直接返回时间戳

默认 ASR 模型是 `Qwen/Qwen3-ASR-1.7B`。  
`ForcedAligner` 默认不启用，需要显式设置环境变量。

## 什么时候用哪种接口

### 1. 只有音频，没有歌词文本

用：

- `POST /v1/audio/transcriptions`

如果你还想拿时间戳：

- 启动服务时加载 `ForcedAligner`
- 请求时加 `return_timestamps=true`

这条路径适合：

- 自动转录歌词
- 自动转录歌曲片段
- 自动转录语音并顺带拿时间戳

### 2. 已经有准确的歌词文本，只想把歌词对齐到音频

用：

- `POST /v1/audio/alignments`

这条路径适合：

- 你已经有标准歌词文本
- 你想拿更稳定的逐词或逐字时间戳
- 你不想依赖 ASR 先识别出文本

对歌词场景来说，如果你已经有原始歌词文本，通常这条路径更稳。

## 返回格式

### 1. 转写接口

不带时间戳时：

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "aligner_model": null,
  "file_name": "test.wav"
}
```

带时间戳时：

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
  "file_name": "test.wav",
  "timestamps": [
    {
      "text": "你好",
      "start_time": 0.12,
      "end_time": 0.56
    }
  ]
}
```

### 2. 强制对齐接口

```json
{
  "text": "我已提供的歌词文本",
  "language": "Chinese",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
  "file_name": "song.wav",
  "timestamps": [
    {
      "text": "我",
      "start_time": 0.11,
      "end_time": 0.18
    }
  ]
}
```

时间戳单位是秒。

## Python 版本

本项目固定使用：

- Python `3.12`

`pyproject.toml` 里也已经限制成了：

- `>=3.12,<3.13`

## 目录结构

```text
.
├── app.py
├── pyproject.toml
└── README.md
```

## Linux

下面的 Linux 说明适用于：

- Linux 服务器
- 本地 Linux 开发机
- 大多数 bash / zsh 环境

### Linux 前置条件

至少准备好：

- Python `3.12`
- `uv`
- 一个可写目录用于项目和缓存
- 把仓库下载到本地（点击 CODE 按钮选择 Download Zip，然后自己找个目录解压，或者 `git clone https://github.com/MeidoPromotionAssociation/Qwen3-ASR-Custom-Server.git`）

如果要跑 GPU，还需要：

- NVIDIA 驱动
- 与 PyTorch 对应的 CUDA 环境

### Linux 1. 创建环境

在项目目录中执行：

```bash
uv venv --python 3.12
uv pip install -e .
```

### Linux 2. 安装 PyTorch

#### Linux GPU

下面示例使用 CUDA 12.8。  
如果你的 CUDA 版本不同，请以 PyTorch 官方安装页面为准：

- https://pytorch.org/get-started/locally/

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 GPU 是否可见：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

#### Linux CPU

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

检查 PyTorch：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Linux 3. 可选：提前下载模型

默认情况下，服务第一次启动时会自动下载模型。

#### 从 Hugging Face 下载

```bash
uv pip install "huggingface_hub[cli]"
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
uv run huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B
```

#### 从 ModelScope 下载

```bash
uv pip install modelscope
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
uv run modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
```

如果你下载到了本地目录，后面启动时可以这样指定：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-1.7B
QWEN_ALIGNER_MODEL=./models/Qwen3-ForcedAligner-0.6B
```

### Linux 4. 启动服务

#### Linux GPU，只做转写

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Linux GPU，转写并支持时间戳

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B \
QWEN_ALIGNER_DTYPE=float16 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

如果你已经安装了 FlashAttention 2，并且系统是 Linux，也可以尝试：

```bash
QWEN_ASR_ATTN_IMPLEMENTATION=flash_attention_2 \
QWEN_ALIGNER_ATTN_IMPLEMENTATION=flash_attention_2 \
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B \
QWEN_ALIGNER_DTYPE=float16 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Linux CPU，只做转写

```bash
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float32 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=2 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
QWEN_ASR_TMPDIR=/dev/shm \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Linux CPU，转写并支持时间戳

```bash
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float32 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B \
QWEN_ALIGNER_DTYPE=float32 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=1 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
QWEN_ASR_TMPDIR=/dev/shm \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- 时间戳模式会额外加载一个对齐模型，占用更多显存或内存
- CPU 上启用时间戳会明显更慢
- 歌词或歌曲场景建议显式传 `language`
- 如果你只是为了先验证链路，建议先跑“不带时间戳”的版本

### Linux 5. 验证服务

#### 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

如果已经启用了 `ForcedAligner`，你会看到：

```json
{
  "ok": true,
  "model": "Qwen/Qwen3-ASR-1.7B",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
  "device": "cuda:0",
  "dtype": "float16",
  "aligner_dtype": "float16",
  "max_batch_size": 4,
  "max_new_tokens": 256,
  "max_upload_mb": 100
}
```

#### 自动转录歌词并返回时间戳

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./song.wav" \
  -F "language=Chinese" \
  -F "return_timestamps=true"
```

#### 自动转录歌词但不返回时间戳

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./song.wav" \
  -F "language=Chinese"
```

#### 已有歌词文本，直接做强制对齐

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/alignments \
  -F "file=@./song.wav" \
  -F "language=Chinese" \
  -F "text=这里填写你已经准备好的歌词全文"
```

#### 批量转写并返回时间戳

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese" \
  -F "return_timestamps=true"
```

## Windows

下面的 Windows 说明默认使用 PowerShell。

### Windows 前置条件

至少准备好：

- Windows 10 或 Windows 11
- Python `3.12`
- `uv`
- PowerShell
- 把仓库下载到本地（点击 CODE 按钮选择 Download Zip，然后自己找个目录解压，或者 `git clone https://github.com/MeidoPromotionAssociation/Qwen3-ASR-Custom-Server.git`）

如果要跑 GPU，还需要：

- NVIDIA 显卡
- 最新可用驱动
- 与 PyTorch 对应的 CUDA 环境

### Windows 1. 创建环境

```powershell
uv venv --python 3.12
uv pip install -e .
```

如果 PowerShell 阻止脚本执行，可以先临时放开当前会话：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Windows 2. 安装 PyTorch

#### Windows GPU

下面示例仍然使用 CUDA 12.8。  
如果你的环境不是这个版本，请以 PyTorch 官方安装页为准：

- https://pytorch.org/get-started/locally/

```powershell
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 GPU：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

#### Windows CPU

```powershell
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

检查 PyTorch：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Windows 3. 可选：提前下载模型

#### 从 Hugging Face 下载

```powershell
uv pip install "huggingface_hub[cli]"
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir .\models\Qwen3-ASR-1.7B
uv run huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir .\models\Qwen3-ForcedAligner-0.6B
```

#### 从 ModelScope 下载

```powershell
uv pip install modelscope
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir .\models\Qwen3-ASR-1.7B
uv run modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir .\models\Qwen3-ForcedAligner-0.6B
```

### Windows 4. 启动服务

#### Windows GPU，只做转写

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Windows GPU，转写并支持时间戳

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
$env:QWEN_ALIGNER_DTYPE = "float16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Windows CPU，只做转写

```powershell
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ASR_THREADS = "16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "1"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

#### Windows CPU，转写并支持时间戳

```powershell
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
$env:QWEN_ALIGNER_DTYPE = "float32"
$env:QWEN_ASR_THREADS = "16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "1"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- Windows 上也可以跑 `ForcedAligner`
- 但时间戳模式会更慢，也更吃内存或显存
- 如果只是为了先跑通服务，建议先不启用 `ForcedAligner`

### Windows 5. 验证服务

PowerShell 中建议使用 `curl.exe`，避免和 PowerShell 自己的 `curl` 别名混淆。

#### 健康检查

```powershell
curl.exe http://127.0.0.1:8000/healthz
```

#### 自动转录歌词并返回时间戳

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@.\song.wav" `
  -F "language=Chinese" `
  -F "return_timestamps=true"
```

#### 已有歌词文本，直接做强制对齐

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/alignments `
  -F "file=@.\song.wav" `
  -F "language=Chinese" `
  -F "text=这里填写你已经准备好的歌词全文"
```

#### 批量转写并返回时间戳

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch `
  -F "files=@.\a.wav" `
  -F "files=@.\b.wav" `
  -F "language=Chinese" `
  -F "return_timestamps=true"
```

## 常用环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR 模型 ID 或本地目录 |
| `QWEN_ALIGNER_MODEL` | 空 | `ForcedAligner` 模型 ID 或本地目录 |
| `QWEN_ASR_DEVICE` | 自动检测 | `cuda:0` 或 `cpu` |
| `QWEN_ASR_DTYPE` | GPU=`float16` / CPU=`float32` | ASR 推理精度 |
| `QWEN_ALIGNER_DTYPE` | 跟随 `QWEN_ASR_DTYPE` | 对齐模型推理精度 |
| `QWEN_ASR_THREADS` | 逻辑 CPU 数的一半 | 仅 CPU 模式使用 |
| `QWEN_ASR_MAX_BATCH_SIZE` | GPU=`4` / CPU=`2` | 推理 batch 大小 |
| `QWEN_ASR_MAX_NEW_TOKENS` | `256` | 最大生成 token 数 |
| `QWEN_ASR_MAX_UPLOAD_MB` | `100` | 单文件上传大小限制 |
| `QWEN_ASR_TMPDIR` | 系统默认临时目录 | 上传文件缓存目录 |
| `QWEN_ASR_HOST` | `0.0.0.0` | 监听地址 |
| `QWEN_ASR_PORT` | `8000` | 监听端口 |
| `QWEN_ASR_ATTN_IMPLEMENTATION` | 空 | 可选，例如 `flash_attention_2` |
| `QWEN_ALIGNER_ATTN_IMPLEMENTATION` | 空 | 对齐模型的可选 attention 实现 |

## 歌词和时间戳的实用建议

- 如果你只有歌曲音频，没有现成歌词：
  - 用 `/v1/audio/transcriptions`
  - 加 `return_timestamps=true`
- 如果你已经有歌词全文：
  - 用 `/v1/audio/alignments`
  - 这通常会比“先 ASR 再对齐”更稳
- 已知语言时尽量显式传 `language`
- 时间戳模式会更慢，也会更占内存或显存
- 在 CPU 上跑时间戳会明显慢于 GPU

## 参考

- Qwen3-ASR 项目主页：https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR README：https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md
- PyTorch 安装指南：https://pytorch.org/get-started/locally/
- uv 文档：https://docs.astral.sh/uv/
