# Qwen3-ASR HTTP Service

一个最小可运行的 `Qwen/Qwen3-ASR-1.7B` HTTP 服务示例，使用：

- `transformers` 后端
- `uv` 管理 Python 环境
- `FastAPI` 提供 HTTP 接口
- 单文件转写和批量转写

这个项目的目标很简单：让别人照着 README 做，就能启动一个本地或服务器上的 ASR HTTP 服务。

## 功能

- `GET /healthz`
  - 健康检查
- `POST /v1/audio/transcriptions`
  - 单文件转写
- `POST /v1/audio/transcriptions/batch`
  - 批量转写

默认模型是 `Qwen/Qwen3-ASR-1.7B`。

## 目录结构

```text
.
├── app.py
├── pyproject.toml
└── README.md
```

## 使用前先看

- 这个项目默认使用 Hugging Face 上的 `Qwen/Qwen3-ASR-1.7B`
- 第一次启动时，如果本地没有模型文件，会自动下载
- CPU 也能跑 `1.7B`，但会明显比 GPU 慢
- 这个示例默认只返回 `text` 和 `language`
- 这个示例没有接入 `ForcedAligner`，所以不返回时间戳
- 这个示例没有做流式识别
- 服务内部默认串行执行推理；如果你需要更高吞吐，请优先使用批量接口，而不是直接开多个进程

## 请求和返回格式

### 1. 单文件转写

请求：

- 方法：`POST`
- 路径：`/v1/audio/transcriptions`
- `multipart/form-data` 字段：
  - `file`
    - 必填，音频文件
  - `language`
    - 可选，例如 `Chinese`、`English`
  - `prompt`
    - 可选，附加上下文提示

示例返回：

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "file_name": "test.wav"
}
```

### 2. 批量转写

请求：

- 方法：`POST`
- 路径：`/v1/audio/transcriptions/batch`
- `multipart/form-data` 字段：
  - `files`
    - 必填，可重复上传多个文件
  - `language`
    - 可选，可以只传一个值，也可以按文件数量重复传多个值
  - `prompt`
    - 可选，可以只传一个值，也可以按文件数量重复传多个值

示例返回：

```json
{
  "model": "Qwen/Qwen3-ASR-1.7B",
  "results": [
    {
      "index": 0,
      "file_name": "a.wav",
      "language": "Chinese",
      "text": "第一条转写结果"
    },
    {
      "index": 1,
      "file_name": "b.wav",
      "language": "English",
      "text": "Second transcription result"
    }
  ]
}
```

## Linux

下面的 Linux 说明同时适用于：

- Linux 服务器
- 本地 Linux 开发机
- 大多数基于 bash 的环境

### Linux 前置条件

你至少需要准备好：

- Python `3.12`
- `uv`
- 一个可写目录用于项目和缓存
- 把仓库下载到本地（点击 CODE 按钮选择 Download Zip，然后自己找个目录解压，或者 `git clone https://github.com/MeidoPromotionAssociation/Qwen3-ASR-Custom-Server.git`）

如果你打算在 Linux 上用 GPU，还需要：

- NVIDIA 驱动
- 可正常工作的 CUDA 运行环境

如果你打算在 Linux 上用 CPU，额外建议：

- CPU 核数尽量足够多
- 有较大的内存

### Linux 1. 安装 uv

参考官方文档：

- https://docs.astral.sh/uv/

安装完成后，在项目目录中执行：

```bash
uv venv --python 3.12
```

如果你愿意，也可以激活虚拟环境：

```bash
source .venv/bin/activate
```

但这不是必须的，因为下面所有命令都可以直接使用 `uv run` 或 `uv pip`。

### Linux 2. 安装项目依赖

先安装当前项目依赖：

```bash
uv pip install -e .
```

### Linux 3. 选择 GPU 或 CPU 安装 PyTorch

#### Linux GPU

下面示例使用 CUDA 12.8。如果你的机器使用的是其他 CUDA 版本，请以 PyTorch 官方安装页面为准：

- https://pytorch.org/get-started/locally/

安装：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 CUDA 是否可见：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出里 `torch.cuda.is_available()` 是 `True`，说明 GPU 可以被 PyTorch 使用。

#### Linux CPU

安装 CPU 版 PyTorch：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

检查 PyTorch：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU 模式下最后一项通常会输出 `False`，这是正常的。

### Linux 4. 可选：提前下载模型到本地

默认情况下，服务第一次启动时会自动下载 `Qwen/Qwen3-ASR-1.7B`。

如果你想先把模型下载到本地，再从本地目录加载，可以这样做。

#### Linux 4.1 从 Hugging Face 下载

```bash
uv pip install "huggingface_hub[cli]"
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
```

后面启动服务时加上：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-1.7B
```

#### Linux 4.2 从 ModelScope 下载

如果你在中国大陆环境里更方便使用 ModelScope：

```bash
uv pip install modelscope
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
```

后面启动服务时同样加上：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-1.7B
```

### Linux 5. 启动服务

#### Linux GPU 启动

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

参数说明：

- `QWEN_ASR_DEVICE=cuda:0`
  - 使用第一张 GPU
- `QWEN_ASR_DTYPE=float16`
  - GPU 上更常用，也更省显存
- `QWEN_ASR_MAX_BATCH_SIZE=4`
  - 先用保守值，显存不足时降到 `1` 或 `2`
- `QWEN_ASR_MAX_NEW_TOKENS=256`
  - 对常见语音转写通常够用

如果你已经在 Linux 上额外安装了 FlashAttention 2，也可以尝试：

```bash
QWEN_ASR_ATTN_IMPLEMENTATION=flash_attention_2 \
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

但这不是必须项。

#### Linux CPU 启动

```bash
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float32 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=2 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
QWEN_ASR_TMPDIR=/dev/shm \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

参数说明：

- `QWEN_ASR_THREADS`
  - 建议优先设置为物理核数，而不是逻辑线程数
- `QWEN_ASR_MAX_BATCH_SIZE=2`
  - CPU 上更稳的起点
- `QWEN_ASR_TMPDIR=/dev/shm`
  - 使用内存盘存放上传的临时文件，减少磁盘 I/O

如果你的 CPU 较弱，或者内存压力较大，可以进一步调低：

```bash
QWEN_ASR_MAX_BATCH_SIZE=1
QWEN_ASR_MAX_NEW_TOKENS=128
```

### Linux 6. 验证服务

#### Linux 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

示例返回：

```json
{
  "ok": true,
  "model": "Qwen/Qwen3-ASR-1.7B",
  "device": "cpu",
  "dtype": "float32",
  "max_batch_size": 2,
  "max_new_tokens": 256,
  "max_upload_mb": 100
}
```

#### Linux 单文件转写

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav" \
  -F "language=Chinese"
```

如果你不传 `language`，模型会自行识别语言：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav"
```

如果你想提供上下文提示，也可以传 `prompt`：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav" \
  -F "language=Chinese" \
  -F "prompt=这是一个客服录音，里面可能出现订单号和商品名。"
```

#### Linux 批量转写

如果多个文件使用同一个语言：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese"
```

如果不同文件使用不同语言或不同提示词：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese" \
  -F "language=English" \
  -F "prompt=这是中文会议录音。" \
  -F "prompt=This is an English meeting recording."
```

## Windows

下面的 Windows 说明默认使用 PowerShell。

### Windows 前置条件

你至少需要准备好：

- Windows 10 或 Windows 11
- Python `3.12`
- `uv`
- PowerShell
- 把仓库下载到本地（点击 CODE 按钮选择 Download Zip，然后自己找个目录解压，或者 `git clone https://github.com/MeidoPromotionAssociation/Qwen3-ASR-Custom-Server.git`）

如果你打算在 Windows 上用 GPU，还需要：

- NVIDIA 显卡
- 最新可用驱动
- 与 PyTorch 匹配的 CUDA 运行环境

### Windows 1. 安装 uv

参考官方文档：

- https://docs.astral.sh/uv/

在项目目录中创建虚拟环境：

```powershell
uv venv --python 3.12
```

如果你需要激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本执行，可以先临时放开当前会话：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Windows 2. 安装项目依赖

```powershell
uv pip install -e .
```

### Windows 3. 选择 GPU 或 CPU 安装 PyTorch

#### Windows GPU

下面示例使用 CUDA 12.8。实际命令请优先以 PyTorch 官方安装页面为准：

- https://pytorch.org/get-started/locally/

```powershell
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 GPU 是否可见：

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

### Windows 4. 可选：提前下载模型到本地

#### Windows 4.1 从 Hugging Face 下载

```powershell
uv pip install "huggingface_hub[cli]"
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir .\models\Qwen3-ASR-1.7B
```

启动时指定本地模型目录：

```powershell
$env:QWEN_ASR_MODEL = ".\models\Qwen3-ASR-1.7B"
```

#### Windows 4.2 从 ModelScope 下载

```powershell
uv pip install modelscope
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir .\models\Qwen3-ASR-1.7B
```

启动时指定本地模型目录：

```powershell
$env:QWEN_ASR_MODEL = ".\models\Qwen3-ASR-1.7B"
```

### Windows 5. 启动服务

#### Windows GPU 启动

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- `QWEN_ASR_DEVICE = "cuda:0"`
  - 使用第一张 GPU
- `QWEN_ASR_DTYPE = "float16"`
  - 更符合 Windows GPU 的常见用法
- `QWEN_ASR_MAX_BATCH_SIZE = "4"`
  - 先用较保守的批大小

Windows 上不建议把 FlashAttention 2 作为默认前置条件，因此 README 不把它作为标准步骤。

#### Windows CPU 启动

```powershell
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ASR_THREADS = "16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "1"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- Windows CPU 跑 `1.7B` 会比 Linux CPU 更容易遇到性能瓶颈
- 如果只是验证链路，可以先把 `QWEN_ASR_MAX_NEW_TOKENS` 降到 `128`
- 如果内存压力较大，可以把 `QWEN_ASR_MAX_BATCH_SIZE` 固定为 `1`

### Windows 6. 验证服务

PowerShell 里建议直接用 `curl.exe`，避免和 PowerShell 自己的 `curl` 别名混淆。

#### Windows 健康检查

```powershell
curl.exe http://127.0.0.1:8000/healthz
```

#### Windows 单文件转写

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@.\test.wav" `
  -F "language=Chinese"
```

不传语言：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@.\test.wav"
```

带提示词：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@.\test.wav" `
  -F "language=Chinese" `
  -F "prompt=这是一个客服录音，里面可能出现订单号和商品名。"
```

#### Windows 批量转写

同一语言：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch `
  -F "files=@.\a.wav" `
  -F "files=@.\b.wav" `
  -F "language=Chinese"
```

不同语言和不同提示词：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch `
  -F "files=@.\a.wav" `
  -F "files=@.\b.wav" `
  -F "language=Chinese" `
  -F "language=English" `
  -F "prompt=这是中文会议录音。" `
  -F "prompt=This is an English meeting recording."
```

## 常用环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | 模型 ID 或本地目录 |
| `QWEN_ASR_DEVICE` | 自动检测 | `cuda:0` 或 `cpu` |
| `QWEN_ASR_DTYPE` | GPU=`float16` / CPU=`float32` | 推理精度 |
| `QWEN_ASR_THREADS` | 逻辑 CPU 数的一半 | 仅 CPU 模式使用 |
| `QWEN_ASR_MAX_BATCH_SIZE` | GPU=`4` / CPU=`2` | 推理 batch 大小 |
| `QWEN_ASR_MAX_NEW_TOKENS` | `256` | 最大生成 token 数 |
| `QWEN_ASR_MAX_UPLOAD_MB` | `100` | 单文件上传大小限制 |
| `QWEN_ASR_TMPDIR` | 系统默认临时目录 | 上传文件缓存目录 |
| `QWEN_ASR_HOST` | `0.0.0.0` | 监听地址 |
| `QWEN_ASR_PORT` | `8000` | 监听端口 |
| `QWEN_ASR_ATTN_IMPLEMENTATION` | 空 | 可选，例如 `flash_attention_2` |

## 常见建议

- 已知语言时尽量传 `language`
  - 常见值如 `Chinese`、`English`
- 如果只是为了先跑通链路，优先用 GPU
- 如果要跑 CPU，先从更小的 batch 开始
- 如果转写速度慢，优先尝试：
  - 降低 `QWEN_ASR_MAX_BATCH_SIZE`
  - 降低 `QWEN_ASR_MAX_NEW_TOKENS`
  - 在 CPU 上调整 `QWEN_ASR_THREADS`
- 如果上传的音频较大，可以适当调大：
  - `QWEN_ASR_MAX_UPLOAD_MB`

## 参考

- Qwen3-ASR 项目主页：https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR README：https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md
- PyTorch 安装指南：https://pytorch.org/get-started/locally/
- uv 文档：https://docs.astral.sh/uv/
