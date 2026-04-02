# Qwen3-ASR HTTP Service

一个最小可运行的 `Qwen/Qwen3-ASR-1.7B` HTTP 服务示例，使用：

- `transformers` 后端
- `uv` 管理 Python 环境
- `FastAPI` 提供 HTTP 接口
- 单文件转写和批量转写

这个项目的目标很简单：让别人照着 README 做，就能启动一个本地或服务器上的 ASR HTTP 服务。

## 接口

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

## 1. 准备环境

### 1.1 安装 uv

先安装 `uv`：

- 官方文档：https://docs.astral.sh/uv/

安装完成后，在当前目录执行：

```bash
uv venv --python 3.11
```

后面的命令默认都在这个目录里执行。

### 1.2 安装项目依赖

先安装项目本身依赖：

```bash
uv pip install -e .
```

然后根据你的运行环境，安装对应的 PyTorch。

## 2. GPU 运行

### 2.1 NVIDIA GPU

下面示例使用 CUDA 12.8 的 PyTorch wheel。如果你的环境不是 CUDA 12.8，请改成 PyTorch 官方安装页面提供的命令：

- https://pytorch.org/get-started/locally/

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

启动服务：

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- `QWEN_ASR_DEVICE=cuda:0`：使用第一张 GPU
- `QWEN_ASR_DTYPE=float16`：GPU 上默认推荐 `float16`
- `QWEN_ASR_MAX_BATCH_SIZE=4`：先用保守值，显存更紧张时可以降到 `1` 或 `2`
- `QWEN_ASR_MAX_NEW_TOKENS=256`：对大多数常见语音转写够用，太大只会更慢

如果你已经额外安装了 FlashAttention 2，并且系统是 Linux，可以再加：

```bash
QWEN_ASR_ATTN_IMPLEMENTATION=flash_attention_2
```

但它不是必须项。

## 3. CPU 运行

CPU 也可以直接跑 `1.7B`，只是会明显慢很多。

安装 CPU 版 PyTorch：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

启动服务：

```bash
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float32 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=2 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ASR_THREADS = "32"
$env:QWEN_ASR_MAX_BATCH_SIZE = "2"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run qwen3-asr-http --host 0.0.0.0 --port 8000
```

说明：

- `QWEN_ASR_THREADS` 建议设成物理核数，而不是逻辑线程数
- `QWEN_ASR_MAX_BATCH_SIZE=2` 是更稳的 CPU 起点
- 如果是双路/NUMA 机器，建议自己做压测再决定是否继续加线程或加 batch

在 Linux 上，如果想减少上传文件的磁盘 I/O，可以把临时目录放到内存盘：

```bash
QWEN_ASR_TMPDIR=/dev/shm
```

## 4. 可选：提前下载模型到本地

默认情况下，服务第一次启动时会自动下载 `Qwen/Qwen3-ASR-1.7B`。

如果你想提前下载到本地目录，再让服务从本地加载：

### 4.1 Hugging Face

```bash
uv pip install "huggingface_hub[cli]"
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
```

然后启动服务时指定：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-1.7B uv run qwen3-asr-http
```

### 4.2 ModelScope

如果你在中国大陆环境里更方便使用 ModelScope：

```bash
uv pip install modelscope
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
```

然后：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-1.7B uv run qwen3-asr-http
```

## 5. 验证服务是否启动

### 5.1 健康检查

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

### 5.2 单文件转写

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav" \
  -F "language=Chinese"
```

也可以不传 `language`，让模型自己识别语言：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav"
```

如果你有上下文提示，也可以传 `prompt`：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./test.wav" \
  -F "language=Chinese" \
  -F "prompt=这是一个客服录音，里面可能出现订单号和商品名。"
```

示例返回：

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "file_name": "test.wav"
}
```

### 5.3 批量转写

批量接口使用重复的 `files` 字段上传多个音频文件：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese"
```

如果不同文件需要不同提示词或不同语言，可以重复传多个 `prompt` / `language` 字段，数量要和文件数一致：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese" \
  -F "language=English" \
  -F "prompt=这是中文会议录音。" \
  -F "prompt=This is an English meeting recording."
```

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

## 6. 常用环境变量

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

## 7. 说明和限制

- 这个示例默认只返回 `text` 和 `language`
- 这个示例没有接入 `ForcedAligner`，所以不返回时间戳
- 这个示例没有做流式识别
- 为了保持实现简单，服务内部默认串行执行推理；如果你需要更高吞吐，请从批量接口开始，而不是直接开多个进程
- `Qwen/Qwen3-ASR-1.7B` 在 CPU 上可以跑，但会比较慢；如果你只是验证链路，建议优先在 GPU 上使用

## 8. 参考

- Qwen3-ASR 项目主页：https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR README：https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md
- PyTorch 安装指南：https://pytorch.org/get-started/locally/
- uv 文档：https://docs.astral.sh/uv/
