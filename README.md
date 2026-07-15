# Qwen3-ASR HTTP Service

为 [COM3D2 Translate Tool](https://github.com/MeidoPromotionAssociation/COM3D2_TRANSLATE_TOOL) 准备的 Qwen3-ASR HTTP 服务。

本项目 99% 由 GPT-5.6-sol Utral 编写。

2026-07-15 服务现已从旧的专用 Python 后端迁移到 Hugging Face Transformers 原生 Qwen3-ASR 实现。HTTP 路径、表单参数和转写/对齐响应结构保持不变，调用方不需要因为后端迁移而更改接口。

当前 Transformers 尚未发布包含该实现的正式版本，因此依赖被固定到源码提交：

```text
https://github.com/huggingface/transformers.git@498d6e984e84d29186e671656817a53a024af930
```

这样既能使用原生的 `AutoProcessor`、Qwen3-ASR 生成模型和 Forced Aligner，又能避免上游源码变化造成不可重复安装。

## 核心结论

- Transformers 原生 Qwen3-ASR 可以在 CPU 上运行，不要求 CUDA。
- 服务默认 ASR 模型是 `Qwen/Qwen3-ASR-1.7B-hf`。
- CPU 部署推荐改用 `Qwen/Qwen3-ASR-0.6B-hf`、`float16`、batch size 1；它比 1.7B 更适合延迟和内存受限的机器。
- 当前项目的默认精度策略是 CPU=`float16`、CUDA/GPU=`float32`，Aligner 默认跟随 ASR 精度。
- 强制对齐模型是可选的；需要时间戳时设置 `QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf`。
- CPU 也能运行 Forced Aligner，但会额外占用内存，而且通常明显慢于 GPU。
- `torch.compile` 默认关闭。开启后，每个模型会在第一次实际推理请求时触发编译，因此第一次请求会显著更慢。

## 功能和接口

- `GET /healthz`
  - 健康检查及当前后端、模型、设备、精度和编译配置
- `POST /v1/audio/transcriptions`
  - 单文件转写
  - 可选 `language`、`prompt` 和 `return_timestamps=true`
- `POST /v1/audio/transcriptions/batch`
  - 批量转写
  - 可选为每个文件提供 `language` 和 `prompt`
  - 可选 `return_timestamps=true`
- `POST /v1/audio/alignments`
  - 使用 Forced Aligner 对“已有文本”和音频做强制对齐
  - 直接返回逐词或逐字时间戳

`QWEN_ALIGNER_MODEL` 默认留空，因此时间戳功能默认不加载。未启用对齐模型时，请求 `return_timestamps=true` 或调用 `/v1/audio/alignments` 会返回 HTTP 400。

## 什么时候使用哪个接口

### 只有音频，没有准确文本

使用 `POST /v1/audio/transcriptions`。如果还需要时间戳：

1. 启动服务时设置 `QWEN_ALIGNER_MODEL`。
2. 请求时提交 `return_timestamps=true`。

这适用于自动转录歌词、歌曲片段和普通语音。

### 已有准确文本，只需要对齐到音频

使用 `POST /v1/audio/alignments`。对歌词场景，如果已有标准歌词全文，这通常比“先 ASR、再对齐 ASR 结果”更稳定。

## CPU、GPU 和内存估算

### CPU 是否可用

可以。原生 Transformers 后端会在 `QWEN_ASR_DEVICE=cpu` 时使用 CPU 版 PyTorch。CPU 建议配置为：

```text
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf
QWEN_ASR_DEVICE=cpu
QWEN_ASR_DTYPE=float16
QWEN_ASR_MAX_BATCH_SIZE=1
```

本项目在 CPU 上默认并推荐先使用 `float16`，主要目的是把权重内存降到 `float32` 的约一半。FP16 在 CPU 上不一定更快：较旧的 CPU、部分 Windows PyTorch 构建或个别算子可能缺少高效的 Half 实现，表现为推理更慢或报“不支持 Half/float16 算子”的错误。遇到这种情况时，显式设置 `QWEN_ASR_DTYPE=float32`；如果启用了 Aligner，也设置 `QWEN_ALIGNER_DTYPE=float32`。FP32 兼容性更稳，但权重内存约翻倍。

默认的 1.7B 模型同样能在 CPU 上加载和运行，但会使用更多内存，生成延迟也更高。没有 CUDA 不代表不能运行，只代表需要接受较低的吞吐和较高的延迟。

GPU 按项目要求默认使用 `float32`。与 FP16/BF16 相比，它通常占用约两倍权重显存且推理更慢，但数值路径更保守；下面所有 GPU 启动示例均按这一默认策略编写。

### 权重和实际内存

下面是容量规划用的粗略估算，不是固定上限。权重下限按模型名中的名义参数量乘以每参数字节数计算，使用十进制 GB：`float16` 每参数约 2 字节，`float32` 每参数约 4 字节。

| ASR 配置 | 仅 ASR 权重理论值 | batch=1、短音频时的实际进程/显存规划值 |
|---|---:|---:|
| 0.6B + `float16`（CPU 推荐） | 约 1.2 GB | 约 3–6 GB RAM |
| 1.7B + `float16`（CPU） | 约 3.4 GB | 约 6–10 GB RAM |
| 0.6B + `float32`（CPU 兼容回退） | 约 2.4 GB | 约 4–8 GB RAM |
| 1.7B + `float32`（CPU 兼容回退） | 约 6.8 GB | 约 10–16 GB RAM |
| 0.6B + `float32`（GPU 默认） | 约 2.4 GB | 约 4–7 GB VRAM |
| 1.7B + `float32`（GPU 默认） | 约 6.8 GB | 约 9–14 GB VRAM |

启用 `Qwen/Qwen3-ForcedAligner-0.6B-hf` 后，还要常驻一个约 0.6B 参数的模型：

- `float16` 对齐权重理论值约 1.2 GB；`float32` 对照值约 2.4 GB。
- CPU 使用 0.6B ASR + 0.6B Aligner、两者均为 `float16` 时，权重理论值合计约 2.4 GB，建议先按 5–10 GB 的服务进程内存规划。
- CPU 使用 1.7B ASR + 0.6B Aligner、两者均为 `float16` 时，权重理论值合计约 4.6 GB，建议先按 8–14 GB 的服务进程内存规划。
- 如果因 CPU/PyTorch 兼容性回退到 `float32`，上面两种组合的权重理论值分别约为 4.8 GB 和 9.2 GB，实际进程还会在此基础上增加运行时开销。
- GPU 默认以 `float32` 加载 Aligner，因此启用时间戳还会增加约 2.4 GB 的对齐权重，实际新增显存高于这个权重值。

实际峰值还受音频时长、batch size、生成 token 数、KV cache、音频特征、PyTorch 运行时、对齐输入长度和 `torch.compile` 缓存影响。长音频或 batch 大于 1 时应留出更多余量，并以目标机器上的峰值 RSS/VRAM 实测为准。

## `torch.compile`

ASR 和 Forced Aligner 都支持按模型分别启用编译：

```text
QWEN_ASR_COMPILE=true
QWEN_ALIGNER_COMPILE=true
QWEN_COMPILE_MODE=
```

- `QWEN_ASR_COMPILE` 默认 `false`。
- `QWEN_ALIGNER_COMPILE` 默认 `false`。
- `QWEN_COMPILE_MODE` 默认留空；留空时不向 `torch.compile` 传 `mode`，由当前 PyTorch 使用自身默认模式。
- 常见显式值包括 `default`、`reduce-overhead` 和 `max-autotune`。不同 PyTorch/设备对模式的支持和收益可能不同。
- 编译是惰性的：ASR 或 Aligner 第一次实际执行时才会编译，因此相应模型的第一次请求会更慢并可能额外占用内存。
- 修改输入形状、batch 或执行路径可能触发重新编译。

官方给出的约 `2.4×` ASR generate 加速和约 `2.5×` Forced Aligner 加速，是在 **A100、batch size 4** 条件下观察到的数据；上游示例还使用了 BF16。它不是本项目 GPU FP32 默认路径、CPU、消费级显卡、batch size 1 或所有音频长度下的保证。服务因此默认关闭编译；建议先测不编译的延迟和内存，再单独开启 ASR/Aligner 编译并计入首次请求预热成本。

## Python 和系统要求

项目固定使用：

- Python `>=3.12,<3.13`
- `uv`
- Git（安装锁定的 Transformers 源码提交时需要）

Linux GPU 还需要 NVIDIA 驱动。PyTorch CUDA wheel 已通过 `cu128` extra 指向 CUDA 12.8 软件源；CPU wheel 通过 `cpu` extra 安装。

## 安装

先进入项目目录并创建 Python 3.12 环境：

```bash
uv venv --python 3.12
```

### CPU

Linux 或 Windows 均可执行：

```bash
uv sync --extra cpu
```

检查 PyTorch：

```bash
uv run --extra cpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

最后一项在 CPU 环境中应为 `False`，这不影响 CPU 推理。

### CUDA 12.8

```bash
uv sync --extra cu128
```

检查 GPU：

```bash
uv run --extra cu128 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`cpu` 和 `cu128` 是互斥 extra，不要同时启用。

### 日语、韩语强制对齐

Forced Aligner 的日语分词需要 `nagisa`，韩语分词需要 `soynlp`。它们不属于默认依赖，按需要安装 `align-ja-ko` extra：

```bash
uv sync --extra cpu --extra align-ja-ko
```

或在 CUDA 环境中：

```bash
uv sync --extra cu128 --extra align-ja-ko
```

其他受支持语言不需要这两个可选包。Forced Aligner 支持中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语和西班牙语。

## 模型下载

第一次启动时，Transformers 会自动从 Hugging Face 下载模型并使用本机缓存。也可以提前下载。

先确保 `uv` 使用 Python 3.12，避免 `uvx` 在旧服务器上自动选中系统 Python 3.6：

```bash
uv python install 3.12
uv run --no-project --python 3.12 python -V
```


```bash
uvx --python 3.12 --with "socksio" hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir ./models/Qwen3-ASR-1.7B-hf
```

```bash
uvx --python 3.12 --with "socksio" hf download Qwen/Qwen3-ForcedAligner-0.6B-hf --local-dir ./models/Qwen3-ForcedAligner-0.6B-hf
```

如果错误回溯中出现 `python3.6` 或 `lib/python3.6`，说明没有成功选中 Python 3.12；不要通过给 Python 3.6 补装 `dataclasses` 来绕过，因为当前 Transformers 和本项目都不再支持 Python 3.6。

指定本地目录时，环境变量写法如下：

```bash
QWEN_ASR_MODEL=./models/Qwen3-ASR-0.6B-hf
QWEN_ALIGNER_MODEL=./models/Qwen3-ForcedAligner-0.6B-hf
```

Windows PowerShell 使用：

```powershell
$env:QWEN_ASR_MODEL = ".\models\Qwen3-ASR-0.6B-hf"
$env:QWEN_ALIGNER_MODEL = ".\models\Qwen3-ForcedAligner-0.6B-hf"
```

## Linux 启动示例

### GPU：只做转写

不设置 `QWEN_ASR_MODEL` 时使用默认的 1.7B-hf：

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float32 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

### GPU：转写并支持时间戳

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float32 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf \
QWEN_ALIGNER_DTYPE=float32 \
QWEN_ASR_MAX_BATCH_SIZE=4 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

FlashAttention 2 不支持本项目的 GPU `float32` 默认精度。只有在 Linux 正确安装 FlashAttention 2，并且显式把 ASR 和 Aligner dtype 都改成 `float16` 或 `bfloat16` 时才能启用；下面示例使用 `float16`：

```bash
QWEN_ASR_ATTN_IMPLEMENTATION=flash_attention_2 \
QWEN_ALIGNER_ATTN_IMPLEMENTATION=flash_attention_2 \
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float16 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf \
QWEN_ALIGNER_DTYPE=float16 \
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

不能在 `QWEN_ASR_DTYPE=float32` 或 `QWEN_ALIGNER_DTYPE=float32` 时设置对应的 `flash_attention_2`。如果只为 ASR 启用 FlashAttention 2，至少要把 ASR dtype 改成半精度；如果 ASR 和 Aligner 都启用，则两者都要显式改成 `float16`/`bfloat16`。

### CPU：推荐配置，只做转写

```bash
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf \
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float16 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=1 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
QWEN_ASR_TMPDIR=/dev/shm \
uv run --extra cpu qwen3-asr-http --host 0.0.0.0 --port 8000
```

### CPU：转写并支持时间戳

```bash
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf \
QWEN_ASR_DEVICE=cpu \
QWEN_ASR_DTYPE=float16 \
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf \
QWEN_ALIGNER_DTYPE=float16 \
QWEN_ASR_THREADS=32 \
QWEN_ASR_MAX_BATCH_SIZE=1 \
QWEN_ASR_MAX_NEW_TOKENS=256 \
QWEN_ASR_TMPDIR=/dev/shm \
uv run --extra cpu qwen3-asr-http --host 0.0.0.0 --port 8000
```

如果只想验证链路，先不要加载 Aligner。CPU 时间戳模式会更慢，也会让常驻内存明显增加。

### Linux CPU 线程配置

`QWEN_ASR_THREADS` 会传给 `torch.set_num_threads(...)`，默认值是逻辑 CPU 数的一半。建议从物理核心数开始测试，并让 OpenMP/MKL 线程与它一致：

```bash
export QWEN_ASR_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export QWEN_ASR_MAX_BATCH_SIZE=1
```

主推理开销通常在 PyTorch。把 OpenBLAS 线程也拉满可能造成线程争抢，因此 `OPENBLAS_NUM_THREADS=1` 是较稳妥的起点。双路 NUMA 或延迟抖动明显时，应分别测试更小线程数和 batch size 1。

## Windows PowerShell 启动示例

### GPU：只做转写

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

### GPU：转写并支持时间戳

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
$env:QWEN_ALIGNER_DTYPE = "float32"
$env:QWEN_ASR_MAX_BATCH_SIZE = "4"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

### CPU：推荐配置，只做转写

```powershell
$env:QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ASR_THREADS = "16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "1"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run --extra cpu qwen3-asr-http --host 0.0.0.0 --port 8000
```

### CPU：转写并支持时间戳

```powershell
$env:QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
$env:QWEN_ASR_DEVICE = "cpu"
$env:QWEN_ASR_DTYPE = "float16"
$env:QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
$env:QWEN_ALIGNER_DTYPE = "float16"
$env:QWEN_ASR_THREADS = "16"
$env:QWEN_ASR_MAX_BATCH_SIZE = "1"
$env:QWEN_ASR_MAX_NEW_TOKENS = "256"
uv run --extra cpu qwen3-asr-http --host 0.0.0.0 --port 8000
```

PowerShell 会保留当前会话中已经设置的环境变量。切换示例前，如需恢复“未设置”状态，可以执行：

```powershell
Remove-Item Env:QWEN_ALIGNER_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:QWEN_ASR_COMPILE -ErrorAction SilentlyContinue
Remove-Item Env:QWEN_ALIGNER_COMPILE -ErrorAction SilentlyContinue
Remove-Item Env:QWEN_COMPILE_MODE -ErrorAction SilentlyContinue
```

## 可选启用编译

建议先用固定的一组实际音频测量未编译基线，再开启。例如 Linux GPU：

```bash
QWEN_ASR_DEVICE=cuda:0 \
QWEN_ASR_DTYPE=float32 \
QWEN_ASR_COMPILE=true \
QWEN_ALIGNER_COMPILE=false \
QWEN_COMPILE_MODE=reduce-overhead \
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

Windows PowerShell：

```powershell
$env:QWEN_ASR_DEVICE = "cuda:0"
$env:QWEN_ASR_DTYPE = "float32"
$env:QWEN_ASR_COMPILE = "true"
$env:QWEN_ALIGNER_COMPILE = "false"
$env:QWEN_COMPILE_MODE = "reduce-overhead"
uv run --extra cu128 qwen3-asr-http --host 0.0.0.0 --port 8000
```

如果同时加载 Forced Aligner，可独立设置 `QWEN_ALIGNER_COMPILE=true`。两个开关互不依赖；未加载 Aligner 时设置其编译开关没有实际作用。

## HTTP API 请求和输出格式

服务默认监听 `http://127.0.0.1:8000`（从其他机器访问时把主机名替换为服务器地址）。当前接口不要求鉴权，也不接受 JSON 音频请求体；所有带音频的接口都必须使用 `multipart/form-data` 上传文件。

| Method | Path | 请求格式 | 用途 |
|---|---|---|---|
| `GET` | `/healthz` | 无请求体 | 健康检查与当前模型配置 |
| `POST` | `/v1/audio/transcriptions` | `multipart/form-data` | 单文件转写，可选时间戳 |
| `POST` | `/v1/audio/transcriptions/batch` | `multipart/form-data` | 多文件批量转写，可选时间戳 |
| `POST` | `/v1/audio/alignments` | `multipart/form-data` | 将已有文本强制对齐到音频 |

FastAPI 还会自动提供：

- Swagger UI：`GET /docs`
- ReDoc：`GET /redoc`
- OpenAPI JSON：`GET /openapi.json`

### 通用约定

- 所有成功响应的 `Content-Type` 都是 `application/json`。
- 上传字段中的 `@` 是 curl 的“读取本地文件”语法，不属于实际文件名。
- `language` 可以使用完整名称，例如 `Chinese`、`English`、`Japanese`，也可以使用代码，例如 `zh`、`en`、`ja`。
- 转写接口省略 `language` 时由模型自动检测；已知语言时显式填写通常更稳定。
- `prompt` 是可选上下文，可用于传入领域词汇、人名、作品名或其他识别提示；可以与 `language` 同时使用。
- `return_timestamps` 接受 `true`/`false`，也接受 `1`/`0`、`yes`/`no`、`on`/`off`。
- `timestamps[].start_time` 和 `timestamps[].end_time` 的单位都是秒。
- `aligner_model` 表示服务是否已经加载 Aligner，不表示当前请求一定返回了时间戳。
- 只有请求中指定 `return_timestamps=true` 时，转写响应才包含 `timestamps` 字段。
- 使用时间戳或 `/v1/audio/alignments` 前，服务端必须配置 `QWEN_ALIGNER_MODEL`。
- 单个上传文件不能超过 `QWEN_ASR_MAX_UPLOAD_MB`；批量接口对每个文件分别应用这个限制。
- 路径虽然与 OpenAI 音频接口相似，但这里返回的是本项目定义的 JSON；不要假设它支持 OpenAI SDK 的 `model`、`response_format`、`temperature` 等额外字段。

所有接口中的单个时间戳对象都使用同一结构：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 当前对齐单元；中文通常是单字，空格分词语言通常是单词 |
| `start_time` | number | 开始时间，单位秒 |
| `end_time` | number | 结束时间，单位秒 |

### `GET /healthz`：健康检查

请求没有参数和请求体：

```bash
curl http://127.0.0.1:8000/healthz
```

成功响应：`HTTP 200`

```json
{
  "ok": true,
  "backend": "transformers-native",
  "transformers_version": "5.14.0.dev0",
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "device": "cuda:0",
  "dtype": "float32",
  "aligner_dtype": "float32",
  "max_batch_size": 4,
  "max_new_tokens": 256,
  "max_upload_mb": 100,
  "compile_asr": false,
  "compile_aligner": false,
  "compile_mode": null
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | boolean | 模型和服务状态是否已经就绪 |
| `backend` | string | 当前推理后端，原生实现固定为 `transformers-native` |
| `transformers_version` | string | 实际加载的 Transformers 版本 |
| `model` | string | ASR 模型 ID 或本地目录 |
| `aligner_model` | string \| null | Aligner 模型 ID；未加载时为 `null` |
| `device` | string | 例如 `cpu` 或 `cuda:0` |
| `dtype` | string | ASR dtype |
| `aligner_dtype` | string | Aligner dtype；未加载时仍显示配置值 |
| `max_batch_size` | integer | 原生推理内部一次处理的最大 batch |
| `max_new_tokens` | integer | 单个转写结果最多生成的 token 数 |
| `max_upload_mb` | integer | 单文件上传上限，单位 MB |
| `compile_asr` | boolean | ASR 是否启用 `torch.compile` |
| `compile_aligner` | boolean | Aligner 是否启用 `torch.compile` |
| `compile_mode` | string \| null | 传给 `torch.compile` 的 mode |

`transformers_version` 应以实际响应为准，不建议客户端对这个字符串做精确匹配。

### `POST /v1/audio/transcriptions`：单文件转写

请求类型：`multipart/form-data`

| 表单字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `file` | file | 是 | 无 | 要转写的音频文件 |
| `language` | string | 否 | 自动检测 | 语言名称或代码 |
| `prompt` | string | 否 | 空字符串 | 识别上下文或领域词汇提示 |
| `return_timestamps` | boolean string | 否 | `false` | 是否在转写后调用 Aligner |

不返回时间戳的请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./song.wav" \
  -F "language=zh" \
  -F "prompt=角色名：小春、真理、爱丽丝"
```

成功响应：`HTTP 200`

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": null,
  "file_name": "song.wav"
}
```

输出字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 转写文本 |
| `language` | string \| null | 规范化后的语言名称或模型检测结果 |
| `model` | string | 当前 ASR 模型 |
| `aligner_model` | string \| null | 服务加载的 Aligner；未加载时为 `null` |
| `file_name` | string \| null | 上传时提交的原始文件名 |
| `timestamps` | array | 仅在 `return_timestamps=true` 时出现 |

请求时间戳时：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./song.wav" \
  -F "language=Chinese" \
  -F "return_timestamps=true"
```

成功响应：`HTTP 200`

```json
{
  "text": "你好，这里是一个转写示例。",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "file_name": "song.wav",
  "timestamps": [
    {
      "text": "你",
      "start_time": 0.12,
      "end_time": 0.28
    },
    {
      "text": "好",
      "start_time": 0.28,
      "end_time": 0.56
    }
  ]
}
```

静音或空转写请求时间戳时，`timestamps` 会是空数组：

```json
{
  "text": "",
  "language": "Chinese",
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "file_name": "silence.wav",
  "timestamps": []
}
```

### `POST /v1/audio/transcriptions/batch`：批量转写

请求类型：`multipart/form-data`

| 表单字段 | 类型 | 必填 | 数量规则 | 说明 |
|---|---|---:|---|---|
| `files` | file[] | 是 | 至少 1 个，可重复 | 按提交顺序处理的音频文件 |
| `language` | string[] | 否 | 0、1 或与文件数相同 | 省略为全部自动检测；1 个值广播到全部文件 |
| `prompt` | string[] | 否 | 0、1 或与文件数相同 | 省略为空；1 个值广播到全部文件 |
| `return_timestamps` | boolean string | 否 | 0 或 1 | 是否为所有非空转写结果生成时间戳 |

所有文件使用相同语言和 prompt：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./a.wav" \
  -F "files=@./b.wav" \
  -F "language=Chinese" \
  -F "prompt=角色名：小春、真理" \
  -F "return_timestamps=true"
```

为每个文件分别指定语言和 prompt 时，字段顺序应与 `files` 顺序一致：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch \
  -F "files=@./zh.wav" \
  -F "files=@./en.wav" \
  -F "language=zh" \
  -F "language=en" \
  -F "prompt=中文角色名：小春" \
  -F "prompt=English character name: Alice"
```

带时间戳的成功响应：`HTTP 200`

```json
{
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "results": [
    {
      "index": 0,
      "file_name": "a.wav",
      "language": "Chinese",
      "text": "第一段文本。",
      "timestamps": [
        {
          "text": "第",
          "start_time": 0.1,
          "end_time": 0.22
        }
      ]
    },
    {
      "index": 1,
      "file_name": "b.wav",
      "language": "Chinese",
      "text": "第二段文本。",
      "timestamps": [
        {
          "text": "第",
          "start_time": 0.08,
          "end_time": 0.2
        }
      ]
    }
  ]
}
```

不请求时间戳时，每个 result 中不会出现 `timestamps`：

```json
{
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "aligner_model": null,
  "results": [
    {
      "index": 0,
      "file_name": "a.wav",
      "language": "Chinese",
      "text": "第一段文本。"
    },
    {
      "index": 1,
      "file_name": "b.wav",
      "language": "English",
      "text": "The second transcription."
    }
  ]
}
```

`index` 从 0 开始，响应顺序始终与 `files` 的提交顺序一致。`QWEN_ASR_MAX_BATCH_SIZE` 只影响服务器内部如何分块推理，不改变输出顺序。

### `POST /v1/audio/alignments`：已有文本强制对齐

这个接口不执行 ASR，而是把调用方提供的准确文本直接对齐到音频。服务端必须已经配置 `QWEN_ALIGNER_MODEL`。

请求类型：`multipart/form-data`

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `file` | file | 是 | 要对齐的音频文件 |
| `text` | string | 是 | 与音频内容对应的完整文本 |
| `language` | string | 是 | 对齐语言名称或代码 |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/alignments \
  -F "file=@./song.wav" \
  -F "language=Chinese" \
  -F "text=这里填写已经准备好的歌词全文"
```

成功响应：`HTTP 200`

```json
{
  "text": "这里填写已经准备好的歌词全文",
  "language": "Chinese",
  "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "file_name": "song.wav",
  "timestamps": [
    {
      "text": "这",
      "start_time": 0.11,
      "end_time": 0.18
    },
    {
      "text": "里",
      "start_time": 0.18,
      "end_time": 0.28
    }
  ]
}
```

Forced Aligner 支持中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语和西班牙语。日语需要安装 `nagisa`，韩语需要安装 `soynlp`；可通过 `align-ja-ko` extra 一并安装。

### Windows PowerShell 请求

PowerShell 中建议使用 `curl.exe`，避免和 PowerShell 自带的 `curl` 别名混淆。

单文件转写：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@.\song.wav" `
  -F "language=Chinese" `
  -F "prompt=角色名：小春、真理"
```

批量转写：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions/batch `
  -F "files=@.\a.wav" `
  -F "files=@.\b.wav" `
  -F "language=Chinese" `
  -F "return_timestamps=true"
```

已有文本强制对齐：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/alignments `
  -F "file=@.\song.wav" `
  -F "language=Chinese" `
  -F "text=这里填写已经准备好的歌词全文"
```

### 错误响应

服务主动返回的错误通常使用下面的 JSON 结构：

```json
{
  "detail": "错误说明"
}
```

常见状态码：

| 状态码 | 场景 | 示例 |
|---:|---|---|
| `400` | 参数值非法、语言不支持、batch 字段数量不匹配、未启用 Aligner | `{"detail":"ForcedAligner is not enabled on the server. Set QWEN_ALIGNER_MODEL first."}` |
| `413` | 单个上传文件超过 `QWEN_ASR_MAX_UPLOAD_MB` | `{"detail":"Uploaded file is too large. Limit is 100 MB."}` |
| `422` | 缺少 FastAPI 必填表单字段，例如未提交 `file`、`files`、`text` 或 `language` | `detail` 为字段校验错误数组 |
| `503` | 服务或模型尚未就绪 | `{"detail":"Model is not ready yet."}` |

例如，在没有配置 Aligner 时请求时间戳：

```json
{
  "detail": "ForcedAligner is not enabled on the server. Set QWEN_ALIGNER_MODEL first."
}
```

批量接口中 `language` 或 `prompt` 的数量既不是 1、也不等于文件数量时：

```json
{
  "detail": "Field 'language' expects 1 or 2 values, got 3."
}
```

## systemd 示例

仓库提供 [qwen3-asr.service.example](./qwen3-asr.service.example)。该文件是 CPU 示例，因此显式使用推荐的 0.6B-hf + `float16`；应用本身未设置 `QWEN_ASR_MODEL` 时仍默认使用 1.7B-hf。

使用步骤：

1. 复制到 `/etc/systemd/system/qwen3-asr.service`。
2. 按实际路径修改 `User`、`WorkingDirectory` 和 `ExecStart`。
3. 不需要时间戳时，删除 `QWEN_ALIGNER_MODEL` 和 `QWEN_ALIGNER_DTYPE`。
4. 根据 CPU 核心数调整线程环境变量。

```bash
sudo cp qwen3-asr.service.example /etc/systemd/system/qwen3-asr.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-asr.service
sudo systemctl status qwen3-asr.service
```

查看日志：

```bash
journalctl -u qwen3-asr.service -f
```

改成 GPU 时，需要同步调整：

- `ExecStart` 中的 `--extra cpu` 改为 `--extra cu128`。
- `QWEN_ASR_MODEL` 可改为默认的 `Qwen/Qwen3-ASR-1.7B-hf`。
- `QWEN_ASR_DEVICE=cpu` 改为 `QWEN_ASR_DEVICE=cuda:0`。
- ASR 和 Aligner dtype 从 CPU 默认的 `float16` 改为 GPU 默认的 `float32`。
- CPU 线程相关环境变量通常可以删除。

## 常用环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | ASR 模型 ID 或本地目录；CPU 推荐显式改为 0.6B-hf |
| `QWEN_ALIGNER_MODEL` | 空 | Forced Aligner 模型 ID 或本地目录；推荐 `Qwen/Qwen3-ForcedAligner-0.6B-hf` |
| `QWEN_ASR_DEVICE` | 自动检测 | 有 CUDA 时为 `cuda:0`，否则为 `cpu` |
| `QWEN_ASR_DTYPE` | GPU=`float32` / CPU=`float16` | ASR 推理精度；CPU 遇到 Half 算子兼容问题时回退 `float32` |
| `QWEN_ALIGNER_DTYPE` | 跟随 `QWEN_ASR_DTYPE` | 对齐模型推理精度 |
| `QWEN_ASR_THREADS` | 逻辑 CPU 数的一半 | CPU 模式传给 PyTorch 的线程数 |
| `QWEN_ASR_MAX_BATCH_SIZE` | GPU=`4` / CPU=`2` | 模型内部最大 batch；CPU 实际部署推荐从 1 开始 |
| `QWEN_ASR_MAX_NEW_TOKENS` | `256` | 每个转写结果最大生成 token 数 |
| `QWEN_ASR_MAX_UPLOAD_MB` | `100` | 单文件上传大小限制 |
| `QWEN_ASR_TMPDIR` | 系统临时目录 | 上传文件缓存目录 |
| `QWEN_ASR_HOST` | `0.0.0.0` | 监听地址 |
| `QWEN_ASR_PORT` | `8000` | 监听端口 |
| `QWEN_ASR_ATTN_IMPLEMENTATION` | 空 | ASR 的可选 attention 实现；`flash_attention_2` 要求显式使用 FP16/BF16，不能配 GPU FP32 |
| `QWEN_ALIGNER_ATTN_IMPLEMENTATION` | 空 | Aligner 的可选 attention 实现；`flash_attention_2` 同样要求 FP16/BF16 |
| `QWEN_ASR_COMPILE` | `false` | 是否对 ASR forward 启用 `torch.compile` |
| `QWEN_ALIGNER_COMPILE` | `false` | 是否对 Aligner forward 启用 `torch.compile` |
| `QWEN_COMPILE_MODE` | 空 | 直接传给 `torch.compile` 的 `mode`；空值使用 PyTorch 自身默认模式 |
| `OPENBLAS_NUM_THREADS` | 不设置 | 可选，控制 OpenBLAS 线程数 |

布尔环境变量接受 `true`/`false`，也接受常见的 `1`/`0`、`yes`/`no`、`on`/`off` 写法。

## 目录结构

```text
.
├── app.py
├── native_asr.py
├── pyproject.toml
├── qwen3-asr.service.example
└── README.md
```

## 实用建议

- CPU 首次部署优先使用 0.6B-hf、`float16`、batch size 1，并先关闭 Aligner 和编译；不支持 Half 算子时回退 `float32`。
- 已知音频语言时显式传 `language`，可以跳过或减少自动语言判断的不确定性。
- 已有标准歌词时直接调用 `/v1/audio/alignments`。
- 先用短 WAV 验证下载、加载和接口，再测试长音频、并发和内存峰值。
- 本服务使用推理锁串行进入模型；batch 接口比客户端同时发很多单文件请求更容易控制显存和内存。
- 编译加速必须在目标硬件、目标音频长度和目标 batch 上实测，不要把 A100 batch 4 的数据直接外推到 CPU。

## 参考

- [Transformers 固定源码提交](https://github.com/huggingface/transformers/commit/498d6e984e84d29186e671656817a53a024af930)
- [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)
- [Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Qwen3-ASR 项目](https://github.com/QwenLM/Qwen3-ASR)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [uv 文档](https://docs.astral.sh/uv/)
- [uv 的 PyTorch 集成指南](https://docs.astral.sh/uv/guides/integration/pytorch/)
