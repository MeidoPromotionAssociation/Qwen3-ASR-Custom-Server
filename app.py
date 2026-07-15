from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from transformers import __version__ as transformers_version

from native_asr import NativeQwen3ASRModel


LOG = logging.getLogger("qwen3_asr_http")
INFER_LOCK = threading.Lock()
STATE: dict[str, Any] = {"model": None, "config": None}
AUDIO_SAMPLE_RATE = 16_000
FFMPEG_ERROR_DETAIL_LIMIT = 2_000


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be true or false, got {value!r}.")


def _auto_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _default_dtype(device: str) -> str:
    # Project policy: CPU prioritizes memory use with FP16, while CUDA uses
    # FP32 by default. Both remain explicitly configurable through the env vars.
    return "float32" if device.startswith("cuda") else "float16"


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _load_config() -> dict[str, Any]:
    device = os.getenv("QWEN_ASR_DEVICE", "").strip() or _auto_device()
    dtype_name = os.getenv("QWEN_ASR_DTYPE", "").strip() or _default_dtype(device)
    aligner_dtype_name = os.getenv("QWEN_ALIGNER_DTYPE", "").strip() or dtype_name
    logical_cpus = max(1, os.cpu_count() or 1)
    default_threads = max(1, logical_cpus // 2)
    default_batch = 4 if device.startswith("cuda") else 2
    temp_dir = os.getenv("QWEN_ASR_TMPDIR", "").strip() or None
    ffmpeg_binary = os.getenv("QWEN_ASR_FFMPEG", "ffmpeg").strip() or "ffmpeg"
    ffmpeg_timeout_seconds = _env_int("QWEN_ASR_FFMPEG_TIMEOUT", 120)
    if ffmpeg_timeout_seconds <= 0:
        raise ValueError("QWEN_ASR_FFMPEG_TIMEOUT must be greater than zero.")

    return {
        "model_name": os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf"),
        "aligner_model_name": os.getenv("QWEN_ALIGNER_MODEL", "").strip() or None,
        "device": device,
        "dtype_name": dtype_name,
        "dtype": _torch_dtype(dtype_name),
        "aligner_dtype_name": aligner_dtype_name,
        "aligner_dtype": _torch_dtype(aligner_dtype_name),
        "max_batch_size": _env_int("QWEN_ASR_MAX_BATCH_SIZE", default_batch),
        "max_new_tokens": _env_int("QWEN_ASR_MAX_NEW_TOKENS", 256),
        "max_upload_mb": _env_int("QWEN_ASR_MAX_UPLOAD_MB", 100),
        "max_upload_bytes": _env_int("QWEN_ASR_MAX_UPLOAD_MB", 100) * 1024 * 1024,
        "threads": _env_int("QWEN_ASR_THREADS", default_threads),
        "temp_dir": temp_dir,
        "ffmpeg_binary": ffmpeg_binary,
        "ffmpeg_timeout_seconds": ffmpeg_timeout_seconds,
        "host": os.getenv("QWEN_ASR_HOST", "0.0.0.0"),
        "port": _env_int("QWEN_ASR_PORT", 8000),
        "attn_implementation": os.getenv("QWEN_ASR_ATTN_IMPLEMENTATION", "").strip() or None,
        "aligner_attn_implementation": os.getenv("QWEN_ALIGNER_ATTN_IMPLEMENTATION", "").strip() or None,
        "compile_asr": _env_bool("QWEN_ASR_COMPILE"),
        "compile_aligner": _env_bool("QWEN_ALIGNER_COMPILE"),
        "compile_mode": os.getenv("QWEN_COMPILE_MODE", "").strip() or None,
    }


def _resolve_ffmpeg(binary: str) -> Optional[str]:
    return shutil.which(binary)


def _configure_torch(config: dict[str, Any]) -> None:
    if not config["device"].startswith("cuda"):
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)


def _build_model(config: dict[str, Any]) -> NativeQwen3ASRModel:
    return NativeQwen3ASRModel.from_pretrained(
        config["model_name"],
        device_map=config["device"],
        dtype=config["dtype"],
        max_batch_size=config["max_batch_size"],
        max_new_tokens=config["max_new_tokens"],
        attn_implementation=config["attn_implementation"],
        compile_asr=config["compile_asr"],
        aligner_model_name=config["aligner_model_name"],
        aligner_dtype=config["aligner_dtype"],
        aligner_attn_implementation=config["aligner_attn_implementation"],
        compile_aligner=config["compile_aligner"],
        compile_mode=config["compile_mode"],
    )


def _save_upload(upload: UploadFile, temp_dir: Optional[str], max_upload_bytes: int) -> str:
    suffix = Path(upload.filename or "audio.upload").suffix
    if not (1 < len(suffix) <= 17 and suffix.isascii() and suffix[1:].isalnum()):
        suffix = ".upload"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp:
        written = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_upload_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file is too large. Limit is {max_upload_bytes // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
        return tmp.name


def _unlink_temp_file(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _ffmpeg_error_detail(stderr: Any, source_path: str, output_path: str) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    detail = " ".join(str(stderr or "").split())
    detail = detail.replace(source_path, "<uploaded audio>")
    detail = detail.replace(output_path, "<converted audio>")
    if len(detail) > FFMPEG_ERROR_DETAIL_LIMIT:
        detail = "..." + detail[-(FFMPEG_ERROR_DETAIL_LIMIT - 3) :]
    return detail


def _convert_audio_to_wav(
    source_path: str,
    *,
    ffmpeg_path: Optional[str],
    temp_dir: Optional[str],
    timeout_seconds: int,
    original_filename: Optional[str],
) -> str:
    display_name = original_filename or "uploaded audio"
    if ffmpeg_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server-side audio conversion is unavailable because ffmpeg was not found. "
                "Install ffmpeg or retry with convert_audio=false for an already supported audio file."
            ),
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir) as tmp:
        output_path = tmp.name

    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-y",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        output_path,
    ]
    try:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        _unlink_temp_file(output_path)
        raise HTTPException(
            status_code=408,
            detail=(
                f"Server-side audio conversion timed out after {timeout_seconds} seconds "
                f"for {display_name!r}."
            ),
        ) from exc
    except subprocess.CalledProcessError as exc:
        _unlink_temp_file(output_path)
        reason = _ffmpeg_error_detail(exc.stderr, source_path, output_path)
        detail = (
            f"ffmpeg could not decode {display_name!r}. The file may be corrupt, use a codec "
            "unavailable in this ffmpeg build, or contain no audio stream."
        )
        if reason:
            detail = f"{detail} ffmpeg reported: {reason}"
        raise HTTPException(status_code=400, detail=detail) from exc
    except OSError as exc:
        _unlink_temp_file(output_path)
        raise HTTPException(
            status_code=503,
            detail=f"Server-side audio conversion could not start ffmpeg: {exc}",
        ) from exc

    return output_path


def _prepare_audio_upload(
    upload: UploadFile,
    config: dict[str, Any],
    *,
    convert_audio: bool,
) -> str:
    source_path = _save_upload(upload, config["temp_dir"], config["max_upload_bytes"])
    if not convert_audio:
        return source_path

    try:
        return _convert_audio_to_wav(
            source_path,
            ffmpeg_path=config.get("ffmpeg_path"),
            temp_dir=config["temp_dir"],
            timeout_seconds=config["ffmpeg_timeout_seconds"],
            original_filename=upload.filename,
        )
    finally:
        _unlink_temp_file(source_path)


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise HTTPException(status_code=400, detail="Invalid boolean value. Use true or false.")


def _is_audio_decode_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        module = type(current).__module__
        if module == "soundfile" or module.startswith(("audioread", "librosa")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _inference_http_exception(
    exc: Exception,
    *,
    convert_audio: bool,
) -> Optional[HTTPException]:
    if _is_audio_decode_error(exc):
        if convert_audio:
            return HTTPException(
                status_code=500,
                detail=(
                    "The server audio backend could not read the WAV produced by ffmpeg. "
                    "Check the server's soundfile/librosa installation."
                ),
            )
        return HTTPException(
            status_code=400,
            detail=(
                "The uploaded audio could not be decoded without server-side conversion. "
                "Retry with convert_audio=true or upload an audio format supported by the model backend."
            ),
        )
    if isinstance(exc, (ValueError, ImportError)):
        return HTTPException(status_code=400, detail=str(exc))
    return None


def _normalize_list(values: Optional[list[str]], n: int, field_name: str) -> list[str]:
    if not values:
        return [""] * n
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{field_name}' expects 1 or {n} values, got {len(values)}.",
        )
    return values


def _serialize_timestamps(items: Any) -> list[dict[str, Any]]:
    serialized = []
    for item in items or []:
        if isinstance(item, dict):
            serialized.append(
                {
                    "text": item.get("text"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                }
            )
        else:
            serialized.append(
                {
                    "text": getattr(item, "text", None),
                    "start_time": getattr(item, "start_time", None),
                    "end_time": getattr(item, "end_time", None),
                }
            )
    return serialized


def _require_aligner(config: dict[str, Any], model: NativeQwen3ASRModel) -> NativeQwen3ASRModel:
    if not config["aligner_model_name"] or not model.has_aligner:
        raise HTTPException(
            status_code=400,
            detail="ForcedAligner is not enabled on the server. Set QWEN_ALIGNER_MODEL first.",
        )
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _load_config()
    config["ffmpeg_path"] = _resolve_ffmpeg(config["ffmpeg_binary"])
    if config["ffmpeg_path"] is None:
        LOG.warning(
            "ffmpeg=%r was not found; requests explicitly using convert_audio=true will return "
            "HTTP 503 until ffmpeg is installed. The default convert_audio=false path remains "
            "available for audio supported by the model backend.",
            config["ffmpeg_binary"],
        )
    else:
        LOG.info(
            "Server-side audio conversion enabled ffmpeg=%s timeout=%ss target=wav/pcm_s16le/%sHz/mono",
            config["ffmpeg_path"],
            config["ffmpeg_timeout_seconds"],
            AUDIO_SAMPLE_RATE,
        )
    _configure_torch(config)
    LOG.info(
        "Loading native Transformers model=%s aligner=%s device=%s dtype=%s "
        "aligner_dtype=%s batch=%s max_new_tokens=%s compile_asr=%s compile_aligner=%s compile_mode=%s",
        config["model_name"],
        config["aligner_model_name"],
        config["device"],
        config["dtype_name"],
        config["aligner_dtype_name"],
        config["max_batch_size"],
        config["max_new_tokens"],
        config["compile_asr"],
        config["compile_aligner"],
        config["compile_mode"],
    )
    STATE["config"] = config
    STATE["model"] = _build_model(config)
    LOG.info("Model loaded")
    yield


app = FastAPI(
    title="Qwen3-ASR HTTP Service",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return {
        "ok": True,
        "backend": "transformers-native",
        "transformers_version": transformers_version,
        "model": config["model_name"],
        "aligner_model": config["aligner_model_name"],
        "device": config["device"],
        "dtype": config["dtype_name"],
        "aligner_dtype": config["aligner_dtype_name"],
        "max_batch_size": config["max_batch_size"],
        "max_new_tokens": config["max_new_tokens"],
        "max_upload_mb": config["max_upload_mb"],
        "audio_conversion_default": False,
        "audio_conversion_available": config["ffmpeg_path"] is not None,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "ffmpeg_timeout_seconds": config["ffmpeg_timeout_seconds"],
        "compile_asr": config["compile_asr"],
        "compile_aligner": config["compile_aligner"],
        "compile_mode": config["compile_mode"],
    }


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: str = Form(""),
    return_timestamps: Optional[str] = Form(None),
    convert_audio: Optional[str] = Form(None),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    want_timestamps = _parse_bool(return_timestamps, default=False)
    should_convert_audio = _parse_bool(convert_audio, default=False)
    if want_timestamps:
        _require_aligner(config, model)

    temp_path: Optional[str] = None
    try:
        temp_path = _prepare_audio_upload(
            file,
            config,
            convert_audio=should_convert_audio,
        )
        with INFER_LOCK:
            try:
                result = model.transcribe(
                    audio=temp_path,
                    context=prompt,
                    language=_clean_optional(language),
                    return_time_stamps=want_timestamps,
                )[0]
            except Exception as exc:
                http_exc = _inference_http_exception(
                    exc,
                    convert_audio=should_convert_audio,
                )
                if http_exc is not None:
                    raise http_exc from exc
                raise

        payload = {
            "text": result.text,
            "language": result.language,
            "model": config["model_name"],
            "aligner_model": config["aligner_model_name"],
            "file_name": file.filename,
        }
        if want_timestamps:
            payload["timestamps"] = _serialize_timestamps(getattr(result, "time_stamps", None))
        return payload
    finally:
        _unlink_temp_file(temp_path)
        file.file.close()


@app.post("/v1/audio/transcriptions/batch")
def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: Optional[list[str]] = Form(None),
    prompt: Optional[list[str]] = Form(None),
    return_timestamps: Optional[str] = Form(None),
    convert_audio: Optional[str] = Form(None),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    want_timestamps = _parse_bool(return_timestamps, default=False)
    should_convert_audio = _parse_bool(convert_audio, default=False)
    if want_timestamps:
        _require_aligner(config, model)

    prompts = _normalize_list(prompt, len(files), "prompt")
    languages = _normalize_list(language, len(files), "language")
    temp_paths: list[str] = []

    try:
        for upload in files:
            temp_paths.append(
                _prepare_audio_upload(
                    upload,
                    config,
                    convert_audio=should_convert_audio,
                )
            )

        with INFER_LOCK:
            try:
                results = model.transcribe(
                    audio=temp_paths,
                    context=prompts,
                    language=[_clean_optional(item) for item in languages],
                    return_time_stamps=want_timestamps,
                )
            except Exception as exc:
                http_exc = _inference_http_exception(
                    exc,
                    convert_audio=should_convert_audio,
                )
                if http_exc is not None:
                    raise http_exc from exc
                raise

        return {
            "model": config["model_name"],
            "aligner_model": config["aligner_model_name"],
            "results": [
                {
                    "index": idx,
                    "file_name": upload.filename,
                    "language": result.language,
                    "text": result.text,
                    **(
                        {"timestamps": _serialize_timestamps(getattr(result, "time_stamps", None))}
                        if want_timestamps
                        else {}
                    ),
                }
                for idx, (upload, result) in enumerate(zip(files, results))
            ],
        }
    finally:
        for path in temp_paths:
            _unlink_temp_file(path)
        for upload in files:
            upload.file.close()


@app.post("/v1/audio/alignments")
def align_text_to_audio(
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(...),
    convert_audio: Optional[str] = Form(None),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    aligner = _require_aligner(config, model)
    should_convert_audio = _parse_bool(convert_audio, default=False)

    temp_path: Optional[str] = None
    try:
        temp_path = _prepare_audio_upload(
            file,
            config,
            convert_audio=should_convert_audio,
        )
        with INFER_LOCK:
            try:
                result = aligner.align(
                    audio=temp_path,
                    text=text,
                    language=language,
                )[0]
            except Exception as exc:
                http_exc = _inference_http_exception(
                    exc,
                    convert_audio=should_convert_audio,
                )
                if http_exc is not None:
                    raise http_exc from exc
                raise

        return {
            "text": text,
            "language": language,
            "aligner_model": config["aligner_model_name"],
            "file_name": file.filename,
            "timestamps": _serialize_timestamps(result),
        }
    finally:
        _unlink_temp_file(temp_path)
        file.file.close()


def main() -> None:
    config = _load_config()
    parser = argparse.ArgumentParser(description="Run the Qwen3-ASR HTTP service.")
    parser.add_argument("--host", default=config["host"])
    parser.add_argument("--port", type=int, default=config["port"])
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
