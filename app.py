from __future__ import annotations

import argparse
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from qwen_asr import Qwen3ASRModel


LOG = logging.getLogger("qwen3_asr_http")
INFER_LOCK = threading.Lock()
STATE: dict[str, Any] = {"model": None, "config": None}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _auto_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _default_dtype(device: str) -> str:
    return "float16" if device.startswith("cuda") else "float32"


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

    return {
        "model_name": os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
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
        "host": os.getenv("QWEN_ASR_HOST", "0.0.0.0"),
        "port": _env_int("QWEN_ASR_PORT", 8000),
        "attn_implementation": os.getenv("QWEN_ASR_ATTN_IMPLEMENTATION", "").strip() or None,
        "aligner_attn_implementation": os.getenv("QWEN_ALIGNER_ATTN_IMPLEMENTATION", "").strip() or None,
    }


def _configure_torch(config: dict[str, Any]) -> None:
    if not config["device"].startswith("cuda"):
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)


def _build_model(config: dict[str, Any]) -> Qwen3ASRModel:
    kwargs: dict[str, Any] = {
        "device_map": config["device"],
        "dtype": config["dtype"],
        "max_inference_batch_size": config["max_batch_size"],
        "max_new_tokens": config["max_new_tokens"],
    }
    if config["attn_implementation"]:
        kwargs["attn_implementation"] = config["attn_implementation"]
    if config["aligner_model_name"]:
        aligner_kwargs: dict[str, Any] = {
            "device_map": config["device"],
            "dtype": config["aligner_dtype"],
        }
        if config["aligner_attn_implementation"]:
            aligner_kwargs["attn_implementation"] = config["aligner_attn_implementation"]
        kwargs["forced_aligner"] = config["aligner_model_name"]
        kwargs["forced_aligner_kwargs"] = aligner_kwargs

    model = Qwen3ASRModel.from_pretrained(config["model_name"], **kwargs)
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    return model


def _save_upload(upload: UploadFile, temp_dir: Optional[str], max_upload_bytes: int) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
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
    return [
        {
            "text": getattr(item, "text", None),
            "start_time": getattr(item, "start_time", None),
            "end_time": getattr(item, "end_time", None),
        }
        for item in (items or [])
    ]


def _require_aligner(config: dict[str, Any], model: Qwen3ASRModel) -> Any:
    aligner = getattr(model, "forced_aligner", None)
    if not config["aligner_model_name"] or aligner is None:
        raise HTTPException(
            status_code=400,
            detail="ForcedAligner is not enabled on the server. Set QWEN_ALIGNER_MODEL first.",
        )
    return aligner


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _load_config()
    _configure_torch(config)
    LOG.info(
        "Loading model=%s aligner=%s device=%s dtype=%s aligner_dtype=%s batch=%s max_new_tokens=%s",
        config["model_name"],
        config["aligner_model_name"],
        config["device"],
        config["dtype_name"],
        config["aligner_dtype_name"],
        config["max_batch_size"],
        config["max_new_tokens"],
    )
    STATE["config"] = config
    STATE["model"] = _build_model(config)
    LOG.info("Model loaded")
    yield


app = FastAPI(
    title="Qwen3-ASR HTTP Service",
    version="0.2.0",
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
        "model": config["model_name"],
        "aligner_model": config["aligner_model_name"],
        "device": config["device"],
        "dtype": config["dtype_name"],
        "aligner_dtype": config["aligner_dtype_name"],
        "max_batch_size": config["max_batch_size"],
        "max_new_tokens": config["max_new_tokens"],
        "max_upload_mb": config["max_upload_mb"],
    }


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: str = Form(""),
    return_timestamps: Optional[str] = Form(None),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    want_timestamps = _parse_bool(return_timestamps, default=False)
    if want_timestamps:
        _require_aligner(config, model)

    temp_path = _save_upload(file, config["temp_dir"], config["max_upload_bytes"])
    try:
        with INFER_LOCK:
            try:
                result = model.transcribe(
                    audio=temp_path,
                    context=prompt,
                    language=_clean_optional(language),
                    return_time_stamps=want_timestamps,
                )[0]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        file.file.close()


@app.post("/v1/audio/transcriptions/batch")
def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: Optional[list[str]] = Form(None),
    prompt: Optional[list[str]] = Form(None),
    return_timestamps: Optional[str] = Form(None),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    want_timestamps = _parse_bool(return_timestamps, default=False)
    if want_timestamps:
        _require_aligner(config, model)

    prompts = _normalize_list(prompt, len(files), "prompt")
    languages = _normalize_list(language, len(files), "language")
    temp_paths: list[str] = []

    try:
        for upload in files:
            temp_paths.append(_save_upload(upload, config["temp_dir"], config["max_upload_bytes"]))

        with INFER_LOCK:
            try:
                results = model.transcribe(
                    audio=temp_paths,
                    context=prompts,
                    language=[_clean_optional(item) for item in languages],
                    return_time_stamps=want_timestamps,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

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
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for upload in files:
            upload.file.close()


@app.post("/v1/audio/alignments")
def align_text_to_audio(
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(...),
) -> dict[str, Any]:
    config = STATE["config"]
    model = STATE["model"]
    if config is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    aligner = _require_aligner(config, model)

    temp_path = _save_upload(file, config["temp_dir"], config["max_upload_bytes"])
    try:
        with INFER_LOCK:
            try:
                result = aligner.align(
                    audio=temp_path,
                    text=text,
                    language=language,
                )[0]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "text": text,
            "language": language,
            "aligner_model": config["aligner_model_name"],
            "file_name": file.filename,
            "timestamps": _serialize_timestamps(getattr(result, "items", None)),
        }
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
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
