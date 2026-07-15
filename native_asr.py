from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor
from transformers.audio_utils import load_audio


# Qwen3-ASR's native processor accepts both ISO codes and these canonical names.
# Keeping the normalization here is only needed for the context+language compatibility
# path, where the language is used as an assistant prefix (matching qwen-asr 0.0.6).
LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str]
    time_stamps: Optional[list[dict[str, Any]]] = None


def _canonical_language(language: Optional[str]) -> Optional[str]:
    if language is None or language.strip() == "":
        return None

    normalized = language.strip().lower()
    if normalized in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[normalized]

    for name in LANGUAGE_CODE_TO_NAME.values():
        if normalized == name.lower():
            return name

    supported_codes = ", ".join(sorted(LANGUAGE_CODE_TO_NAME))
    raise ValueError(
        f"Unsupported language: {language!r}. Use an ISO code or a full language name. "
        f"Supported codes: {supported_codes}."
    )


def _normalize_text_batch(
    value: Optional[str | Sequence[Optional[str]]],
    count: int,
    field_name: str,
) -> list[Optional[str]]:
    if value is None:
        return [None] * count
    if isinstance(value, str):
        return [value] * count

    values = list(value)
    if len(values) == 1 and count > 1:
        values *= count
    if len(values) != count:
        raise ValueError(f"Batch size mismatch: audio={count}, {field_name}={len(values)}")
    return values


def _audio_content_item(audio: Any) -> dict[str, Any]:
    if isinstance(audio, str):
        return {"type": "audio", "path": audio}
    return {"type": "audio", "audio": audio}


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_dtype(model: Any) -> torch.dtype:
    return getattr(model, "dtype", torch.float32)


def _compile_forward(model: Any, enabled: bool, mode: Optional[str]) -> None:
    if not enabled:
        return
    kwargs = {"mode": mode} if mode else {}
    model.forward = torch.compile(model.forward, **kwargs)


class NativeQwen3ASRModel:
    """Small service-oriented wrapper around Transformers' native Qwen3-ASR APIs."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        max_batch_size: int,
        max_new_tokens: int,
        aligner_processor: Any = None,
        aligner_model: Any = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if (aligner_processor is None) != (aligner_model is None):
            raise ValueError("aligner_processor and aligner_model must be configured together")

        self.processor = processor
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.aligner_processor = aligner_processor
        self.aligner_model = aligner_model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device_map: str,
        dtype: torch.dtype,
        max_batch_size: int,
        max_new_tokens: int,
        attn_implementation: Optional[str] = None,
        compile_asr: bool = False,
        aligner_model_name: Optional[str] = None,
        aligner_dtype: Optional[torch.dtype] = None,
        aligner_attn_implementation: Optional[str] = None,
        compile_aligner: bool = False,
        compile_mode: Optional[str] = None,
    ) -> "NativeQwen3ASRModel":
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs).eval()
        _compile_forward(model, compile_asr, compile_mode)

        aligner_processor = None
        aligner_model = None
        if aligner_model_name:
            aligner_kwargs: dict[str, Any] = {
                "device_map": device_map,
                "dtype": aligner_dtype or dtype,
            }
            if aligner_attn_implementation:
                aligner_kwargs["attn_implementation"] = aligner_attn_implementation

            aligner_processor = AutoProcessor.from_pretrained(aligner_model_name)
            aligner_model = AutoModelForTokenClassification.from_pretrained(
                aligner_model_name,
                **aligner_kwargs,
            ).eval()
            _compile_forward(aligner_model, compile_aligner, compile_mode)

        return cls(
            processor=processor,
            model=model,
            max_batch_size=max_batch_size,
            max_new_tokens=max_new_tokens,
            aligner_processor=aligner_processor,
            aligner_model=aligner_model,
        )

    @property
    def has_aligner(self) -> bool:
        return self.aligner_processor is not None and self.aligner_model is not None

    def transcribe(
        self,
        audio: Any | Sequence[Any],
        context: str | Sequence[str] = "",
        language: Optional[str | Sequence[Optional[str]]] = None,
        return_time_stamps: bool = False,
    ) -> list[TranscriptionResult]:
        audio_items = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        if not audio_items:
            raise ValueError("audio must contain at least one sample")
        if return_time_stamps and not self.has_aligner:
            raise ValueError("return_time_stamps=True requires a forced aligner")

        contexts = _normalize_text_batch(context, len(audio_items), "context")
        languages = _normalize_text_batch(language, len(audio_items), "language")
        clean_contexts = [item if item and item.strip() else "" for item in contexts]
        canonical_languages = [_canonical_language(item) for item in languages]

        results: list[Optional[TranscriptionResult]] = [None] * len(audio_items)
        for start in range(0, len(audio_items), self.max_batch_size):
            stop = min(start + self.max_batch_size, len(audio_items))
            indices = list(range(start, stop))

            # The recommended native helper covers the common path. Context requests
            # use the native chat template directly so the existing HTTP `prompt`
            # field remains functional. Requests are grouped by template shape to
            # retain batching even when a batch mixes context modes.
            groups = {
                "native": [idx for idx in indices if not clean_contexts[idx]],
                "context": [
                    idx
                    for idx in indices
                    if clean_contexts[idx] and canonical_languages[idx] is None
                ],
                "context_language": [
                    idx
                    for idx in indices
                    if clean_contexts[idx] and canonical_languages[idx] is not None
                ],
            }
            for mode, group_indices in groups.items():
                if not group_indices:
                    continue
                batch_results = self._transcribe_group(
                    audio=[audio_items[idx] for idx in group_indices],
                    contexts=[clean_contexts[idx] for idx in group_indices],
                    languages=[canonical_languages[idx] for idx in group_indices],
                    mode=mode,
                )
                for idx, result in zip(group_indices, batch_results):
                    results[idx] = result

        completed = [result for result in results if result is not None]
        if len(completed) != len(audio_items):
            raise RuntimeError("ASR backend returned an incomplete batch")

        if return_time_stamps:
            align_indices = [idx for idx, result in enumerate(completed) if result.text.strip()]
            if align_indices:
                timestamps = self.align(
                    audio=[audio_items[idx] for idx in align_indices],
                    text=[completed[idx].text for idx in align_indices],
                    language=[completed[idx].language or "English" for idx in align_indices],
                )
                for idx, items in zip(align_indices, timestamps):
                    completed[idx].time_stamps = items
            for result in completed:
                if result.time_stamps is None:
                    result.time_stamps = []

        return completed

    def _transcribe_group(
        self,
        *,
        audio: list[Any],
        contexts: list[str],
        languages: list[Optional[str]],
        mode: str,
    ) -> list[TranscriptionResult]:
        if mode == "native":
            inputs = self.processor.apply_transcription_request(
                audio=audio,
                language=languages,
            )
        else:
            conversations = []
            for audio_item, context, language in zip(audio, contexts, languages):
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": [_audio_content_item(audio_item)]},
                ]
                conversations.append(messages)

            if mode == "context_language":
                # Qwen3-ASR's dedicated template renders only the system/audio
                # prompt and ignores assistant messages. Render the normal prompt
                # first, then append the same language prefix used by qwen-asr
                # 0.0.6 before asking the native processor to tokenize it.
                prompts = self.processor.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(prompts, str):
                    prompts = [prompts]
                else:
                    prompts = list(prompts)
                if len(prompts) != len(audio):
                    raise RuntimeError(
                        f"Chat template returned {len(prompts)} prompt(s) "
                        f"for {len(audio)} audio sample(s)"
                    )
                prompts = [
                    prompt + f"language {language}<asr_text>"
                    for prompt, language in zip(prompts, languages)
                ]
                sampling_rate = getattr(
                    getattr(self.processor, "feature_extractor", None),
                    "sampling_rate",
                    16_000,
                )
                loaded_audio = [
                    load_audio(audio_item, sampling_rate=sampling_rate)
                    for audio_item in audio
                ]
                inputs = self.processor(
                    text=prompts,
                    audio=loaded_audio,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=True,
                )

        inputs = inputs.to(_model_device(self.model), _model_dtype(self.model))
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[:, input_length:]
        parsed = self.processor.decode(generated_ids, return_format="parsed")
        if isinstance(parsed, dict):
            parsed_items = [parsed]
        elif isinstance(parsed, (list, tuple)):
            parsed_items = list(parsed)
        else:
            parsed_items = [parsed]
        if len(parsed_items) != len(audio):
            raise RuntimeError(
                f"ASR backend returned {len(parsed_items)} result(s) for {len(audio)} audio sample(s)"
            )

        output_results = []
        for item, requested_language in zip(parsed_items, languages):
            if not isinstance(item, dict):
                item = {"language": None, "transcription": str(item)}
            output_results.append(
                TranscriptionResult(
                    language=requested_language or item.get("language"),
                    text=str(item.get("transcription", "")),
                )
            )
        return output_results

    def align(
        self,
        audio: Any | Sequence[Any],
        text: str | Sequence[str],
        language: str | Sequence[str],
    ) -> list[list[dict[str, Any]]]:
        if not self.has_aligner:
            raise ValueError("ForcedAligner is not enabled")

        audio_items = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        texts = _normalize_text_batch(text, len(audio_items), "text")
        languages = _normalize_text_batch(language, len(audio_items), "language")
        if any(item is None for item in texts):
            raise ValueError("text is required for forced alignment")
        canonical_languages = [_canonical_language(item) for item in languages]

        all_timestamps: list[list[dict[str, Any]]] = []
        for start in range(0, len(audio_items), self.max_batch_size):
            stop = min(start + self.max_batch_size, len(audio_items))
            aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
                audio=audio_items[start:stop],
                transcript=[str(item) for item in texts[start:stop]],
                language=canonical_languages[start:stop],
            )
            aligner_inputs = aligner_inputs.to(
                _model_device(self.aligner_model),
                _model_dtype(self.aligner_model),
            )
            with torch.inference_mode():
                outputs = self.aligner_model(**aligner_inputs)

            batch_timestamps = self.aligner_processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=self.aligner_model.config.timestamp_token_id,
            )
            all_timestamps.extend(batch_timestamps)

        if len(all_timestamps) != len(audio_items):
            raise RuntimeError(
                f"Forced aligner returned {len(all_timestamps)} result(s) "
                f"for {len(audio_items)} audio sample(s)"
            )
        return all_timestamps
