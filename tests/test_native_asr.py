# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from native_asr import NativeQwen3ASRModel, _canonical_language


class FakeBatchFeature(dict[str, Any]):
    """Small stand-in for transformers.BatchFeature with observable .to calls."""

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.to_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def to(self, *args: Any, **kwargs: Any) -> "FakeBatchFeature":
        self.to_calls.append((args, kwargs))
        return self


class FakeProcessor:
    def __init__(
        self,
        audios: list[str],
        *,
        transcript_by_audio: dict[str, str] | None = None,
        detected_language_by_audio: dict[str, str | None] | None = None,
    ) -> None:
        self.audio_to_token = {audio: index + 1 for index, audio in enumerate(audios)}
        self.token_to_audio = {token: audio for audio, token in self.audio_to_token.items()}
        self.transcript_by_audio = transcript_by_audio or {
            audio: f"transcript:{audio}" for audio in audios
        }
        self.detected_language_by_audio = detected_language_by_audio or {
            audio: "English" for audio in audios
        }
        self.transcription_calls: list[dict[str, Any]] = []
        self.chat_template_calls: list[dict[str, Any]] = []
        self.direct_calls: list[dict[str, Any]] = []
        self.decode_calls: list[dict[str, Any]] = []
        self.features: list[FakeBatchFeature] = []
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)

    def _make_features(self, audios: list[str]) -> FakeBatchFeature:
        tokens = [self.audio_to_token[audio] for audio in audios]
        feature = FakeBatchFeature(
            {
                "input_ids": torch.zeros((len(audios), 2), dtype=torch.long),
                "sample_ids": torch.tensor(tokens, dtype=torch.long),
            }
        )
        self.features.append(feature)
        return feature

    def apply_transcription_request(
        self,
        *,
        audio: list[str],
        language: list[str | None],
    ) -> FakeBatchFeature:
        audios = list(audio)
        self.transcription_calls.append(
            {"audio": audios, "language": list(language)}
        )
        return self._make_features(audios)

    def apply_chat_template(
        self,
        conversations: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> FakeBatchFeature | list[str]:
        audios: list[str] = []
        for conversation in conversations:
            user_message = next(message for message in conversation if message["role"] == "user")
            audio_content = user_message["content"][0]
            audios.append(audio_content.get("path", audio_content.get("audio")))
        self.chat_template_calls.append(
            {
                "conversations": conversations,
                "kwargs": dict(kwargs),
                "audio": audios,
            }
        )
        if not kwargs.get("tokenize", False):
            return [f"rendered:{audio}:" for audio in audios]
        return self._make_features(audios)

    def __call__(
        self,
        *,
        text: list[str],
        audio: list[str],
        **kwargs: Any,
    ) -> FakeBatchFeature:
        self.direct_calls.append(
            {"text": list(text), "audio": list(audio), "kwargs": dict(kwargs)}
        )
        return self._make_features(list(audio))

    def decode(self, generated_ids: torch.Tensor, **kwargs: Any) -> list[dict[str, Any]]:
        tokens = [int(row[0]) for row in generated_ids]
        self.decode_calls.append({"tokens": tokens, "kwargs": dict(kwargs)})
        return [
            {
                "language": self.detected_language_by_audio[audio],
                "transcription": self.transcript_by_audio[audio],
            }
            for audio in (self.token_to_audio[token] for token in tokens)
        ]


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(dict(kwargs))
        return torch.cat(
            (kwargs["input_ids"], kwargs["sample_ids"].reshape(-1, 1)),
            dim=1,
        )


class FakeAlignerProcessor:
    def __init__(self) -> None:
        self.prepare_calls: list[dict[str, Any]] = []
        self.decode_calls: list[dict[str, Any]] = []
        self.features: list[FakeBatchFeature] = []

    def prepare_forced_aligner_inputs(
        self,
        *,
        audio: list[str],
        transcript: list[str],
        language: list[str | None],
    ) -> tuple[FakeBatchFeature, list[list[str]]]:
        call_index = len(self.prepare_calls)
        self.prepare_calls.append(
            {
                "audio": list(audio),
                "transcript": list(transcript),
                "language": list(language),
            }
        )
        word_lists = [[text] for text in transcript]
        feature = FakeBatchFeature(
            {
                "input_ids": torch.full(
                    (len(audio), 2),
                    fill_value=call_index + 1,
                    dtype=torch.long,
                )
            }
        )
        self.features.append(feature)
        return feature, word_lists

    def decode_forced_alignment(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.decode_calls.append(dict(kwargs))
        return [
            [
                {
                    "text": words[0],
                    "start_time": float(index),
                    "end_time": float(index) + 0.5,
                }
            ]
            for index, words in enumerate(kwargs["word_lists"])
        ]


class FakeAlignerModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(timestamp_token_id=77)
        self.forward_calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls.append(dict(kwargs))
        batch_size = kwargs["input_ids"].shape[0]
        return SimpleNamespace(logits=torch.zeros((batch_size, 2, 3)))


def make_asr(
    audios: list[str],
    *,
    max_batch_size: int = 4,
    transcript_by_audio: dict[str, str] | None = None,
    detected_language_by_audio: dict[str, str | None] | None = None,
    with_aligner: bool = False,
) -> tuple[
    NativeQwen3ASRModel,
    FakeProcessor,
    FakeModel,
    FakeAlignerProcessor | None,
    FakeAlignerModel | None,
]:
    processor = FakeProcessor(
        audios,
        transcript_by_audio=transcript_by_audio,
        detected_language_by_audio=detected_language_by_audio,
    )
    model = FakeModel()
    aligner_processor = FakeAlignerProcessor() if with_aligner else None
    aligner_model = FakeAlignerModel() if with_aligner else None
    wrapper = NativeQwen3ASRModel(
        processor=processor,
        model=model,
        max_batch_size=max_batch_size,
        max_new_tokens=123,
        aligner_processor=aligner_processor,
        aligner_model=aligner_model,
    )
    return wrapper, processor, model, aligner_processor, aligner_model


class LanguageNormalizationTests(unittest.TestCase):
    def test_language_codes_and_names_are_canonicalized(self) -> None:
        cases = {
            "zh": "Chinese",
            " ZH ": "Chinese",
            "chINESE": "Chinese",
            "en": "English",
            " yue ": "Cantonese",
            "Portuguese": "Portuguese",
        }
        for supplied, expected in cases.items():
            with self.subTest(language=supplied):
                self.assertEqual(_canonical_language(supplied), expected)

        self.assertIsNone(_canonical_language(None))
        self.assertIsNone(_canonical_language("  "))

    def test_invalid_language_reports_supported_codes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported language: 'xx'.*Supported codes:.*zh",
        ):
            _canonical_language("xx")


class NativeTranscriptionTests(unittest.TestCase):
    def test_common_path_uses_apply_transcription_request(self) -> None:
        asr, processor, model, _, _ = make_asr(["sample.wav"])

        result = asr.transcribe("sample.wav", language=" zh ")

        self.assertEqual(
            processor.transcription_calls,
            [{"audio": ["sample.wav"], "language": ["Chinese"]}],
        )
        self.assertEqual(processor.chat_template_calls, [])
        self.assertEqual(result[0].text, "transcript:sample.wav")
        self.assertEqual(result[0].language, "Chinese")
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 123)
        self.assertIs(model.generate_calls[0]["do_sample"], False)
        self.assertEqual(
            processor.features[0].to_calls,
            [((torch.device("cpu"), torch.float32), {})],
        )
        self.assertEqual(processor.decode_calls[0]["kwargs"], {"return_format": "parsed"})

    def test_prompt_only_uses_chat_template_generation_prompt(self) -> None:
        asr, processor, _, _, _ = make_asr(["prompt.wav"])

        result = asr.transcribe("prompt.wav", context="product vocabulary")

        self.assertEqual(processor.transcription_calls, [])
        self.assertEqual(len(processor.chat_template_calls), 1)
        call = processor.chat_template_calls[0]
        self.assertEqual(
            call["kwargs"],
            {
                "tokenize": True,
                "return_dict": True,
                "add_generation_prompt": True,
            },
        )
        self.assertEqual(
            call["conversations"],
            [
                [
                    {"role": "system", "content": "product vocabulary"},
                    {
                        "role": "user",
                        "content": [{"type": "audio", "path": "prompt.wav"}],
                    },
                ]
            ],
        )
        self.assertEqual(result[0].language, "English")
        self.assertEqual(result[0].text, "transcript:prompt.wav")

    def test_prompt_and_language_appends_prefix_after_template_and_keeps_language(self) -> None:
        asr, processor, _, _, _ = make_asr(
            ["prefix.wav"],
            detected_language_by_audio={"prefix.wav": "English"},
        )

        with patch("native_asr.load_audio", side_effect=lambda item, sampling_rate: item):
            result = asr.transcribe(
                "prefix.wav",
                context="domain terms",
                language="zh",
            )

        call = processor.chat_template_calls[0]
        self.assertEqual(
            call["kwargs"],
            {
                "tokenize": False,
                "add_generation_prompt": True,
            },
        )
        self.assertEqual(call["conversations"][0][-1]["role"], "user")
        self.assertEqual(
            processor.direct_calls,
            [
                {
                    "text": ["rendered:prefix.wav:language Chinese<asr_text>"],
                    "audio": ["prefix.wav"],
                    "kwargs": {"return_tensors": "pt", "padding": True},
                }
            ],
        )
        self.assertEqual(result[0].language, "Chinese")

    def test_max_batch_size_chunks_requests_and_preserves_order(self) -> None:
        audios = [f"audio-{index}.wav" for index in range(7)]
        asr, processor, model, _, _ = make_asr(audios, max_batch_size=3)

        results = asr.transcribe(audios)

        self.assertEqual(
            [call["audio"] for call in processor.transcription_calls],
            [audios[0:3], audios[3:6], audios[6:7]],
        )
        self.assertEqual(
            [call["sample_ids"].numel() for call in model.generate_calls],
            [3, 3, 1],
        )
        self.assertEqual(
            [result.text for result in results],
            [f"transcript:{audio}" for audio in audios],
        )

    def test_mixed_template_groups_are_restored_to_original_order(self) -> None:
        audios = [f"mixed-{index}.wav" for index in range(6)]
        contexts = ["", "prompt one", "", "prompt three", "", "prompt five"]
        languages: list[str | None] = [None, None, "zh", "en", None, None]
        asr, processor, _, _, _ = make_asr(audios, max_batch_size=3)

        with patch("native_asr.load_audio", side_effect=lambda item, sampling_rate: item):
            results = asr.transcribe(audios, context=contexts, language=languages)

        self.assertEqual(
            [result.text for result in results],
            [f"transcript:{audio}" for audio in audios],
        )
        self.assertEqual(
            [call["audio"] for call in processor.transcription_calls],
            [[audios[0], audios[2]], [audios[4]]],
        )
        self.assertEqual(
            [call["audio"] for call in processor.chat_template_calls],
            [[audios[1]], [audios[5]], [audios[3]]],
        )
        self.assertEqual(
            [result.language for result in results][2:4],
            ["Chinese", "English"],
        )


class ForcedAlignmentTests(unittest.TestCase):
    def test_aligner_uses_native_prepare_and_decode_in_chunks(self) -> None:
        audios = ["align-0.wav", "align-1.wav", "align-2.wav"]
        asr, _, _, aligner_processor, aligner_model = make_asr(
            audios,
            max_batch_size=2,
            with_aligner=True,
        )
        assert aligner_processor is not None
        assert aligner_model is not None

        timestamps = asr.align(
            audio=audios,
            text=["zero", "one", "two"],
            language=["en", "zh", "Japanese"],
        )

        self.assertEqual(
            aligner_processor.prepare_calls,
            [
                {
                    "audio": audios[:2],
                    "transcript": ["zero", "one"],
                    "language": ["English", "Chinese"],
                },
                {
                    "audio": audios[2:],
                    "transcript": ["two"],
                    "language": ["Japanese"],
                },
            ],
        )
        self.assertEqual(len(aligner_model.forward_calls), 2)
        self.assertEqual(len(aligner_processor.decode_calls), 2)
        self.assertTrue(
            all(
                call["timestamp_token_id"] == 77
                for call in aligner_processor.decode_calls
            )
        )
        self.assertEqual(
            [[item["text"] for item in sample] for sample in timestamps],
            [["zero"], ["one"], ["two"]],
        )
        self.assertTrue(
            all(
                feature.to_calls == [((torch.device("cpu"), torch.float32), {})]
                for feature in aligner_processor.features
            )
        )

    def test_transcription_attaches_alignment_and_empty_text_gets_empty_timestamps(self) -> None:
        audios = ["spoken.wav", "silent.wav"]
        asr, _, _, aligner_processor, _ = make_asr(
            audios,
            transcript_by_audio={"spoken.wav": "hello world", "silent.wav": ""},
            with_aligner=True,
        )
        assert aligner_processor is not None

        results = asr.transcribe(
            audios,
            language=["en", "zh"],
            return_time_stamps=True,
        )

        self.assertEqual(
            aligner_processor.prepare_calls,
            [
                {
                    "audio": ["spoken.wav"],
                    "transcript": ["hello world"],
                    "language": ["English"],
                }
            ],
        )
        self.assertEqual(
            results[0].time_stamps,
            [{"text": "hello world", "start_time": 0.0, "end_time": 0.5}],
        )
        self.assertEqual(results[1].time_stamps, [])


if __name__ == "__main__":
    unittest.main()
