# -*- coding: utf-8 -*-
from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import app as app_module


class TimestampSerializationTests(unittest.TestCase):
    def test_serializes_native_dict_and_legacy_attribute_objects(self) -> None:
        items = [
            {
                "text": "native",
                "start_time": 0.125,
                "end_time": 0.75,
                "ignored": "value",
            },
            SimpleNamespace(text="legacy", start_time=1.0, end_time=1.5),
        ]

        self.assertEqual(
            app_module._serialize_timestamps(items),
            [
                {"text": "native", "start_time": 0.125, "end_time": 0.75},
                {"text": "legacy", "start_time": 1.0, "end_time": 1.5},
            ],
        )
        self.assertEqual(app_module._serialize_timestamps(None), [])


class EnvironmentConfigTests(unittest.TestCase):
    def test_default_dtype_policy_is_cpu_float16_and_cuda_float32(self) -> None:
        self.assertEqual(app_module._default_dtype("cpu"), "float16")
        self.assertEqual(app_module._default_dtype("cuda:0"), "float32")

    def test_env_bool_accepts_common_values_and_default(self) -> None:
        for value in ("1", "true", "TRUE", " yes ", "y", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"TEST_BOOLEAN": value},
                clear=True,
            ):
                self.assertTrue(app_module._env_bool("TEST_BOOLEAN"))

        for value in ("0", "false", "FALSE", " no ", "n", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"TEST_BOOLEAN": value},
                clear=True,
            ):
                self.assertFalse(app_module._env_bool("TEST_BOOLEAN", default=True))

        for environment in ({}, {"TEST_BOOLEAN": "  "}):
            with self.subTest(environment=environment), patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                self.assertTrue(app_module._env_bool("TEST_BOOLEAN", default=True))

    def test_env_bool_rejects_ambiguous_value(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOLEAN": "sometimes"}, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                r"Environment variable TEST_BOOLEAN must be true or false",
            ):
                app_module._env_bool("TEST_BOOLEAN")

    def test_load_config_defaults_to_hugging_face_model_on_cpu(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(app_module, "_auto_device", return_value="cpu"),
            patch.object(app_module.os, "cpu_count", return_value=8),
        ):
            config = app_module._load_config()

        self.assertEqual(config["model_name"], "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertIsNone(config["aligner_model_name"])
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["dtype_name"], "float16")
        self.assertIs(config["dtype"], app_module.torch.float16)
        self.assertEqual(config["max_batch_size"], 2)
        self.assertEqual(config["max_new_tokens"], 256)
        self.assertEqual(config["ffmpeg_binary"], "ffmpeg")
        self.assertEqual(config["ffmpeg_timeout_seconds"], 120)
        self.assertFalse(config["compile_asr"])
        self.assertFalse(config["compile_aligner"])

    def test_ffmpeg_timeout_must_be_positive(self) -> None:
        with patch.dict(
            os.environ,
            {"QWEN_ASR_FFMPEG_TIMEOUT": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"QWEN_ASR_FFMPEG_TIMEOUT must be greater than zero",
            ):
                app_module._load_config()


class AudioConversionTests(unittest.TestCase):
    @unittest.skipUnless(app_module._resolve_ffmpeg("ffmpeg"), "ffmpeg is not installed")
    def test_real_ffmpeg_converts_mpeg_to_expected_wav(self) -> None:
        ffmpeg_path = app_module._resolve_ffmpeg("ffmpeg")
        assert ffmpeg_path is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = str(Path(temp_dir, "source.mpeg"))
            subprocess.run(
                [
                    ffmpeg_path,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.1",
                    "-c:a",
                    "mp2",
                    "-f",
                    "mpeg",
                    source_path,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=10,
            )

            output_path = app_module._convert_audio_to_wav(
                source_path,
                ffmpeg_path=ffmpeg_path,
                temp_dir=temp_dir,
                timeout_seconds=10,
                original_filename="source.mpeg",
            )

            with wave.open(output_path, "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 16_000)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertGreater(audio.getnframes(), 0)

    def test_ffmpeg_converts_to_16khz_mono_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = str(Path(temp_dir, "source.mpeg"))
            Path(source_path).write_bytes(b"input")

            with patch.object(app_module.subprocess, "run") as run:
                output_path = app_module._convert_audio_to_wav(
                    source_path,
                    ffmpeg_path="test-ffmpeg",
                    temp_dir=temp_dir,
                    timeout_seconds=17,
                    original_filename="sample.mpeg",
                )

            command = run.call_args.args[0]
            self.assertEqual(command[0], "test-ffmpeg")
            self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file,pipe")
            self.assertEqual(command[command.index("-map") + 1], "0:a:0")
            self.assertEqual(command[command.index("-ac") + 1], "1")
            self.assertEqual(command[command.index("-ar") + 1], "16000")
            self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
            self.assertEqual(command[command.index("-f") + 1], "wav")
            self.assertEqual(command[-1], output_path)
            self.assertEqual(run.call_args.kwargs["timeout"], 17)
            self.assertTrue(output_path.endswith(".wav"))

    def test_invalid_audio_becomes_http_400_and_cleans_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = str(Path(temp_dir, "source.mpeg"))
            Path(source_path).write_bytes(b"not audio")
            failure = subprocess.CalledProcessError(
                returncode=1,
                cmd=["test-ffmpeg"],
                stderr=f"{source_path}: Invalid data found when processing input",
            )

            with (
                patch.object(app_module.subprocess, "run", side_effect=failure),
                self.assertRaises(HTTPException) as raised,
            ):
                app_module._convert_audio_to_wav(
                    source_path,
                    ffmpeg_path="test-ffmpeg",
                    temp_dir=temp_dir,
                    timeout_seconds=17,
                    original_filename="broken.mpeg",
                )

            self.assertEqual(raised.exception.status_code, 400)
            self.assertIn("broken.mpeg", raised.exception.detail)
            self.assertIn("Invalid data found", raised.exception.detail)
            self.assertIn("<uploaded audio>", raised.exception.detail)
            self.assertNotIn(source_path, raised.exception.detail)
            self.assertEqual(
                [path.suffix for path in Path(temp_dir).iterdir()],
                [".mpeg"],
            )

    def test_missing_ffmpeg_becomes_http_503(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            app_module._convert_audio_to_wav(
                "source.mpeg",
                ffmpeg_path=None,
                temp_dir=None,
                timeout_seconds=17,
                original_filename="sample.mpeg",
            )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("convert_audio=false", raised.exception.detail)

    def test_timeout_becomes_http_408_and_cleans_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = str(Path(temp_dir, "source.mpeg"))
            Path(source_path).write_bytes(b"input")

            with (
                patch.object(
                    app_module.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired(["test-ffmpeg"], 3),
                ),
                self.assertRaises(HTTPException) as raised,
            ):
                app_module._convert_audio_to_wav(
                    source_path,
                    ffmpeg_path="test-ffmpeg",
                    temp_dir=temp_dir,
                    timeout_seconds=3,
                    original_filename="slow.mpeg",
                )

            self.assertEqual(raised.exception.status_code, 408)
            self.assertIn("3 seconds", raised.exception.detail)
            self.assertEqual(
                [path.suffix for path in Path(temp_dir).iterdir()],
                [".mpeg"],
            )

    def test_unconverted_upload_is_preserved_for_model_and_uses_safe_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload = UploadFile(
                file=BytesIO(b"raw upload"),
                filename="unsafe.:name",
            )
            config = {
                "temp_dir": temp_dir,
                "max_upload_bytes": 1024,
            }

            path = app_module._prepare_audio_upload(
                upload,
                config,
                convert_audio=False,
            )
            try:
                self.assertEqual(Path(path).suffix, ".upload")
                self.assertEqual(Path(path).read_bytes(), b"raw upload")
            finally:
                app_module._unlink_temp_file(path)
                upload.file.close()


class FakeApiModel:
    has_aligner = False

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.audio_bytes: bytes | None = None

    def transcribe(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        audio_path = Path(str(kwargs["audio"]))
        self.audio_bytes = audio_path.read_bytes()
        return [SimpleNamespace(text="ok", language="English")]


class AudioConversionApiTests(unittest.TestCase):
    @staticmethod
    def _config(temp_dir: str, ffmpeg_path: str | None = "test-ffmpeg") -> dict[str, object]:
        return {
            "temp_dir": temp_dir,
            "max_upload_bytes": 1024 * 1024,
            "ffmpeg_path": ffmpeg_path,
            "ffmpeg_timeout_seconds": 30,
            "model_name": "test-model",
            "aligner_model_name": None,
        }

    def test_convert_audio_defaults_to_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeApiModel()
            client = TestClient(app_module.app)
            with (
                patch.dict(
                    app_module.STATE,
                    {"config": self._config(temp_dir), "model": model},
                    clear=True,
                ),
                patch.object(app_module.subprocess, "run") as run,
            ):
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("sample.mpeg", b"raw mpeg", "audio/mpeg")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "ok")
            self.assertEqual(model.audio_bytes, b"raw mpeg")
            self.assertTrue(str(model.calls[0]["audio"]).endswith(".mpeg"))
            run.assert_not_called()
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_convert_audio_true_sends_normalized_wav_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeApiModel()
            client = TestClient(app_module.app)

            def fake_run(command: list[str], **_: object) -> None:
                Path(command[-1]).write_bytes(b"normalized wav")

            with (
                patch.dict(
                    app_module.STATE,
                    {"config": self._config(temp_dir), "model": model},
                    clear=True,
                ),
                patch.object(app_module.subprocess, "run", side_effect=fake_run) as run,
            ):
                response = client.post(
                    "/v1/audio/transcriptions",
                    data={"convert_audio": "true"},
                    files={"file": ("sample.mpeg", b"raw mpeg", "audio/mpeg")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(model.audio_bytes, b"normalized wav")
            self.assertTrue(str(model.calls[0]["audio"]).endswith(".wav"))
            run.assert_called_once()
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_ffmpeg_failure_is_returned_as_json_http_400(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeApiModel()
            client = TestClient(app_module.app)
            failure = subprocess.CalledProcessError(
                returncode=1,
                cmd=["test-ffmpeg"],
                stderr="Invalid data found when processing input",
            )
            with (
                patch.dict(
                    app_module.STATE,
                    {"config": self._config(temp_dir), "model": model},
                    clear=True,
                ),
                patch.object(app_module.subprocess, "run", side_effect=failure),
            ):
                response = client.post(
                    "/v1/audio/transcriptions",
                    data={"convert_audio": "true"},
                    files={"file": ("broken.mpeg", b"not audio", "audio/mpeg")},
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("ffmpeg could not decode", response.json()["detail"])
            self.assertEqual(model.calls, [])
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_raw_decoder_failure_is_returned_as_json_http_400(self) -> None:
        decoder_error_type = type(
            "NoBackendError",
            (Exception,),
            {"__module__": "audioread.exceptions"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeApiModel(error=decoder_error_type())
            client = TestClient(app_module.app)
            with patch.dict(
                app_module.STATE,
                {"config": self._config(temp_dir), "model": model},
                clear=True,
            ):
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("unsupported.mpeg", b"raw mpeg", "audio/mpeg")},
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("convert_audio=true", response.json()["detail"])
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
