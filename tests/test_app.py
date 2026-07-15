# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
        self.assertFalse(config["compile_asr"])
        self.assertFalse(config["compile_aligner"])


if __name__ == "__main__":
    unittest.main()
