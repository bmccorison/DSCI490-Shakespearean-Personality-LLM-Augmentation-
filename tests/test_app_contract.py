import importlib
import sys
import types
import unittest
from unittest import mock

try:
    from fastapi import HTTPException
except ModuleNotFoundError:
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


def install_runtime_stubs() -> None:
    if "fastapi" not in sys.modules:
        fastapi_module = types.ModuleType("fastapi")

        class FastAPI:
            def get(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

        class Response:
            def __init__(self, content=b"", media_type=None):
                self.body = content
                self.media_type = media_type

        fastapi_module.FastAPI = FastAPI
        fastapi_module.HTTPException = HTTPException
        fastapi_module.Response = Response
        sys.modules["fastapi"] = fastapi_module

    if "scipy" not in sys.modules:
        scipy_module = types.ModuleType("scipy")
        scipy_io_module = types.ModuleType("scipy.io")
        wavfile_module = types.ModuleType("scipy.io.wavfile")
        wavfile_module.write = lambda *args, **kwargs: None
        scipy_io_module.wavfile = wavfile_module
        scipy_module.io = scipy_io_module
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.io"] = scipy_io_module
        sys.modules["scipy.io.wavfile"] = wavfile_module

    if "uvicorn" not in sys.modules:
        uvicorn_module = types.ModuleType("uvicorn")
        uvicorn_module.run = lambda *args, **kwargs: None
        sys.modules["uvicorn"] = uvicorn_module


def install_pipeline_stub() -> None:
    pipeline_module = types.ModuleType("pipeline")
    pipeline_module.__path__ = []

    lm_generation_module = types.ModuleType("pipeline.lm_generation")
    lm_generation_module.generate_output = lambda *args, **kwargs: "reply"
    lm_generation_module.refresh_chat_history = lambda: None
    lm_generation_module.model_selection = lambda: []
    lm_generation_module.get_model = lambda *args, **kwargs: ("model", "tokenizer")
    lm_generation_module.set_character_context = lambda *args, **kwargs: None

    pipeline_module.lm_generation = lm_generation_module
    sys.modules["pipeline"] = pipeline_module
    sys.modules["pipeline.lm_generation"] = lm_generation_module


def load_app_module():
    original_pipeline = sys.modules.get("pipeline")
    original_lm_generation = sys.modules.get("pipeline.lm_generation")
    original_fastapi = sys.modules.get("fastapi")
    original_scipy = sys.modules.get("scipy")
    original_scipy_io = sys.modules.get("scipy.io")
    original_wavfile = sys.modules.get("scipy.io.wavfile")
    original_uvicorn = sys.modules.get("uvicorn")

    install_runtime_stubs()
    install_pipeline_stub()
    sys.modules.pop("app", None)
    sys.modules.pop("bark", None)

    try:
        return importlib.import_module("app")
    finally:
        if original_pipeline is None:
            sys.modules.pop("pipeline", None)
        else:
            sys.modules["pipeline"] = original_pipeline

        if original_lm_generation is None:
            sys.modules.pop("pipeline.lm_generation", None)
        else:
            sys.modules["pipeline.lm_generation"] = original_lm_generation

        if original_fastapi is None:
            sys.modules.pop("fastapi", None)
        else:
            sys.modules["fastapi"] = original_fastapi

        if original_scipy is None:
            sys.modules.pop("scipy", None)
        else:
            sys.modules["scipy"] = original_scipy

        if original_scipy_io is None:
            sys.modules.pop("scipy.io", None)
        else:
            sys.modules["scipy.io"] = original_scipy_io

        if original_wavfile is None:
            sys.modules.pop("scipy.io.wavfile", None)
        else:
            sys.modules["scipy.io.wavfile"] = original_wavfile

        if original_uvicorn is None:
            sys.modules.pop("uvicorn", None)
        else:
            sys.modules["uvicorn"] = original_uvicorn


class AppContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app_module()

    def test_app_import_does_not_load_bark(self):
        self.assertNotIn("bark", sys.modules)

    def test_generate_tts_rejects_blank_text(self):
        with self.assertRaises(HTTPException) as context:
            self.app_module.generate_tts("   ")

        self.assertEqual(400, context.exception.status_code)
        self.assertEqual("Text is required.", context.exception.detail)

    def test_generate_tts_returns_fallback_audio_when_bark_is_unavailable(self):
        with mock.patch.object(
            self.app_module,
            "_load_bark_dependencies",
            side_effect=RuntimeError("TTS backend is unavailable."),
        ), mock.patch.object(
            self.app_module,
            "_generate_fallback_tts_audio",
            return_value=b"fallback-wav",
        ):
            response = self.app_module.generate_tts("Speak, I pray thee.")

        self.assertEqual("audio/wav", response.media_type)
        self.assertEqual(b"fallback-wav", response.body)

    def test_generate_tts_returns_503_when_all_backends_are_unavailable(self):
        with mock.patch.object(
            self.app_module,
            "_load_bark_dependencies",
            side_effect=RuntimeError("TTS backend is unavailable."),
        ), mock.patch.object(
            self.app_module,
            "_generate_fallback_tts_audio",
            side_effect=RuntimeError("Fallback unavailable."),
        ):
            with self.assertRaises(HTTPException) as context:
                self.app_module.generate_tts("Speak, I pray thee.")

        self.assertEqual(503, context.exception.status_code)
        self.assertIn("Fallback unavailable.", context.exception.detail)

    def test_generate_tts_returns_audio_response_when_dependencies_load(self):
        with mock.patch.object(
            self.app_module,
            "_load_bark_dependencies",
            return_value=(lambda *args, **kwargs: [0, 1, 0, -1], 22050),
        ), mock.patch.object(
            self.app_module,
            "_resolve_bark_use_gpu",
            return_value=False,
        ), mock.patch.object(
            self.app_module.wav,
            "write",
            side_effect=lambda buffer, sample_rate, data: buffer.write(b"wav-bytes"),
        ):
            response = self.app_module.generate_tts("Speak, I pray thee.")

        self.assertEqual("audio/wav", response.media_type)
        self.assertEqual(b"wav-bytes", response.body)

    def test_generate_response_uses_model_loaded_by_select_model(self):
        selected_model = object()
        selected_tokenizer = object()

        with mock.patch.object(
            self.app_module,
            "get_model",
            return_value=(selected_model, selected_tokenizer),
        ) as get_model_mock, mock.patch.object(
            self.app_module,
            "generate_output",
            return_value="reply-from-selected-model",
        ) as generate_output_mock:
            select_response = self.app_module.select_model(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "__base__",
            )
            response = self.app_module.generate_response_endpoint(
                "Who are you?",
                shakespeare_style=False,
            )

        self.assertEqual("Model loaded.", select_response["message"])
        get_model_mock.assert_called_once_with(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "__base__",
        )
        generate_output_mock.assert_called_once_with(
            "Who are you?",
            selected_tokenizer,
            selected_model,
            None,
            apply_shakespeare_style=False,
        )
        self.assertEqual({"response": "reply-from-selected-model"}, response)


if __name__ == "__main__":
    unittest.main()
