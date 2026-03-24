import importlib
import sys
import types
import unittest


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_ADAPTER_PATH = "__base__"
ADAPTER_PATH = "models/lora_finetuned_model1"


def install_transformer_stubs() -> None:
    torch_module = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_module.cuda = _Cuda()
    torch_module.bfloat16 = "bfloat16"
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"

    peft_module = types.ModuleType("peft")

    class DummyLoadedModel:
        def eval(self):
            return None

    class DummyPeftModel:
        @staticmethod
        def from_pretrained(base_model, path):
            return DummyLoadedModel()

    peft_module.PeftModel = DummyPeftModel

    transformers_module = types.ModuleType("transformers")

    class DummyTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.eos_token_id = 2
            self.pad_token_id = 0

        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": "tokenized-chat", "attention_mask": "attention-mask"}

        def apply_chat_template(self, *args, **kwargs):
            return "tokenized-chat"

        def decode(self, *args, **kwargs):
            return "decoded reply"

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(source):
            return DummyTokenizer()

    class DummyBaseModel:
        def eval(self):
            return None

        def generate(self, *args, **kwargs):
            return [[1, 2, 3]]

    class DummyAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyBaseModel()

    transformers_module.AutoTokenizer = DummyAutoTokenizer
    transformers_module.AutoModelForCausalLM = DummyAutoModelForCausalLM

    sys.modules["torch"] = torch_module
    sys.modules["peft"] = peft_module
    sys.modules["transformers"] = transformers_module


def load_lm_generation_module():
    install_transformer_stubs()
    sys.modules.pop("pipeline.lm_generation", None)
    return importlib.import_module("pipeline.lm_generation")


class LmGenerationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm_generation = load_lm_generation_module()

    def setUp(self):
        self.lm_generation.set_character_context("Hamlet", "Hamlet")

    def test_model_selection_returns_only_loadable_models(self):
        models = self.lm_generation.model_selection()

        self.assertEqual(1, len(models))
        self.assertEqual(MODEL_NAME, models[0]["name"])
        self.assertEqual(BASE_ADAPTER_PATH, models[0]["default_adapter_path"])
        self.assertEqual(
            ["base_chat", "hamlet_lora_1", "hamlet_lora_2"],
            [adapter["name"] for adapter in models[0]["adapters"]],
        )

    def test_model_selection_accepts_legacy_adapter_paths_shape(self):
        original_model_list = self.lm_generation.model_list
        self.lm_generation.model_list = lambda: [
            {
                "name": MODEL_NAME,
                "description": "legacy",
                "adapter_paths": [
                    {
                        "hamlet_lora_legacy": ADAPTER_PATH,
                        "description": "legacy adapter shape",
                    }
                ],
            }
        ]
        try:
            models = self.lm_generation.model_selection()
        finally:
            self.lm_generation.model_list = original_model_list

        self.assertEqual(1, len(models))
        self.assertEqual("hamlet_lora_legacy", models[0]["adapters"][0]["name"])
        self.assertEqual(ADAPTER_PATH, models[0]["adapters"][0]["path"])

    def test_set_character_context_resets_messages(self):
        self.lm_generation.add_chat_history(user_msg="What news?")

        self.lm_generation.set_character_context("Ophelia", "Hamlet")

        self.assertEqual(1, len(self.lm_generation.messages))
        self.assertIn("You are Ophelia from Shakespeare's work Hamlet.", self.lm_generation.messages[0]["content"])

    def test_get_model_rejects_empty_adapter_path(self):
        with self.assertRaisesRegex(ValueError, "Adapter path is required."):
            self.lm_generation.get_model(MODEL_NAME, "")

    def test_get_model_loads_base_selection(self):
        model, tokenizer = self.lm_generation.get_model(MODEL_NAME, BASE_ADAPTER_PATH)

        self.assertIsNotNone(model)
        self.assertEqual("<eos>", tokenizer.pad_token)

    def test_get_model_rejects_invalid_model_adapter_pair(self):
        with self.assertRaisesRegex(ValueError, "is not valid for model"):
            self.lm_generation.get_model(MODEL_NAME, "models/missing_adapter")

    def test_get_model_loads_valid_selection(self):
        model, tokenizer = self.lm_generation.get_model(MODEL_NAME, ADAPTER_PATH)

        self.assertIsNotNone(model)
        self.assertEqual("<eos>", tokenizer.pad_token)

    def test_add_chat_history_trims_old_turns(self):
        original_limit = self.lm_generation.MAX_CHAT_HISTORY_TURNS
        self.lm_generation.MAX_CHAT_HISTORY_TURNS = 2
        try:
            self.lm_generation.refresh_chat_history()
            self.lm_generation.add_chat_history(user_msg="Question 1", model_response="Reply 1")
            self.lm_generation.add_chat_history(user_msg="Question 2", model_response="Reply 2")
            self.lm_generation.add_chat_history(user_msg="Question 3")

            self.assertEqual(
                [
                    {"role": "system", "content": self.lm_generation.get_system_prompt()},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Reply 2"},
                    {"role": "user", "content": "Question 3"},
                ],
                self.lm_generation.messages,
            )
        finally:
            self.lm_generation.MAX_CHAT_HISTORY_TURNS = original_limit

    def test_generate_response_accepts_mapping_chat_template(self):
        class DummyTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def to(self, device):
                return self

        class RecordingModel:
            def __init__(self):
                self.device = "cpu"
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[11, 12, 13, 21, 22]]

        class RecordingTokenizer:
            def __init__(self):
                self.decoded_tokens = None
                self.eos_token_id = 99
                self.pad_token_id = None

            def decode(self, tokens, skip_special_tokens=True):
                self.decoded_tokens = tokens
                return "Hamlet reply"

        model = RecordingModel()
        tokenizer = RecordingTokenizer()

        response = self.lm_generation.generate_response(
            {
                "input_ids": DummyTensor([11, 12, 13]),
                "attention_mask": DummyTensor([1, 1, 1]),
            },
            model,
            tokenizer,
        )

        self.assertEqual("Hamlet reply", response)
        self.assertEqual([21, 22], tokenizer.decoded_tokens)
        self.assertIn("input_ids", model.kwargs)
        self.assertIn("attention_mask", model.kwargs)
        self.assertEqual(self.lm_generation.MAX_NEW_TOKENS, model.kwargs["max_new_tokens"])
        self.assertTrue(model.kwargs["do_sample"])
        self.assertEqual(1, model.kwargs["num_beams"])
        self.assertTrue(model.kwargs["use_cache"])
        self.assertEqual(self.lm_generation.TEMPERATURE, model.kwargs["temperature"])
        self.assertEqual(self.lm_generation.TOP_P, model.kwargs["top_p"])
        self.assertEqual(self.lm_generation.REPETITION_PENALTY, model.kwargs["repetition_penalty"])
        self.assertEqual(self.lm_generation.NO_REPEAT_NGRAM_SIZE, model.kwargs["no_repeat_ngram_size"])
        self.assertEqual(99, model.kwargs["pad_token_id"])
        self.assertEqual(99, model.kwargs["eos_token_id"])

    def test_generate_response_falls_back_to_base_model_when_adapter_is_degenerate(self):
        class DummyTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def to(self, device):
                return self

        class RecordingModel:
            def __init__(self):
                self.device = "cpu"
                self.kwargs_history = []
                self._use_base = False

            def generate(self, **kwargs):
                self.kwargs_history.append(kwargs)
                if not self._use_base:
                    return [[11, 12, 13, 21, 21, 21, 21, 21]]
                return [[11, 12, 13, 31, 32, 33]]

            def disable_adapter(self):
                parent = self

                class _DisabledAdapterContext:
                    def __enter__(self_inner):
                        parent._use_base = True
                        return parent

                    def __exit__(self_inner, exc_type, exc, tb):
                        parent._use_base = False

                return _DisabledAdapterContext()

        class RecordingTokenizer:
            def __init__(self):
                self.eos_token_id = 99
                self.pad_token_id = 99

            def decode(self, tokens, skip_special_tokens=True):
                if list(tokens) == [21, 21, 21, 21, 21]:
                    return "I I I I I"
                return "To be resolved."

        model = RecordingModel()
        tokenizer = RecordingTokenizer()

        response = self.lm_generation.generate_response(
            {
                "input_ids": DummyTensor([11, 12, 13]),
                "attention_mask": DummyTensor([1, 1, 1]),
            },
            model,
            tokenizer,
        )

        self.assertEqual("To be resolved.", response)
        self.assertEqual(2, len(model.kwargs_history))

    def test_post_processing_applies_shakespeare_style_when_enabled(self):
        processed = self.lm_generation.post_processing(
            "You are your own counsel before the court.",
            apply_shakespeare_style=True,
        )

        self.assertEqual("Thou art thy own counsel ere the court.", processed)

    def test_post_processing_skips_shakespeare_style_when_disabled(self):
        processed = self.lm_generation.post_processing(
            "You are your own counsel before the court.",
            apply_shakespeare_style=False,
        )

        self.assertEqual("You are your own counsel before the court.", processed)


if __name__ == "__main__":
    unittest.main()
