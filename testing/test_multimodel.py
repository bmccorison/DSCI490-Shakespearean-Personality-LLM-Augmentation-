import unittest

from pipeline.multimodel import (
    HARD_MAX_TURNS,
    MultiModelConversation,
    MultiModelParticipant,
)


def make_participant(index: int) -> MultiModelParticipant:
    return MultiModelParticipant(
        name=f"Speaker {index}",
        character=f"Character {index}",
        work="Hamlet",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path="__base__",
    )


class FakeTokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, return_tensors=None):
        self.prompts.append(prompt)
        return {"prompt": prompt, "return_tensors": return_tensors}


class MultiModelConversationTests(unittest.TestCase):
    def test_rejects_invalid_participant_counts(self):
        with self.assertRaises(ValueError):
            MultiModelConversation([make_participant(1)], "Begin.")

        with self.assertRaises(ValueError):
            MultiModelConversation(
                [make_participant(index) for index in range(1, 6)],
                "Begin.",
            )

    def test_rejects_turn_limits_above_hard_cap(self):
        with self.assertRaises(ValueError):
            MultiModelConversation(
                [make_participant(1), make_participant(2)],
                "Begin.",
                max_turns=HARD_MAX_TURNS + 1,
            )

    def test_generates_round_robin_turns_and_prompt_context(self):
        tokenizer = FakeTokenizer()
        loaded_models = []

        def fake_loader(model_name, adapter_path):
            loaded_models.append((model_name, adapter_path))
            return object(), tokenizer

        def fake_response(tokenized_chat, model, tokenizer, apply_shakespeare_style=True):
            return f"reply {len(tokenizer.prompts)}"

        conversation = MultiModelConversation(
            [make_participant(1), make_participant(2), make_participant(3)],
            "Begin with a question.",
            max_turns=4,
        )

        turns = [
            conversation.generate_next_turn(fake_loader, fake_response)
            for _ in range(4)
        ]

        self.assertEqual(
            [turn.speaker_name for turn in turns],
            ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 1"],
        )
        self.assertEqual(len(loaded_models), 4)
        self.assertIn("Initial prompt: Begin with a question.", tokenizer.prompts[0])
        self.assertIn("No one has spoken yet.", tokenizer.prompts[0])
        self.assertIn("Speaker 1 (Character 1): reply 1", tokenizer.prompts[1])
        self.assertTrue(conversation.is_complete)

    def test_stop_prevents_later_generation(self):
        conversation = MultiModelConversation(
            [make_participant(1), make_participant(2)],
            "Begin.",
        )
        conversation.stop()

        def unused_loader(model_name, adapter_path):
            raise AssertionError("Loader should not be called after stop.")

        self.assertIsNone(conversation.generate_next_turn(unused_loader))
        self.assertEqual(conversation.status, "stopped")

    def test_stop_during_generation_discards_response(self):
        tokenizer = FakeTokenizer()
        conversation = MultiModelConversation(
            [make_participant(1), make_participant(2)],
            "Begin.",
        )

        def fake_loader(model_name, adapter_path):
            return object(), tokenizer

        def stopping_response(tokenized_chat, model, tokenizer, apply_shakespeare_style=True):
            conversation.stop()
            return "late reply"

        self.assertIsNone(
            conversation.generate_next_turn(fake_loader, stopping_response)
        )
        self.assertEqual(conversation.turns, [])
        self.assertEqual(conversation.status, "stopped")


if __name__ == "__main__":
    unittest.main()
