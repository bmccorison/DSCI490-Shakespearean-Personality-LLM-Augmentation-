'''Shared fixtures and helpers for the test suite.'''

import os

import pytest

# Prefix every log file written during a test run with logging/<date>/test/.
# Set before any project import so the override is in place when loggers are created.
os.environ.setdefault("SHAKESPEARE_LOG_CATEGORY", "test")


def make_participant(index: int, model_name: str = "ModelA", adapter_path: str = "__base__"):
    '''Build a valid MultiModelParticipant for use in tests.'''
    from pipeline.multimodel import MultiModelParticipant
    return MultiModelParticipant(
        name=f"Speaker{index}",
        character=f"Character{index}",
        work="Hamlet",
        model_name=model_name,
        adapter_path=adapter_path,
    )


def make_conversation(n_participants: int = 2, max_turns: int = 4, **kwargs):
    '''Build a MultiModelConversation with fake participants.'''
    from pipeline.multimodel import MultiModelConversation
    return MultiModelConversation(
        participants=[make_participant(i) for i in range(1, n_participants + 1)],
        initial_prompt="Begin.",
        max_turns=max_turns,
        **kwargs,
    )


class FakeTokenizer:
    '''Minimal tokenizer stub that records calls without touching torch.'''

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, return_tensors=None):
        self.calls.append(prompt)
        return {"input_ids": [[1, 2, 3]], "prompt": prompt}


def fake_loader(model_name, adapter_path):
    '''Model loader stub that returns sentinel objects — no disk access.'''
    return object(), FakeTokenizer()


def make_response_generator(prefix: str = "reply"):
    '''Return a response generator stub that yields deterministic text.'''
    call_count = [0]

    def generator(tokenized_chat, model, tokenizer, apply_shakespeare_style=True):
        call_count[0] += 1
        return f"{prefix} {call_count[0]}"

    return generator
