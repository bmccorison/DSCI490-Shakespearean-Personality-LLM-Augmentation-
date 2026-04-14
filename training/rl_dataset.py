# training/rl_dataset.py

from lora_5 import _coerce_messages, render_message_history


def build_rl_prompts(message_records, limit=0):
    prompts = []

    for record in message_records:
        messages = _coerce_messages(record["messages"])

        # need at least 1 user + 1 assistant
        if len(messages) < 2:
            continue

        # remove final assistant response → becomes prompt
        prompt_messages = messages[:-1]

        prompt_text = render_message_history(
            prompt_messages,
            append_assistant_header=True,
        )

        prompts.append(prompt_text)

    if limit:
        prompts = prompts[:limit]

    return prompts
