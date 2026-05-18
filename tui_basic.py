"""Basic numbered CLI mirroring the web demo's chat flow.

Loads a model + adapter via the same hot-swap machinery the FastAPI backend uses,
applies the adapter's character metadata, retrieves RAG context per turn, and
exposes runtime commands that mirror toggles available in `interface/src/App.jsx`.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
HELP_COMMANDS = {"help", "/help"}
RESET_COMMANDS = {"reset", "/reset"}
STYLE_COMMANDS = {"style", "/style"}
CHARACTER_COMMAND_PREFIXES = ("/character", "character ")


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Parser help formatter that preserves dynamic numbering blocks."""


def _default_adapter_index(model_info: dict[str, Any]) -> int:
    """Return the 1-based default adapter index for a model."""
    default_path = str(model_info.get("default_adapter_path", "")).strip()
    for index, adapter in enumerate(model_info["adapters"], start=1):
        if adapter["path"] == default_path:
            return index
    return 1


def _build_help_epilog(models: list[dict[str, Any]]) -> str:
    """Render the numbered model + adapter list shown in `--help`."""
    if not models:
        return "No models are currently available."

    lines = [
        "Examples:",
        "  python tui_basic.py --model 1",
        "  python tui_basic.py --model 1 --adapter 3",
        "",
        "Available models and adapters:",
    ]

    for model_index, model_info in enumerate(models, start=1):
        lines.append(f"  {model_index}. {model_info['name']}")
        if model_info.get("description"):
            lines.append(f"     {model_info['description']}")

        default_adapter_index = _default_adapter_index(model_info)
        lines.append("     adapters:")
        for adapter_index, adapter in enumerate(model_info["adapters"], start=1):
            suffix = " [default]" if adapter_index == default_adapter_index else ""
            lines.append(
                f"       {adapter_index}. {adapter['name']} -> {adapter['path']}{suffix}"
            )
            if adapter.get("description"):
                lines.append(f"          {adapter['description']}")

    return "\n".join(lines)


def build_parser(models: list[dict[str, Any]]) -> argparse.ArgumentParser:
    """Build the CLI parser with a dynamic numbered help block."""
    parser = argparse.ArgumentParser(
        description="Basic CLI chat prompt for the configured Shakespeare models.",
        epilog=_build_help_epilog(models),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        required=True,
        metavar="N",
        help="1-based model number from the help list.",
    )
    parser.add_argument(
        "-a",
        "--adapter",
        type=int,
        metavar="N",
        help="1-based adapter number for the selected model. Defaults to that model's default adapter.",
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Start with the Shakespearean wording post-processing disabled. Toggle later with /style.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG context retrieval per turn (matches the web flag-less generate path).",
    )
    return parser


def _resolve_model_choice(
    parser: argparse.ArgumentParser,
    models: list[dict[str, Any]],
    model_number: int,
) -> dict[str, Any]:
    """Resolve a 1-based model number into model metadata."""
    if not 1 <= model_number <= len(models):
        parser.error(f"--model must be between 1 and {len(models)}.")
    return models[model_number - 1]


def _resolve_adapter_choice(
    parser: argparse.ArgumentParser,
    model_info: dict[str, Any],
    adapter_number: int | None,
) -> tuple[int, dict[str, str]]:
    """Resolve a 1-based adapter number for the chosen model."""
    if adapter_number is None:
        adapter_number = _default_adapter_index(model_info)

    adapters = model_info["adapters"]
    if not 1 <= adapter_number <= len(adapters):
        parser.error(
            f"--adapter must be between 1 and {len(adapters)} for model '{model_info['name']}'."
        )
    return adapter_number, adapters[adapter_number - 1]


def _print_runtime_help(state: dict[str, Any]) -> None:
    """Show the prompt-time commands and current toggle state."""
    print("Commands: /help, /reset, /style, /character <name>, /quit")
    print(
        f"  style: {'on' if state['apply_shakespeare_style'] else 'off'} | "
        f"rag: {'on' if state['rag_enabled'] else 'off'} | "
        f"character: {state['character']} ({state['work']})"
    )


def _handle_character_command(user_message: str, state: dict[str, Any]) -> bool:
    """Handle `/character <name>` and `character <name>`; return True if consumed."""
    for prefix in CHARACTER_COMMAND_PREFIXES:
        if user_message.lower().startswith(prefix):
            new_character = user_message[len(prefix):].strip()
            if not new_character:
                print(f"Current character: {state['character']} ({state['work']})")
                return True
            from pipeline import lm_generation

            try:
                lm_generation.set_character_context(new_character, state["work"])
            except ValueError as exc:
                print(f"error> {exc}")
                return True
            state["character"] = new_character
            lm_generation.refresh_chat_history()
            print(f"Character set to {new_character}. Chat history reset.")
            return True
    return False


def run_prompt(
    generation_pipeline,
    tokenizer,
    model,
    state: dict[str, Any],
) -> None:
    """Run the plain stdin/stdout chat loop."""
    _print_runtime_help(state)

    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        if not user_message:
            continue

        normalized_command = user_message.lower()
        if normalized_command in EXIT_COMMANDS:
            return
        if normalized_command in HELP_COMMANDS:
            _print_runtime_help(state)
            continue
        if normalized_command in RESET_COMMANDS:
            generation_pipeline.refresh_chat_history()
            print("Chat history reset.")
            continue
        if normalized_command in STYLE_COMMANDS:
            state["apply_shakespeare_style"] = not state["apply_shakespeare_style"]
            print(
                f"Shakespeare style {'enabled' if state['apply_shakespeare_style'] else 'disabled'}."
            )
            continue
        if _handle_character_command(user_message, state):
            continue

        context = None
        if state["rag_enabled"]:
            from pipeline.rag import get_context

            try:
                context = get_context(user_message)
            except Exception as exc:
                print(f"warn> RAG context unavailable: {exc}")

        try:
            response = generation_pipeline.generate_output(
                user_message,
                tokenizer,
                model,
                context=context,
                apply_shakespeare_style=state["apply_shakespeare_style"],
            )
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            return
        except Exception as exc:
            print(f"error> {exc}")
            continue

        print(f"{state['character'].lower()}> {response}")


def main() -> None:
    """Parse CLI flags, load the chosen model, then start the basic chat prompt."""
    from pipeline import lm_generation
    from pipeline.utils import ensure_loaded_model

    models = lm_generation.model_selection()
    parser = build_parser(models)
    args = parser.parse_args()

    if not models:
        parser.error("No models are available.")

    selected_model = _resolve_model_choice(parser, models, args.model)
    adapter_number, selected_adapter = _resolve_adapter_choice(parser, selected_model, args.adapter)

    character = str(
        selected_adapter.get("character")
        or selected_model.get("character")
        or "Hamlet"
    ).strip()
    work = str(
        selected_adapter.get("work")
        or selected_model.get("work")
        or "Hamlet"
    ).strip()
    lm_generation.set_character_context(character, work)

    print(f"Loading model #{args.model}: {selected_model['name']}")
    print(
        f"Using adapter #{adapter_number}: {selected_adapter['name']} ({selected_adapter['path']})"
    )
    print(f"Character: {character} ({work})")
    print("Model loading can take a while on first use.")

    model, tokenizer = ensure_loaded_model(
        selected_model["name"],
        selected_adapter["path"],
    )

    print("Model loaded. Start chatting.")
    state: dict[str, Any] = {
        "apply_shakespeare_style": not args.no_style,
        "rag_enabled": not args.no_rag,
        "character": character,
        "work": work,
    }
    run_prompt(lm_generation, tokenizer, model, state)


if __name__ == "__main__":
    main()
