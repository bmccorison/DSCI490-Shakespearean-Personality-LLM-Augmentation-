"""Basic numbered CLI for loading a model + adapter and chatting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from models.models import model_list

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL_ADAPTER_PATH = "__base__"
DEFAULT_CHARACTER = "Hamlet"
DEFAULT_WORK = "Hamlet"
REPO_ROOT = Path(__file__).resolve().parent
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
HELP_COMMANDS = {"help", "/help"}
RESET_COMMANDS = {"reset", "/reset"}


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Parser help formatter that preserves dynamic numbering blocks."""


def _configured_adapters(configured_model: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize adapters from the current config shape and the legacy fallback."""
    adapters = configured_model.get("adapters")
    if isinstance(adapters, list):
        return [
            adapter
            for adapter in adapters
            if isinstance(adapter, dict)
        ]

    legacy_adapters = configured_model.get("adapter_paths")
    if not isinstance(legacy_adapters, list):
        return []

    normalized_adapters: list[dict[str, str]] = []
    for adapter in legacy_adapters:
        if not isinstance(adapter, dict):
            continue

        description = str(adapter.get("description", ""))
        adapter_fields = [
            (key, value)
            for key, value in adapter.items()
            if key != "description" and isinstance(value, str)
        ]
        if not adapter_fields:
            continue

        adapter_name, adapter_path = adapter_fields[0]
        normalized_adapters.append(
            {
                "name": adapter_name,
                "path": adapter_path,
                "description": description,
            }
        )

    return normalized_adapters


def _resolve_adapter_path(adapter_path: str) -> Path:
    """Resolve adapter paths relative to the repository root when needed."""
    resolved_path = Path(adapter_path)
    if not resolved_path.is_absolute():
        resolved_path = REPO_ROOT / resolved_path
    return resolved_path


def _is_base_adapter(adapter_path: str) -> bool:
    """Return whether the adapter token means 'use the base model only'."""
    return adapter_path.strip() == BASE_MODEL_ADAPTER_PATH


def available_models() -> list[dict[str, Any]]:
    """Return loadable models and adapters using the repo's configured metadata."""
    models: list[dict[str, Any]] = []

    for configured_model in model_list():
        adapters: list[dict[str, str]] = []
        for adapter in _configured_adapters(configured_model):
            adapter_path = str(adapter.get("path", "")).strip()
            if not adapter_path:
                continue

            if _is_base_adapter(adapter_path) or _resolve_adapter_path(adapter_path).exists():
                adapters.append(
                    {
                        "name": str(adapter.get("name", adapter_path)),
                        "path": adapter_path,
                        "description": str(adapter.get("description", "")),
                    }
                )

        if not adapters:
            adapters.append(
                {
                    "name": "base_model",
                    "path": BASE_MODEL_ADAPTER_PATH,
                    "description": "Base model without a LoRA adapter.",
                }
            )

        default_adapter_path = str(configured_model.get("default_adapter_path", "")).strip()
        if default_adapter_path not in {adapter["path"] for adapter in adapters}:
            default_adapter_path = adapters[0]["path"]

        models.append(
            {
                "name": str(configured_model["name"]),
                "description": str(configured_model.get("description", "")),
                "default_adapter_path": default_adapter_path,
                "adapters": adapters,
            }
        )

    return models


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
        if model_info["description"]:
            lines.append(f"     {model_info['description']}")

        default_adapter_index = _default_adapter_index(model_info)
        lines.append("     adapters:")
        for adapter_index, adapter in enumerate(model_info["adapters"], start=1):
            suffix = " [default]" if adapter_index == default_adapter_index else ""
            lines.append(
                f"       {adapter_index}. {adapter['name']} -> {adapter['path']}{suffix}"
            )
            if adapter["description"]:
                lines.append(f"          {adapter['description']}")

    return "\n".join(lines)


def build_parser(models: list[dict[str, Any]]) -> argparse.ArgumentParser:
    """Build the CLI parser with a dynamic numbered help block."""
    parser = argparse.ArgumentParser(
        description="Very small CLI chat prompt for the configured Shakespeare models.",
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
        help="Disable the lightweight Shakespearean wording post-processing.",
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


def _print_runtime_help() -> None:
    """Show the tiny set of prompt-time commands."""
    print("Commands: /help, /reset, /quit")


def run_prompt(
    generation_pipeline,
    tokenizer,
    model,
    *,
    apply_shakespeare_style: bool,
) -> None:
    """Run the plain stdin/stdout chat loop."""
    _print_runtime_help()

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
            _print_runtime_help()
            continue
        if normalized_command in RESET_COMMANDS:
            generation_pipeline.refresh_chat_history()
            print("Chat history reset.")
            continue

        try:
            response = generation_pipeline.generate_output(
                user_message,
                tokenizer,
                model,
                context=None,
                apply_shakespeare_style=apply_shakespeare_style,
            )
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            return
        except Exception as exc:
            print(f"error> {exc}")
            continue

        print(f"{DEFAULT_CHARACTER.lower()}> {response}")


def main() -> None:
    """Parse CLI flags, load the chosen model, then start the basic chat prompt."""
    models = available_models()
    parser = build_parser(models)
    args = parser.parse_args()

    if not models:
        parser.error("No models are available.")

    selected_model = _resolve_model_choice(parser, models, args.model)
    adapter_number, selected_adapter = _resolve_adapter_choice(parser, selected_model, args.adapter)

    from pipeline import lm_generation

    lm_generation.set_character_context(DEFAULT_CHARACTER, DEFAULT_WORK)

    print(
        f"Loading model #{args.model}: {selected_model['name']}"
    )
    print(
        f"Using adapter #{adapter_number}: {selected_adapter['name']} ({selected_adapter['path']})"
    )
    print("Model loading can take a while on first use.")

    model, tokenizer = lm_generation.get_model(
        selected_model["name"],
        selected_adapter["path"],
    )

    print("Model loaded. Start chatting.")
    run_prompt(
        lm_generation,
        tokenizer,
        model,
        apply_shakespeare_style=not args.no_style,
    )


if __name__ == "__main__":
    main()
