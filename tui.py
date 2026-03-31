"""Lightweight curses TUI for chatting with configured Shakespeare models."""

from __future__ import annotations

import curses
import os
import queue
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_CHARACTER = "Hamlet"
DEFAULT_WORK = "Hamlet"
MAX_LOG_ENTRIES = 500
POLL_INTERVAL_SECONDS = 0.05


@dataclass
class ChatMessage:
    """A single chat bubble rendered inside the transcript view."""

    role: str
    content: str
    timestamp: str


@dataclass
class LogEntry:
    """A lightweight event record shown in the log view."""

    kind: str
    detail: str
    timestamp: str


@dataclass
class SelectorState:
    """Modal selector state for model and adapter pickers."""

    kind: str
    title: str
    options: list[dict[str, str]]
    index: int = 0


@dataclass
class InferenceRuntime:
    """Lazy wrapper around the generation pipeline so the TUI paints quickly."""

    pipeline: Any | None = None
    model: Any | None = None
    tokenizer: Any | None = None

    def _pipeline(self):
        if self.pipeline is None:
            from pipeline import lm_generation

            self.pipeline = lm_generation
        return self.pipeline

    def available_models(self) -> list[dict[str, Any]]:
        return self._pipeline().model_selection()

    def set_character(self, character: str, work: str) -> None:
        self._pipeline().set_character_context(character, work)

    def refresh_chat(self) -> None:
        self._pipeline().refresh_chat_history()

    def load_model(self, model_name: str, adapter_path: str) -> None:
        self.model, self.tokenizer = self._pipeline().get_model(model_name, adapter_path)

    def generate(self, question: str, apply_shakespeare_style: bool) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded yet.")

        return self._pipeline().generate_output(
            question,
            self.tokenizer,
            self.model,
            context=None,
            apply_shakespeare_style=apply_shakespeare_style,
        )


class ShakespeareTUI:
    """Minimal terminal UI for swapping models and chatting in-place."""

    def __init__(self) -> None:
        self.runtime = InferenceRuntime()
        self.task_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.result_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)

        self.models: list[dict[str, Any]] = []
        self.current_model_name = ""
        self.current_adapter_path = ""
        self.character = DEFAULT_CHARACTER
        self.work = DEFAULT_WORK
        self.shakespeare_style_enabled = True
        self.status = "Starting terminal interface..."
        self.error = ""
        self.messages: list[ChatMessage] = []
        self.logs: list[LogEntry] = []
        self.input_buffer = ""
        self.input_cursor = 0
        self.view = "chat"
        self.selector: SelectorState | None = None
        self.is_busy = False
        self.busy_label = ""
        self.chat_scroll = 0
        self.log_scroll = 0

    def run(self) -> None:
        self.worker.start()
        self._submit_task("initialize")
        curses.wrapper(self._main)

    def _main(self, stdscr) -> None:
        self._setup_curses(stdscr)

        while True:
            self._drain_result_queue()
            self._render(stdscr)
            key = stdscr.getch()
            if key != -1 and not self._handle_key(key):
                break
            time.sleep(POLL_INTERVAL_SECONDS)

        self.stop_event.set()

    def _setup_curses(self, stdscr) -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.timeout(50)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_YELLOW, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_GREEN, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                task_name, payload = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if task_name == "initialize":
                    self._worker_initialize()
                elif task_name == "load_model":
                    self._worker_load_model(payload["model_name"], payload["adapter_path"])
                elif task_name == "send_message":
                    self._worker_send_message(
                        payload["question"],
                        payload["apply_shakespeare_style"],
                    )
                elif task_name == "refresh_chat":
                    self.runtime.refresh_chat()
                    self.result_queue.put(("chat_reset", {}))
                    self.result_queue.put(
                        (
                            "status",
                            {
                                "text": "Chat history refreshed.",
                                "kind": "refresh",
                            },
                        )
                    )
            except Exception as exc:  # pragma: no cover - defensive UI error path
                detail = str(exc).strip() or exc.__class__.__name__
                self.result_queue.put(("error", {"text": detail}))
                self.result_queue.put(
                    (
                        "log",
                        {
                            "kind": "error",
                            "detail": detail,
                        },
                    )
                )
                self.result_queue.put(
                    (
                        "debug_trace",
                        {
                            "detail": traceback.format_exc(limit=8).strip(),
                        },
                    )
                )
            finally:
                self.result_queue.put(("task_finished", {}))
                self.task_queue.task_done()

    def _worker_initialize(self) -> None:
        self.result_queue.put(
            (
                "status",
                {
                    "text": "Importing inference pipeline...",
                    "kind": "startup",
                },
            )
        )

        self.runtime.set_character(self.character, self.work)
        self.result_queue.put(
            (
                "log",
                {
                    "kind": "character",
                    "detail": f"Character context set to {self.character}.",
                },
            )
        )

        models = self.runtime.available_models()
        self.result_queue.put(("models_loaded", {"models": models}))
        if not models:
            self.result_queue.put(
                (
                    "status",
                    {
                        "text": "No loadable models were found.",
                        "kind": "startup",
                    },
                )
            )
            return

        first_model = models[0]
        adapter_path = self._default_adapter_path(first_model)
        self._worker_load_model(first_model["name"], adapter_path)

    def _worker_load_model(self, model_name: str, adapter_path: str) -> None:
        self.result_queue.put(
            (
                "status",
                {
                    "text": f"Loading model {model_name}...",
                    "kind": "model",
                },
            )
        )
        self.runtime.load_model(model_name, adapter_path)
        self.result_queue.put(
            (
                "model_loaded",
                {
                    "model_name": model_name,
                    "adapter_path": adapter_path,
                },
            )
        )
        self.result_queue.put(
            (
                "status",
                {
                    "text": f"Ready: {model_name}",
                    "kind": "model",
                },
            )
        )

    def _worker_send_message(self, question: str, apply_shakespeare_style: bool) -> None:
        self.result_queue.put(
            (
                "status",
                {
                    "text": f"{self.character} is composing a reply...",
                    "kind": "generation",
                },
            )
        )
        response = self.runtime.generate(
            question,
            apply_shakespeare_style=apply_shakespeare_style,
        )
        self.result_queue.put(("assistant_message", {"content": response}))
        self.result_queue.put(
            (
                "status",
                {
                    "text": "A reply hath arrived.",
                    "kind": "reply",
                },
            )
        )

    def _submit_task(self, task_name: str, **payload: Any) -> bool:
        if self.is_busy:
            self._set_error("Wait for the current task to finish.")
            return False

        self.is_busy = True
        self.busy_label = task_name.replace("_", " ")
        self.task_queue.put((task_name, payload))
        return True

    def _drain_result_queue(self) -> None:
        while True:
            try:
                event_name, payload = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if event_name == "status":
                self.status = payload["text"]
                self._add_log(payload.get("kind", "status"), payload["text"])
            elif event_name == "error":
                self._set_error(payload["text"])
            elif event_name == "log":
                self._add_log(payload["kind"], payload["detail"])
            elif event_name == "debug_trace":
                self._add_log("trace", payload["detail"])
            elif event_name == "models_loaded":
                self.models = payload["models"]
                self._add_log("models", f"Discovered {len(self.models)} model option(s).")
            elif event_name == "model_loaded":
                self.current_model_name = payload["model_name"]
                self.current_adapter_path = payload["adapter_path"]
                self.error = ""
                adapter = self._current_adapter()
                adapter_name = adapter["name"] if adapter else self.current_adapter_path
                self._add_log(
                    "model",
                    f"Active model: {self.current_model_name} [{adapter_name}]",
                )
            elif event_name == "assistant_message":
                content = payload["content"]
                self.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=content,
                        timestamp=self._timestamp(),
                    )
                )
                self.chat_scroll = 0
                self._add_log("assistant", content)
                self.error = ""
            elif event_name == "chat_reset":
                self.messages.clear()
                self.chat_scroll = 0
                self.error = ""
            elif event_name == "task_finished":
                self.is_busy = False
                self.busy_label = ""

            self.result_queue.task_done()

    def _handle_key(self, key: int) -> bool:
        if self.selector is not None:
            return self._handle_selector_key(key)

        if key in (ord("q"), 27):
            return False

        if key == 9:
            self.view = "logs" if self.view == "chat" else "chat"
            if self.view == "logs":
                self.log_scroll = 10**9
            return True

        if key == ord("c"):
            self.view = "chat"
            return True

        if key == ord("l"):
            self.view = "logs"
            self.log_scroll = 10**9
            return True

        if key == ord("m"):
            self._open_model_selector()
            return True

        if key == ord("a"):
            self._open_adapter_selector()
            return True

        if key == ord("r"):
            if self._submit_task("refresh_chat"):
                self._add_log("refresh", "Chat reset requested.")
            return True

        if key == ord("s"):
            self.shakespeare_style_enabled = not self.shakespeare_style_enabled
            style_text = "enabled" if self.shakespeare_style_enabled else "disabled"
            self.status = f"Shakespeare style {style_text}."
            self._add_log("style", self.status)
            return True

        if self.view == "logs":
            self._handle_log_navigation(key)
            return True

        return self._handle_chat_input(key)

    def _handle_selector_key(self, key: int) -> bool:
        if self.selector is None:
            return True

        options = self.selector.options
        if key in (27, ord("q")):
            self.selector = None
            return True

        if key in (curses.KEY_UP, ord("k")) and options:
            self.selector.index = max(0, self.selector.index - 1)
            return True

        if key in (curses.KEY_DOWN, ord("j")) and options:
            self.selector.index = min(len(options) - 1, self.selector.index + 1)
            return True

        if key == curses.KEY_PPAGE and options:
            self.selector.index = max(0, self.selector.index - 8)
            return True

        if key == curses.KEY_NPAGE and options:
            self.selector.index = min(len(options) - 1, self.selector.index + 8)
            return True

        if key in (10, 13):
            selected = options[self.selector.index] if options else None
            kind = self.selector.kind
            self.selector = None
            if selected is None:
                return True

            if kind == "model":
                model = self._find_model(selected["name"])
                if model is None:
                    self._set_error("Selected model is no longer available.")
                    return True

                adapter_path = self._default_adapter_path(model)
                self._submit_task(
                    "load_model",
                    model_name=model["name"],
                    adapter_path=adapter_path,
                )
                return True

            if kind == "adapter":
                self._submit_task(
                    "load_model",
                    model_name=self.current_model_name,
                    adapter_path=selected["path"],
                )
                return True

        return True

    def _handle_log_navigation(self, key: int) -> None:
        if key in (curses.KEY_UP, ord("k")):
            self.log_scroll = max(0, self.log_scroll - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            self.log_scroll += 1
        elif key == curses.KEY_PPAGE:
            self.log_scroll = max(0, self.log_scroll - 10)
        elif key == curses.KEY_NPAGE:
            self.log_scroll += 10
        elif key == curses.KEY_HOME:
            self.log_scroll = 0
        elif key == curses.KEY_END:
            self.log_scroll = 10**9

    def _handle_chat_input(self, key: int) -> bool:
        if key in (10, 13):
            question = self.input_buffer.strip()
            if not question:
                return True
            if not self.current_model_name:
                self._set_error("Load a model before chatting.")
                return True
            if not self._submit_task(
                "send_message",
                question=question,
                apply_shakespeare_style=self.shakespeare_style_enabled,
            ):
                return True

            self.messages.append(
                ChatMessage(
                    role="user",
                    content=question,
                    timestamp=self._timestamp(),
                )
            )
            self._add_log("user", question)
            self.input_buffer = ""
            self.input_cursor = 0
            self.chat_scroll = 0
            self.error = ""
            return True

        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self.input_cursor > 0:
                self.input_buffer = (
                    self.input_buffer[: self.input_cursor - 1]
                    + self.input_buffer[self.input_cursor :]
                )
                self.input_cursor -= 1
            return True

        if key == curses.KEY_DC:
            if self.input_cursor < len(self.input_buffer):
                self.input_buffer = (
                    self.input_buffer[: self.input_cursor]
                    + self.input_buffer[self.input_cursor + 1 :]
                )
            return True

        if key == curses.KEY_LEFT:
            self.input_cursor = max(0, self.input_cursor - 1)
            return True

        if key == curses.KEY_RIGHT:
            self.input_cursor = min(len(self.input_buffer), self.input_cursor + 1)
            return True

        if key == curses.KEY_HOME:
            self.input_cursor = 0
            return True

        if key == curses.KEY_END:
            self.input_cursor = len(self.input_buffer)
            return True

        if key == curses.KEY_PPAGE:
            self.chat_scroll += 8
            return True

        if key == curses.KEY_NPAGE:
            self.chat_scroll = max(0, self.chat_scroll - 8)
            return True

        if 32 <= key <= 126:
            self.input_buffer = (
                self.input_buffer[: self.input_cursor]
                + chr(key)
                + self.input_buffer[self.input_cursor :]
            )
            self.input_cursor += 1
            return True

        return True

    def _render(self, stdscr) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 14 or width < 60:
            self._safe_addstr(
                stdscr,
                0,
                0,
                "Terminal too small. Resize to at least 60x14.",
            )
            stdscr.refresh()
            return

        title_attr = curses.A_BOLD | self._color_attr(1)
        self._safe_addstr(stdscr, 0, 2, "Shakespeare Model TUI", title_attr)
        self._safe_addstr(
            stdscr,
            1,
            2,
            self._truncate(
                f"View: {self.view} | Status: {self._status_line()}",
                width - 4,
            ),
        )
        self._safe_addstr(
            stdscr,
            2,
            2,
            self._truncate(f"Model: {self.current_model_name or 'not loaded'}", width - 4),
        )

        adapter = self._current_adapter()
        adapter_label = adapter["name"] if adapter else (self.current_adapter_path or "n/a")
        settings_line = (
            f"Adapter: {adapter_label} | Character: {self.character} | "
            f"Style: {'on' if self.shakespeare_style_enabled else 'off'}"
        )
        self._safe_addstr(stdscr, 3, 2, self._truncate(settings_line, width - 4))

        help_line = "Keys: Enter send | m model | a adapter | r reset | s style | Tab logs/chat | q quit"
        self._safe_addstr(stdscr, 4, 2, self._truncate(help_line, width - 4))
        stdscr.hline(5, 0, curses.ACS_HLINE, width)

        if self.error:
            self._safe_addstr(
                stdscr,
                6,
                2,
                self._truncate(f"Error: {self.error}", width - 4),
                self._color_attr(2),
            )
            body_top = 7
        else:
            body_top = 6

        input_height = 3 if self.view == "chat" else 0
        body_bottom = height - input_height - 1

        if self.view == "chat":
            self._render_chat(stdscr, body_top, body_bottom, width)
            self._render_input(stdscr, height - 3, width)
        else:
            self._render_logs(stdscr, body_top, body_bottom, width)

        if self.selector is not None:
            self._render_selector(stdscr, height, width)

        stdscr.refresh()

    def _render_chat(self, stdscr, top: int, bottom: int, width: int) -> None:
        transcript_height = max(1, bottom - top)
        line_width = max(20, width - 6)
        lines = self._chat_lines(line_width)

        max_scroll = max(0, len(lines) - transcript_height)
        self.chat_scroll = max(0, min(self.chat_scroll, max_scroll))
        start = max(0, len(lines) - transcript_height - self.chat_scroll)
        visible_lines = lines[start : start + transcript_height]

        for row, line in enumerate(visible_lines, start=top):
            attr = 0
            if line.startswith("[") and "You:" in line:
                attr = self._color_attr(3)
            elif line.startswith("[") and f"{self.character}:" in line:
                attr = self._color_attr(4)
            self._safe_addstr(stdscr, row, 2, self._truncate(line, width - 4), attr)

        if not self.messages and not self.is_busy:
            self._safe_addstr(
                stdscr,
                top,
                2,
                f"Speak to {self.character} to begin the conversation.",
                self._color_attr(5),
            )

    def _render_input(self, stdscr, top: int, width: int) -> None:
        stdscr.hline(top - 1, 0, curses.ACS_HLINE, width)
        prompt = "Prompt> "
        max_text_width = max(10, width - len(prompt) - 4)
        start_index = 0
        if self.input_cursor > max_text_width:
            start_index = self.input_cursor - max_text_width
        visible_text = self.input_buffer[start_index : start_index + max_text_width]

        self._safe_addstr(stdscr, top, 2, prompt, curses.A_BOLD)
        self._safe_addstr(stdscr, top, 2 + len(prompt), visible_text)

        footer = "PageUp/PageDown scroll chat history."
        self._safe_addstr(stdscr, top + 1, 2, self._truncate(footer, width - 4))

        cursor_x = 2 + len(prompt) + (self.input_cursor - start_index)
        cursor_x = min(width - 2, max(2 + len(prompt), cursor_x))
        stdscr.move(top, cursor_x)

    def _render_logs(self, stdscr, top: int, bottom: int, width: int) -> None:
        visible_height = max(1, bottom - top)
        lines = self._log_lines(max(20, width - 6))
        max_scroll = max(0, len(lines) - visible_height)
        self.log_scroll = max(0, min(self.log_scroll, max_scroll))
        visible_lines = lines[self.log_scroll : self.log_scroll + visible_height]

        for row, line in enumerate(visible_lines, start=top):
            attr = 0
            if "[error]" in line or "[trace]" in line:
                attr = self._color_attr(2)
            elif "[user]" in line:
                attr = self._color_attr(3)
            elif "[assistant]" in line:
                attr = self._color_attr(4)
            self._safe_addstr(stdscr, row, 2, self._truncate(line, width - 4), attr)

        if not lines:
            self._safe_addstr(stdscr, top, 2, "No activity yet.")

    def _render_selector(self, stdscr, height: int, width: int) -> None:
        if self.selector is None:
            return

        box_height = min(height - 4, 16)
        box_width = min(width - 4, 90)
        top = max(1, (height - box_height) // 2)
        left = max(2, (width - box_width) // 2)
        list_height = max(4, box_height - 8)
        box = stdscr.derwin(box_height, box_width, top, left)
        box.erase()
        box.box()
        self._safe_addstr(box, 0, 2, f" {self.selector.title} ", curses.A_BOLD)

        options = self.selector.options
        if not options:
            self._safe_addstr(box, 2, 2, "No options available.")
            return

        start = max(0, self.selector.index - list_height // 2)
        end = min(len(options), start + list_height)
        start = max(0, end - list_height)

        for row, option in enumerate(options[start:end], start=2):
            option_index = start + (row - 2)
            attr = curses.A_REVERSE if option_index == self.selector.index else 0
            label = option.get("name", option.get("path", ""))
            self._safe_addstr(
                box,
                row,
                2,
                self._truncate(label, box_width - 4),
                attr,
            )

        current = options[self.selector.index]
        description = current.get("description") or current.get("path", "")
        box.hline(box_height - 5, 1, curses.ACS_HLINE, box_width - 2)
        wrapped = self._wrap_text(description, box_width - 4)
        for offset, line in enumerate(wrapped[:2]):
            self._safe_addstr(box, box_height - 4 + offset, 2, line)
        self._safe_addstr(
            box,
            box_height - 2,
            2,
            self._truncate("Enter load | Esc cancel | Up/Down move", box_width - 4),
        )

    def _open_model_selector(self) -> None:
        if not self.models:
            self._set_error("No models are available.")
            return

        current_index = 0
        for index, model in enumerate(self.models):
            if model["name"] == self.current_model_name:
                current_index = index
                break

        self.selector = SelectorState(
            kind="model",
            title="Select Model",
            options=self.models,
            index=current_index,
        )

    def _open_adapter_selector(self) -> None:
        model = self._current_model()
        if model is None:
            self._set_error("Load a model before choosing an adapter.")
            return

        adapters = list(model.get("adapters", []))
        if not adapters:
            self._set_error("This model has no adapters.")
            return

        current_index = 0
        for index, adapter in enumerate(adapters):
            if adapter["path"] == self.current_adapter_path:
                current_index = index
                break

        self.selector = SelectorState(
            kind="adapter",
            title=f"Select Adapter for {model['name']}",
            options=adapters,
            index=current_index,
        )

    def _current_model(self) -> dict[str, Any] | None:
        return self._find_model(self.current_model_name)

    def _current_adapter(self) -> dict[str, Any] | None:
        model = self._current_model()
        if model is None:
            return None

        return next(
            (
                adapter
                for adapter in model.get("adapters", [])
                if adapter.get("path") == self.current_adapter_path
            ),
            None,
        )

    def _find_model(self, model_name: str) -> dict[str, Any] | None:
        return next((model for model in self.models if model["name"] == model_name), None)

    def _default_adapter_path(self, model: dict[str, Any]) -> str:
        adapters = list(model.get("adapters", []))
        if not adapters:
            return ""

        preferred_path = str(model.get("default_adapter_path", "")).strip()
        if preferred_path and any(adapter.get("path") == preferred_path for adapter in adapters):
            return preferred_path
        return str(adapters[0].get("path", ""))

    def _chat_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        for message in self.messages:
            speaker = "You" if message.role == "user" else self.character
            prefix = f"[{message.timestamp}] {speaker}: "
            lines.extend(self._wrap_prefixed(message.content, prefix, width))

        if self.is_busy and self.busy_label == "send message":
            lines.append(f"[{self._timestamp()}] {self.character}: ...")

        return lines

    def _log_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        for entry in self.logs:
            prefix = f"[{entry.timestamp}] [{entry.kind}] "
            lines.extend(self._wrap_prefixed(entry.detail, prefix, width))
        return lines

    def _wrap_prefixed(self, text: str, prefix: str, width: int) -> list[str]:
        text = text.strip() or "-"
        available_width = max(10, width - len(prefix))
        wrapped_lines = []
        for raw_line in text.splitlines() or [""]:
            next_lines = textwrap.wrap(
                raw_line,
                width=available_width,
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
            wrapped_lines.extend(next_lines)

        if not wrapped_lines:
            return [prefix.rstrip()]

        lines = [f"{prefix}{wrapped_lines[0]}"]
        indent = " " * len(prefix)
        lines.extend(f"{indent}{line}" for line in wrapped_lines[1:])
        return lines

    def _wrap_text(self, text: str, width: int) -> list[str]:
        text = text.strip() or "-"
        wrapped_lines = []
        for raw_line in text.splitlines() or [""]:
            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                or [""]
            )
        return wrapped_lines

    def _add_log(self, kind: str, detail: str) -> None:
        self.logs.append(
            LogEntry(
                kind=kind,
                detail=detail,
                timestamp=self._timestamp(),
            )
        )
        if len(self.logs) > MAX_LOG_ENTRIES:
            self.logs = self.logs[-MAX_LOG_ENTRIES:]
        if self.view == "logs":
            self.log_scroll = 10**9

    def _set_error(self, message: str) -> None:
        self.error = message

    def _status_line(self) -> str:
        if self.is_busy and self.busy_label:
            spinner = "-\\|/"[int(time.time() * 4) % 4]
            return f"{self.status} [{spinner}]"
        return self.status

    def _timestamp(self) -> str:
        return time.strftime("%H:%M:%S")

    def _truncate(self, text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _safe_addstr(self, stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
        height, width = stdscr.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        try:
            stdscr.addstr(y, x, text[: max(0, width - x - 1)], attr)
        except curses.error:
            pass

    def _color_attr(self, pair_number: int) -> int:
        return curses.color_pair(pair_number) if curses.has_colors() else 0


def main() -> None:
    """Launch the terminal UI."""
    ShakespeareTUI().run()


if __name__ == "__main__":
    main()
