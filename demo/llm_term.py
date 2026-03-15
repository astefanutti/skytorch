"""
SkyTorch TUI Chat Demo — Rich terminal UI with GPU metrics.

Requires: pip install textual asciichartpy
"""

import asyncio
import collections
import logging
import os
import platform
import subprocess
import threading
import time

# Enable async scalar copy before any skytorch import (read at import time)
os.environ.setdefault("SKYTORCH_ASYNC_COPY", "1")
# Suppress gRPC fork warning when subprocess (e.g., pbcopy) forks while gRPC threads are active
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import asciichartpy
import torch
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Input, Rule, Select, Static
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer

from skytorch.client import Compute
from skytorch.transformers.cache import SpeculativeCache
from skytorch.transformers.harmony import HarmonyStreamer
from skytorch.transformers.streamer import AsyncStreamer

# Suppress all console logging — the TUI installs its own handler on mount.
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(logging.NullHandler())
# Suppress HF Hub warnings (e.g., unauthenticated requests) that bypass logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress urllib3 InsecureRequestWarning (printed via warnings.warn to stderr)
import urllib3  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fix Textual bug: _query_in_band_window_resize sends DECRPM ($p) without
# the Apple_Terminal guard, causing a literal "p" to appear on startup.
from textual.drivers.linux_driver import LinuxDriver  # noqa: E402

LinuxDriver._query_in_band_window_resize = lambda self: None

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "openai/gpt-oss-120b": {"streamer": "harmony", "triton_modules": ["model.layers.*.mlp"]},
    "Qwen/Qwen3-4B-Instruct-2507": {"streamer": "text"},
    "Qwen/Qwen2.5-0.5B-Instruct": {"streamer": "text"},
}

# ---------------------------------------------------------------------------
# Custom Textual messages
# ---------------------------------------------------------------------------


class GenerationDone(Message):
    def __init__(self, gen_tokens: int, elapsed: float, response: str) -> None:
        super().__init__()
        self.gen_tokens = gen_tokens
        self.elapsed = elapsed
        self.response = response


class ModelReady(Message):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


class StatusUpdate(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class LogLine(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


# ---------------------------------------------------------------------------
# Custom streamers that buffer token deltas for batched rendering
# ---------------------------------------------------------------------------


class TUIHarmonyStreamer(HarmonyStreamer):
    """HarmonyStreamer that buffers token deltas for the Textual app."""

    def __init__(self, tokenizer, app: "LLMTermApp", **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._app = app

    def _post(self, channel: str, delta: str) -> None:
        self._app._token_buffer.append((channel, delta))

    def put(self, value):
        from openai_harmony import StreamState

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("HarmonyStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if self._response_complete:
            self.generated_tokens += value.numel()
            return

        for token_id in value.tolist():
            self.generated_tokens += 1
            if token_id == self._endoftext_token_id:
                return
            if self._parser.state == StreamState.EXPECT_START:
                if token_id != self._start_token_id:
                    continue
            self._parser.process(token_id)
            state = self._parser.state
            channel = self._parser.current_channel

            if state == StreamState.EXPECT_START and self._prev_channel == "final":
                self._response_complete = True
                self._valid_generated = self.generated_tokens

            # Transition into content — post a label delta
            if state == StreamState.CONTENT and self._prev_state != StreamState.CONTENT:
                if channel == "analysis":
                    self._post("thinking", "Thinking: ")
                elif channel == "final":
                    self._post("assistant", "")

            # Stream content deltas
            if state == StreamState.CONTENT:
                delta = self._parser.last_content_delta
                if delta:
                    ch = "thinking" if channel == "analysis" else "assistant"
                    self._post(ch, delta)
                    if channel == "final":
                        self._final_text.append(delta)

            self._prev_state = state
            self._prev_channel = channel

    def end(self):
        if not self._response_complete:
            try:
                self._parser.process_eos()
            except Exception:
                pass
        self.next_tokens_are_prompt = True


class TUITextStreamer(TextStreamer):
    """TextStreamer that buffers token deltas for the Textual app."""

    def __init__(self, tokenizer, app: "LLMTermApp", **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._app = app
        self.generated_tokens = 0
        self._final_text: list[str] = []
        self._valid_generated = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._final_text.append(text)
        self._app._token_buffer.append(("assistant", text))

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TUITextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            self.generated_tokens += value.shape[0]
        super().put(value)

    def reset(self):
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.generated_tokens = 0
        self._final_text = []
        self._valid_generated = 0

    def get_final_response(self):
        return "".join(self._final_text).strip()


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

CSS = """
.section {
    border: round $accent 40%;
    border-title-color: $text 50%;
    padding: 0 1;
}
.section:focus-within {
    border: round $accent;
    border-title-color: $text;
    border-title-style: bold;
}
#sidebar {
    width: 40;
}
#model-section {
    height: auto;
}
#metrics-section {
    height: 1fr;
}
.chart {
    height: 1fr;
}
#main {
    width: 1fr;
}
#log-pane {
    height: 8;
}
#chat-section {
    height: 1fr;
}
#chat-log {
    height: 1fr;
}
#prompt-section {
    height: auto;
}
#prompt-section Input {
    border: none;
}
#prompt-section Input:focus {
    border: none;
}
.user-msg {
    background: $surface;
    margin: 1 0;
    padding: 0 1;
}
.thinking-msg {
    color: $text-muted;
    text-style: italic;
    margin: 0;
    padding: 0 1;
}
.assistant-msg {
    margin: 0 0 1 0;
    padding: 0 1;
}
"""


class LLMTermApp(App):
    TITLE = "LLM Term"
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "cancel_generation", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "gruvbox"
        # Compute / model state
        self._compute: Compute | None = None
        self._model: torch.nn.Module | None = None
        self._tokenizer = None
        self._streamer: AsyncStreamer | None = None
        self._inner_streamer = None
        self._model_name: str | None = None
        self._model_cfg: dict | None = None

        # Chat state
        self._history: list[dict[str, str]] = []
        self._past_key_values = None
        self._past_length = 0
        self._total_tokens = 0
        self._total_time = 0.0
        self._generating = False

        # Current streaming widgets
        self._thinking_widget: Static | None = None
        self._assistant_widget: Static | None = None
        self._thinking_text = ""
        self._assistant_text = ""

        # Non-blocking buffers (appended from background threads, drained by timer)
        self._token_buffer: collections.deque[tuple[str, str]] = collections.deque()
        self._metrics_buffer: collections.deque[object] = collections.deque()
        self._event_buffer: collections.deque[str] = collections.deque()
        self._log_buffer: collections.deque[str] = collections.deque()

        # Metrics history (60 data points = 1 minute)
        self._gpu_util: collections.deque[float] = collections.deque(maxlen=60)
        self._mem_used: collections.deque[float] = collections.deque(maxlen=60)
        self._power: collections.deque[float] = collections.deque(maxlen=60)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                with Vertical(id="model-section", classes="section") as s:
                    s.border_title = "Model"
                    yield Select(
                        [(name, name) for name in MODELS],
                        prompt="Select a model",
                        id="model-select",
                    )
                    yield Static("Waiting for compute...", id="status")
                    yield Static("", id="stats")
                with Vertical(id="metrics-section", classes="section") as s:
                    s.border_title = "Compute"
                    yield Static("", id="gpu-util", classes="chart")
                    yield Rule()
                    yield Static("", id="mem-used", classes="chart")
                    yield Rule()
                    yield Static("", id="power", classes="chart")
            with Vertical(id="main"):
                with VerticalScroll(id="log-pane", classes="section") as s:
                    s.border_title = "Logs"
                with Vertical(id="chat-section", classes="section") as s:
                    s.border_title = "Chat"
                    yield VerticalScroll(id="chat-log")
                with Vertical(id="prompt-section", classes="section") as s:
                    s.border_title = "Prompt"
                    yield Input(placeholder="Type a message...", id="prompt", disabled=True)
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._install_log_handler()
        self.set_interval(1 / 10, self._flush_buffers)
        self._start_compute()

    def _install_log_handler(self) -> None:
        """Route all log records to the TUI log pane instead of stdout/stderr."""
        app = self
        app_thread = threading.current_thread()

        class _TUIHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                text = self.format(record)
                if threading.current_thread() is app_thread:
                    app.post_message(LogLine(text))
                else:
                    app._log_buffer.append(text)

        handler = _TUIHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        # Install on the root logger to capture everything (skytorch, kubernetes, grpc, etc.)
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    @work(thread=False)
    async def _start_compute(self) -> None:
        """Create the Compute resource and wait for readiness."""
        self._set_status("Creating compute...")
        try:
            self._compute = Compute(
                name="term",
                image="ghcr.io/astefanutti/skytorch-server",
                resources={"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
                volumes=[{"name": "cache", "storage": "80Gi", "path": "/cache"}],
                env={
                    "HF_HOME": "/cache",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "TRITON_HOME": "/cache",
                },
                on_events=self._on_event,
                on_metrics=self._on_metrics,
            )
            self._set_status("Waiting for compute...")
            async with self._compute as c:
                c._keep_alive = True
                self._set_status("Select a model")
                try:
                    while not self.app._exit:
                        await asyncio.sleep(1)
                finally:
                    if c._event_watch_future is not None:
                        c._event_watch_future.cancel()
                        c._event_watch_future = None
                    if c._metrics_stream_future is not None:
                        c._metrics_stream_future.cancel()
                        c._metrics_stream_future = None
                    # Skip _cleanup_grpc_client — it blocks on drain_tensors
                    if c._grpc_client is not None:
                        await c._grpc_client.__aexit__(None, None, None)
                        c._grpc_client = None
                    c._keep_alive = False
        except (Exception, asyncio.CancelledError) as e:
            if not isinstance(e, asyncio.CancelledError):
                self.post_message(LogLine(f"Error: {e}"))

    def _on_event(self, event) -> None:
        """Event callback — runs on sky-async-loop, must not block."""
        reason = event.reason or ""
        message = event.message or ""
        self._event_buffer.append(f"{reason}: {message}")

    def _on_metrics(self, snapshot: object) -> None:
        """Metrics callback — runs on sky-async-loop, must not block."""
        self._metrics_buffer.append(snapshot)

    def _process_metrics(self, snapshot: object) -> None:
        """Accumulate metric values from a snapshot into chart deques."""
        for metric in snapshot.metrics:
            if metric.name == "gpu.utilization.compute":
                self._gpu_util.append(metric.value)
            elif metric.name == "gpu.memory.used":
                self._mem_used.append(metric.value / (1024**3))  # bytes -> GB
            elif metric.name == "gpu.power.usage":
                self._power.append(metric.value)

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard using platform-native tools."""
        self._clipboard = text
        try:
            if platform.system() == "Darwin":
                subprocess.run(["pbcopy"], input=text.encode(), check=True)
            else:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"], input=text.encode(), check=True
                )
        except FileNotFoundError:
            super().copy_to_clipboard(text)

    def on_log_line(self, msg: LogLine) -> None:
        log_pane = self.query_one("#log-pane", VerticalScroll)
        log_pane.mount(Static(msg.text))
        log_pane.scroll_end(animate=False)

    def _append_log(self, text: str) -> None:
        """Thread-safe log append — non-blocking."""
        self._log_buffer.append(text)

    def _update_charts(self) -> None:
        for chart_id, title, data, fixed_max in [
            ("gpu-util", "GPU %", self._gpu_util, 100),
            ("mem-used", "Mem GB", self._mem_used, None),
            ("power", "Power W", self._power, None),
        ]:
            widget = self.query_one(f"#{chart_id}", Static)
            if data:
                # Use available width minus label padding, height from widget
                w = max(widget.size.width - 10, 10)
                h = max(widget.size.height - 2, 3)
                series = list(data)[-w:]
                cfg = {"height": h, "min": 0, "format": "{:6.1f}"}
                if fixed_max is not None:
                    cfg["max"] = fixed_max
                else:
                    # Ensure a minimum range so the chart doesn't collapse
                    cfg["max"] = max(max(series) * 1.1, 1)
                text = asciichartpy.plot(series, cfg)
                widget.update(f"{title}\n{text}")
            else:
                widget.update(f"{title}\n(no data)")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        if self._compute is None or self._compute._grpc_client is None:
            self._set_status("Compute not ready yet")
            self.query_one("#model-select", Select).clear()
            return
        self._load_model(str(event.value))

    @work(thread=True, exclusive=True, group="model")
    def _load_model(self, model_name: str) -> None:
        """Load a model on the remote GPU and set up the local skeleton."""
        if self._compute is None:
            return

        self._set_status("Loading model...")
        prompt_input = self.query_one("#prompt", Input)
        self.call_from_thread(setattr, prompt_input, "disabled", True)

        try:
            from skytorch.torch.client.loop import get_event_loop

            loop = get_event_loop()
            model_cfg = MODELS[model_name]

            def load_fn(model):
                return AutoModelForCausalLM.from_pretrained(model, device_map="cuda")

            # Load model weights server-side
            future = asyncio.run_coroutine_threadsafe(
                self._compute.execute(
                    load_fn,
                    model_name,
                    retain_model=True,
                    on_log=lambda _stream, text: self._append_log(text.rstrip()),
                ),
                loop,
            )
            state_dict = future.result()

            # Load tokenizer locally
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Register device mapping (used later in _run_generate)
            self._compute.device("cuda")
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(model_name),
                    attn_implementation="eager",
                )
            triton_modules = model_cfg.get("triton_modules")
            state_dict.load_into(model, triton_modules=triton_modules)
            model.eval()

            # Create the appropriate streamer
            if model_cfg["streamer"] == "harmony":
                inner = TUIHarmonyStreamer(
                    tokenizer, self, skip_prompt=True, skip_special_tokens=True
                )
            else:
                inner = TUITextStreamer(tokenizer, self, skip_prompt=True, skip_special_tokens=True)
            streamer = AsyncStreamer(inner)

            # Store state
            self._model = model
            self._tokenizer = tokenizer
            self._streamer = streamer
            self._inner_streamer = inner
            self._model_name = model_name
            self._model_cfg = model_cfg

            # Reset chat state
            self._history = [{"role": "system", "content": "You are a helpful assistant."}]
            self._past_key_values = None
            self._past_length = 0
            self._total_tokens = 0
            self._total_time = 0.0

            # Clear chat log
            chat_log = self.query_one("#chat-log", VerticalScroll)
            self.call_from_thread(chat_log.remove_children)

            self._set_status("Ready")
            self.call_from_thread(setattr, prompt_input, "disabled", False)
            self.call_from_thread(prompt_input.focus)
            self.post_message(ModelReady(model_name))

        except Exception as e:
            self._append_log(f"Load failed: {e}")
            self.call_from_thread(setattr, prompt_input, "disabled", False)

    # ------------------------------------------------------------------
    # Chat / generation
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._generating or self._model is None:
            return

        event.input.value = ""
        self._generating = True
        event.input.disabled = True

        # Add user message
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(f"[bold]You:[/bold] {text}", classes="user-msg"))
        chat_log.scroll_end(animate=False)

        # Reset streaming widget trackers
        self._thinking_widget = None
        self._assistant_widget = None
        self._thinking_text = ""
        self._assistant_text = ""

        self._run_generate(text)

    @work(thread=True, exclusive=True, group="generate")
    def _run_generate(self, user_text: str) -> None:
        device = self._compute.device("cuda")
        model = self._model
        tokenizer = self._tokenizer
        streamer = self._streamer
        inner = self._inner_streamer

        self._history.append({"role": "user", "content": user_text})

        if self._past_key_values is None:
            inputs = tokenizer.apply_chat_template(
                self._history,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["past_key_values"] = SpeculativeCache(config=model.config)
        else:
            prev_ids = tokenizer.apply_chat_template(
                self._history[:-1],
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            full_ids = tokenizer.apply_chat_template(
                self._history,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            new_ids = full_ids[:, prev_ids.shape[1] :]
            total_len = self._past_length + new_ids.shape[1]
            input_ids = torch.cat(
                [torch.zeros(1, self._past_length, dtype=torch.long), new_ids],
                dim=1,
            ).to(device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones(1, total_len, dtype=torch.long, device=device),
                "past_key_values": self._past_key_values,
                "cache_position": torch.arange(
                    self._past_length, total_len, dtype=torch.long, device=device
                ),
            }

        max_new_tokens = 512
        eos_kwargs = {}
        if hasattr(inner, "eos_token_id"):
            eos_kwargs["eos_token_id"] = inner.eos_token_id

        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                    return_dict_in_generate=True,
                    **eos_kwargs,
                )
            elapsed = time.perf_counter() - t0
        except (KeyboardInterrupt, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "KeyboardInterrupt" not in str(exc):
                raise
            streamer.reset()
            self._history.pop()
            self._past_key_values = None
            self._past_length = 0
            self._generating = False
            self.call_from_thread(self._re_enable_input)
            return

        response = inner.get_final_response()
        self._history.append({"role": "assistant", "content": response})
        gen_tokens = inner.generated_tokens
        self._past_key_values = output.past_key_values
        self._past_length = output.sequences.shape[1]

        # Crop overshoot from KV cache
        valid = getattr(inner, "_valid_generated", 0) or max_new_tokens
        overshoot = max(0, gen_tokens - valid)
        if overshoot > 0:
            self._past_key_values.crop_overshoot(overshoot)
            self._past_length -= overshoot
        streamer.reset()

        self._total_tokens += gen_tokens
        self._total_time += elapsed
        self._generating = False

        self.post_message(GenerationDone(gen_tokens, elapsed, response))

    def _flush_buffers(self) -> None:
        """Drain all non-blocking buffers and update the UI."""
        # Drain log buffer
        if self._log_buffer:
            log_pane = self.query_one("#log-pane", VerticalScroll)
            while self._log_buffer:
                log_pane.mount(Static(self._log_buffer.popleft()))
            log_pane.scroll_end(animate=False)

        # Drain event buffer
        if self._event_buffer:
            log_pane = self.query_one("#log-pane", VerticalScroll)
            while self._event_buffer:
                log_pane.mount(Static(self._event_buffer.popleft()))
            log_pane.scroll_end(animate=False)

        # Drain metrics buffer
        metrics_dirty = False
        while self._metrics_buffer:
            self._process_metrics(self._metrics_buffer.popleft())
            metrics_dirty = True
        if metrics_dirty:
            self._update_charts()

        # Drain token buffer
        if not self._token_buffer:
            return
        chat_log = self.query_one("#chat-log", VerticalScroll)
        thinking_dirty = False
        assistant_dirty = False
        while self._token_buffer:
            channel, delta = self._token_buffer.popleft()
            if channel == "thinking":
                if self._thinking_widget is None:
                    self._thinking_widget = Static("", classes="thinking-msg", markup=False)
                    chat_log.mount(self._thinking_widget)
                self._thinking_text += delta
                thinking_dirty = True
            elif channel == "assistant":
                if self._assistant_widget is None:
                    self._assistant_widget = Static("", classes="assistant-msg", markup=False)
                    chat_log.mount(self._assistant_widget)
                self._assistant_text += delta
                assistant_dirty = True
        if thinking_dirty and self._thinking_widget is not None:
            self._thinking_widget.update(self._thinking_text)
        if assistant_dirty and self._assistant_widget is not None:
            self._assistant_widget.update(self._assistant_text)
        chat_log.scroll_end(animate=False)

    def on_generation_done(self, msg: GenerationDone) -> None:
        self._re_enable_input()
        tokens_per_sec = msg.gen_tokens / msg.elapsed if msg.elapsed > 0 else 0
        stats_widget = self.query_one("#stats", Static)
        stats_widget.update(
            f"{msg.gen_tokens} tokens in {msg.elapsed:.1f}s\n"
            f"{tokens_per_sec:.1f} token/s\n"
            f"Total: {self._total_tokens} tokens"
        )

    def _re_enable_input(self) -> None:
        prompt_input = self.query_one("#prompt", Input)
        prompt_input.disabled = False
        prompt_input.focus()

    def action_cancel_generation(self) -> None:
        if self._generating:
            workers = self.workers._workers  # type: ignore[attr-defined]
            for w in list(workers):
                if w.group == "generate" and w.is_running:
                    w.cancel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self.post_message(StatusUpdate(text))

    def on_status_update(self, msg: StatusUpdate) -> None:
        self.query_one("#status", Static).update(msg.text)


if __name__ == "__main__":
    try:
        LLMTermApp().run()
    except KeyboardInterrupt:
        pass
