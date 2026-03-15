from openai_harmony import (
    HarmonyEncodingName,
    Role,
    StreamableParser,
    StreamState,
    load_harmony_encoding,
)
from transformers import TextStreamer


class HarmonyStreamer(TextStreamer):
    """Streamer for GPT-OSS Harmony response format.

    Uses openai-harmony's StreamableParser to parse the structured output with
    <|start|>, <|channel|>, <|message|>, <|end|>, and <|return|> special tokens.
    Displays analysis/thinking traces in grey and the final response in normal color.
    """

    _GREY = "\033[90m"
    _RESET = "\033[0m"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._start_token_id = self._encoding.encode("<|start|>", allowed_special={"<|start|>"})[0]
        self._endoftext_token_id = self._encoding.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        self.reset()

    def reset(self):
        """Clear state between turns."""
        self._parser = StreamableParser(self._encoding, Role.ASSISTANT, strict=False)
        self._prev_state = self._parser.state
        self._prev_channel = None
        self._final_text = []
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.generated_tokens = 0
        self._response_complete = False
        self._valid_generated = 0

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("HarmonyStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Fast path: skip expensive .tolist() (server sync ~150ms) for
        # post-EOS overshoot tokens.  Still count them via shape metadata
        # (local, no sync) so overshoot is computed correctly for KV crop.
        if self._response_complete:
            self.generated_tokens += value.numel()
            return

        for token_id in value.tolist():
            self.generated_tokens += 1
            # <|endoftext|> is never valid content — skip it in any parser state.
            # With strict=False the parser would pass it through as a content delta.
            if token_id == self._endoftext_token_id:
                return
            # After <|end|>, the parser is in EXPECT_START. Only <|start|>
            # is valid (next message in a multi-message response). Skip
            # everything else (<|return|>, trailing tokens, etc.).
            if self._parser.state == StreamState.EXPECT_START:
                if token_id != self._start_token_id:
                    continue
            self._parser.process(token_id)
            state = self._parser.state
            channel = self._parser.current_channel

            # Final response completed (<|end|> → EXPECT_START after "final"
            # channel): record valid token count. Tokens after the analysis
            # channel's <|end|> are NOT overshoot — the final response follows.
            if state == StreamState.EXPECT_START and self._prev_channel == "final":
                self._response_complete = True
                self._valid_generated = self.generated_tokens

            # Transition out of content: reset ANSI for analysis
            if self._prev_state == StreamState.CONTENT and state != StreamState.CONTENT:
                if self._prev_channel == "analysis":
                    print(self._RESET, end="", flush=True)

            # Transition into content: print channel label
            if state == StreamState.CONTENT and self._prev_state != StreamState.CONTENT:
                if channel == "analysis":
                    print(f"\n{self._GREY}Thinking: ", end="", flush=True)
                elif channel == "final":
                    print("\n\nAssistant: ", end="", flush=True)

            # Stream content deltas
            if state == StreamState.CONTENT:
                delta = self._parser.last_content_delta
                if delta:
                    print(delta, end="", flush=True)
                    if channel == "final":
                        self._final_text.append(delta)

            self._prev_state = state
            self._prev_channel = channel

    def end(self):
        if not self._response_complete:
            try:
                self._parser.process_eos()
            except Exception:
                # Response cut off by max_new_tokens before <|end|> —
                # parser is in a non-terminal state (e.g. CONTENT).
                pass
        if self._prev_state == StreamState.CONTENT:
            if self._prev_channel == "analysis":
                print(self._RESET, end="", flush=True)
        self.next_tokens_are_prompt = True
        print()

    @property
    def eos_token_id(self):
        """Harmony stop tokens for use as eos_token_id in model.generate()."""
        return self._encoding.stop_tokens_for_assistant_actions()

    def get_final_response(self):
        """Return text from only the final channel."""
        if not self._final_text:
            return ""
        return "".join(self._final_text).strip()
