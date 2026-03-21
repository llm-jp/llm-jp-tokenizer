from transformers import LlamaTokenizerFast


class Llmjp4Tokenizer(LlamaTokenizerFast):
    _HARMONY_TOKENS: set[str] = {
        "<|start|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
    }

    # NOTE(odashi):
    # Response schemas are not recognized automatically.
    # We need to define them manually.
    # https://github.com/huggingface/trl/issues/4609
    _RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "role": {"const": "assistant"},
            "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)"},
            "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
            "tool_calls": {
                "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "function"},
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
                                "arguments": {
                                    "type": "object",
                                    "x-regex": r"<\|message\|>(.*)",
                                    "x-parser": "json",
                                    "additionalProperties": {"type": "any"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.response_schema = self._RESPONSE_SCHEMA

        self._harmony_token_ids = {
            self.convert_tokens_to_ids(token)
            for token in self._HARMONY_TOKENS
        }

    def _decode(self, token_ids: int | list[int], *args, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        result: list[str] = []
        prev_pos = 0

        # NOTE(odashi):
        # Ensure that text tokens are decoded without preceding Harmony tokens
        # to avoid incorrect addition of whitespaces.
        for pos, token_id in enumerate(token_ids, start=1):
            if token_id in self._harmony_token_ids or pos == len(token_ids):
                result.append(super()._decode(token_ids[prev_pos:pos], *args, **kwargs))
                prev_pos = pos

        return "".join(result)
