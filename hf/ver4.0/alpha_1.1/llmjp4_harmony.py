# Generic parser for OpenAI Harmony format.

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

from transformers import PreTrainedTokenizerBase as TokenizerLike


class HarmonyMessageEndType(Enum):
    INCOMPLETE = 0
    END = 1
    CALL = 2


@dataclass(frozen=True)
class HarmonySequence:
    """A data class representing a sequence of tokens in the Harmony format."""
    token_ids: list[int]
    start: int  # Start position of the sequence in the original token sequence


@dataclass(frozen=True)
class HarmonyMessage:
    """A data class representing a message in the Harmony format."""
    end: HarmonyMessageEndType
    role: HarmonySequence | None = None
    channel: HarmonySequence | None = None
    constrain: HarmonySequence | None = None
    content: HarmonySequence | None = None


class HarmonyMessageParser:
    """A parser that performs lexical analysis to extract Harmony messages."""

    def __init__(self, tokenizer: TokenizerLike):
        vocab = tokenizer.get_vocab()
        self._begin_map = {
            vocab["<|start|>"]: "role",
            vocab["<|channel|>"]: "channel",
            vocab["<|constrain|>"]: "constrain",
            vocab["<|message|>"]: "content",
        }
        self._end_map = {
            vocab["<|end|>"]: HarmonyMessageEndType.END,
            vocab["<|return|>"]: HarmonyMessageEndType.END,
            vocab["<|call|>"]: HarmonyMessageEndType.CALL,
        }

    def iter_messages(self, token_ids: Sequence[int]) -> Iterator[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Args:
            token_ids: A sequence of token ids to be parsed.
        
        Yields:
            Detected HarmonyMessages.
        """

        message_dict: dict[str, HarmonySequence] = {}
        section: str | None = None  # None indicates out-of-message.
        text_ids: list[int] = []
        text_start: int | None = None

        for token_position, token_id in enumerate(token_ids):
            if token_id in self._begin_map:
                if section is not None:
                    message_dict[section] = HarmonySequence(
                        token_ids=text_ids,
                        start=text_start,
                    )
                section = self._begin_map[token_id]
                text_ids = []
                text_start = token_position + 1

            elif token_id in self._end_map:
                if section is not None:
                    message_dict[section] = HarmonySequence(
                        token_ids=text_ids,
                        start=text_start,
                    )

                yield HarmonyMessage(**message_dict, end=self._end_map[token_id])

                message_dict = {}
                section = None
                text_ids = []
                text_start = None
    
            else:
                if section is not None:
                    text_ids.append(token_id)
        
        if section is not None:
            message_dict[section] = HarmonySequence(
                token_ids=text_ids,
                start=text_start,
            )
            yield HarmonyMessage(**message_dict, end=HarmonyMessageEndType.INCOMPLETE)

    def get_all_messages(self, token_ids: Sequence[int]) -> list[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Args:
            token_ids: A sequence of token ids to be parsed.
        
        Returns:
            A list of detected HarmonyMessages.
        """
        return list(self.iter_messages(token_ids))
    
    def reverse_iter_messages(self, token_ids: Sequence[int]) -> Iterator[HarmonyMessage]:
        """
        Parse given token ids into messages in reverse order.

        Args:
            token_ids: A sequence of token ids to be parsed.
        
        Yields:
            Detected HarmonyMessages in reverse order.
        """
        end_position = len(token_ids)

        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == self._start_id:
                yield next(self.iter_messages(token_ids[i:end_position]))
                end_position = i
