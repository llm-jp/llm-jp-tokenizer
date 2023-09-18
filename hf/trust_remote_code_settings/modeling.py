from typing import List, Union

from torch import Tensor
from transformers import T5Tokenizer


class LlmJpT5Tokenizer(T5Tokenizer):
    def decode(tokenizer: T5Tokenizer, token_ids: Union[List[int], Tensor]) -> str:
        if token_ids is None:
            return None
        elif len(token_ids) == 0:
            return ""
        else:
            return tokenizer.sp_model.decode(token_ids)
