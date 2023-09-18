import json
import sys
from typing import Any, List, Optional

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from tokenizers import Tokenizer

from llmjp_hf_slow_utils import decode_by_sp_model


USAGE = f"""Usage: python test_spm_hf_slow.py sentencepiece_model_file slow_tokenizer_name_or_path [input_files]"""


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def print_results(sp: SentencePieceProcessor, hf_slow, text: str):
    print(f"input text: {dumps(text)}")
    encoded_sp = sp.encode(text, out_type="immutable_proto")
    encoded_hf_slow = hf_slow.tokenize(text)

    sp_result = [_.piece for _ in encoded_sp.pieces]
    hf_slow_result = encoded_hf_slow
    print("encoded pieces")
    print("            sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_slow_result == sp_result else 'NG'} hf_slow:", dumps(hf_slow_result))
    print()

    sp_result = [_.id for _ in encoded_sp.pieces]
    hf_slow_result = hf_slow(text, add_special_tokens=False)["input_ids"]
    print("encoded ids")
    print("            sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_slow_result == sp_result else 'NG'} hf_slow:", dumps(hf_slow_result))
    print()

    sp_result = sp.decode([_.id for _ in encoded_sp.pieces])
    hf_slow_result = decode_by_sp_model(hf_slow, hf_slow_result)
    print(f"decoded pieces  {dumps(text)}")
    print(f" >> {'OK' if sp_result == text else 'NG'}      sp: {dumps(sp_result)}")
    print(f" >> {'OK' if hf_slow_result == text else 'NG'} hf_slow: {dumps(hf_slow_result)}")
    print()


def main(spm_path: str, hf_slow_dir_path: str, target_files: Optional[List[str]]=None):
    sp = SentencePieceProcessor(spm_path)
    hf_slow = AutoTokenizer.from_pretrained(hf_slow_dir_path , legacy=True, use_fast=False)

    if target_files:
        for file in target_files:
            with open(file, "r", encoding="utf8") as fin:
                for line in fin:
                    print_results(sp, hf_slow, line.rstrip("\n"))
    else:
        test_strings = [
            "  これはテストです。\n  これもテストです。  ",
            "  this is x\n  this is x  ",
            "  <unk>  <s>  </s>  <mask>  <pad>  <CLS>  <SEP>  <EOD>  "
        ]
        for text in test_strings:
            print_results(sp, hf_slow, text)


if __name__ == "__main__":
    main(*sys.argv[1:])
