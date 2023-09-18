import json
import sys
from typing import Any, List, Optional

from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer


USAGE = f"""Usage: python test_spm_hf_fast.py sentencepiece_model_file fast_tokenizer_json_path [input_files]"""


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def print_results(sp: SentencePieceProcessor, hf_fast, text: str):
    print(f"input text: {dumps(text)}")
    encoded_sp = sp.encode(text, out_type="immutable_proto")
    encoded_hf_fast = hf_fast.encode(text, add_special_tokens=False)

    sp_result = [_.piece for _ in encoded_sp.pieces]
    hf_fast_result = encoded_hf_fast.tokens
    print("encoded pieces")
    print("            sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_fast_result == sp_result else 'NG'} hf_fast:", dumps(hf_fast_result))
    print()

    sp_result = [_.id for _ in encoded_sp.pieces]
    hf_fast_result = encoded_hf_fast.ids
    print("encoded ids")
    print("            sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_fast_result == sp_result else 'NG'} hf_fast:", dumps(hf_fast_result))
    print()

    sp_result = sp.decode([_.id for _ in encoded_sp.pieces])
    hf_fast_result = hf_fast.decode(encoded_hf_fast.ids)
    print(f"decoded pieces  {dumps(text)}")
    print(f" >> {'OK' if sp_result == text else 'NG'}      sp: {dumps(sp_result)}")
    print(f" >> {'OK' if hf_fast_result == text else 'NG'} hf_fast: {dumps(hf_fast_result)}")
    print()


def main(spm_path: str, hf_fast_json_path: str, *target_files):
    sp = SentencePieceProcessor(spm_path)
    hf_fast = Tokenizer.from_file(hf_fast_json_path)

    if target_files:
        for file in target_files:
            with open(file, "r", encoding="utf8") as fin:
                for line in fin:
                    print_results(sp, hf_fast, line.rstrip("\n"))
    else:
        test_strings = [
            "  これはテストです。\n  これもテストです。  ",
            "  this is x\n  this is x  ",
            "  <unk>  <s>  </s>  <mask>  <pad>  <CLS>  <SEP>  <EOD>  "
        ]
        for text in test_strings:
            print_results(sp, hf_fast, text)


if __name__ == "__main__":
    main(*sys.argv[1:])
