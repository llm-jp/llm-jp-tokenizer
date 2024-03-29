import argparse
import json
import sys
from typing import Any

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def print_results(sp: SentencePieceProcessor, eod_id: int, hf, text: str):
    print(f"input text: {dumps(text)}")
    encoded_sp = sp.encode(text, out_type="immutable_proto")

    sp_result = [_.piece for _ in encoded_sp.pieces]
    hf_result = hf.tokenize(text)
    print("encoded pieces")
    print("       sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_result == sp_result else 'NG'} hf:", dumps(hf_result))
    print()

    sp_result = [_.id for _ in encoded_sp.pieces] + [eod_id]
    hf_result = hf(text, add_special_tokens=True)["input_ids"]
    print("encoded ids")
    print("       sp:", dumps(sp_result))
    print(f" >> {'OK' if hf_result == sp_result else 'NG'} hf:", dumps(hf_result))
    print()

    sp_result = sp.decode([_.id for _ in encoded_sp.pieces])
    hf_result = hf.decode(hf(text, add_special_tokens=False)["input_ids"])
    print(f"decoded pieces  {dumps(text)}")
    print(f" >> {'OK' if sp_result == text else 'NG'} sp: {dumps(sp_result)}")
    print(f" >> {'OK' if hf_result == text else 'NG'} hf: {dumps(hf_result)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spm",
        required=True,
        type=str,
        help="path for sentencepiece model file",
    )
    parser.add_argument(
        "--eod-id",
        required=True,
        type=int,
        help="eod token id",
    )
    parser.add_argument(
        "--fast",
        default=None,
        type=str,
        help="path or name for huggingface fast tokenizer",
    )
    parser.add_argument(
        "--slow",
        default=None,
        type=str,
        help="path or name for huggingface slow tokenizer",
    )
    parser.add_argument(
        "target_files",
        nargs='*',
        help="target text files (use default test strings if not specified)",
    )
    args = parser.parse_args()

    sp = SentencePieceProcessor(args.spm)
    assert (args.fast is None) != (args.slow is None), "you need to specify one of --fast or --slow"
    if args.fast:
        hf = AutoTokenizer.from_pretrained(args.fast, trust_remote_code=True)
    else:
        hf = AutoTokenizer.from_pretrained(args.slow, legacy=True, use_fast=False, trust_remote_code=True)

    if args.target_files:
        for file in args.target_files:
            with open(file, "r", encoding="utf8") as fin:
                for line in fin:
                    print_results(sp, args.eod_id, hf, line.rstrip("\n"))
    else:
        test_strings = [
            "  これはテストです。\n  これもテストです。  ",
            "  this is x\n  this is x  ",
            "  <unk>  <s>  </s>  <mask>  <pad>  <CLS>  <SEP>  <EOD>  "
        ]
        for text in test_strings:
            print_results(sp, args.eod_id, hf, text)


if __name__ == "__main__":
    main()
