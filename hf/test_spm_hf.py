import sys
from typing import List, Optional

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer

from llmjp_hf_slow_utils import decode_by_sp_model

USAGE = f"""Usage: python test_spm_hf.py sentencepiece_model_file huggingface_tokenizer_json [input_files]"""


def print_results(sp: SentencePieceProcessor, hf_slow: Tokenizer, hf_fast, text: str):
    print(f"input text: '{text}'")
    encoded_sp = sp.encode(text, out_type="immutable_proto")
    encoded_hf_slow = hf_slow.tokenize(text)
    encoded_hf_fast = hf_fast.tokenize(text)
    # encoded_hf_fast = hf_fast.encode(text)
    sp_result = [_.piece for _ in encoded_sp.pieces]
    hf_slow_result = encoded_hf_slow
    hf_fast_result = encoded_hf_fast
    # hf_fast_result = encoded_hf_fast.tokens
    print("encoded pieces")
    print("           sp:", sp_result)
    print(f">> {'OK' if hf_slow_result == sp_result else 'NG'} hf_slow:", hf_slow_result)
    print(f">> {'OK' if hf_fast_result == sp_result else 'NG'} hf_fast:", hf_fast_result)
    print()
    sp_result = [_.id for _ in encoded_sp.pieces]
    hf_slow_result = hf_slow(text, add_special_tokens=False)["input_ids"]
    hf_fast_result = hf_fast(text, add_special_tokens=False)["input_ids"]
    # hf_fast_result = encoded_hf_fast.ids
    print("encoded ids")
    print("           sp:", sp_result)
    print(f">> {'OK' if hf_slow_result == sp_result else 'NG'} hf_slow:", hf_slow_result)
    print(f">> {'OK' if hf_fast_result == sp_result else 'NG'} hf_fast:", hf_fast_result)
    print()
    sp_result = sp.decode([_.id for _ in encoded_sp.pieces])
    hf_slow_result = decode_by_sp_model(hf_slow, hf_slow_result)
    hf_fast_result = hf_fast.decode(hf_fast_result)
    # hf_fast_result = hf_fast.decode(encoded_hf_fast.ids)
    print(f"decoded pieces '{text}'")
    print(f">> {'OK' if sp_result == text else 'NG'}      sp: '{sp_result}'")
    print(f">> {'OK' if hf_slow_result == text else 'NG'} hf_slow: '{hf_slow_result}'")
    print(f">> {'OK' if hf_fast_result == text else 'NG'} hf_fast: '{hf_fast_result}'")
    print()


def main(spm_path: str, hf_slow_dir_path: str, hf_fast_json_path: str, target_files: Optional[List[str]]=None):
    sp = SentencePieceProcessor(spm_path)
    hf_slow = AutoTokenizer.from_pretrained(hf_slow_dir_path , legacy=True, use_fast=False)
    hf_fast = AutoTokenizer.from_pretrained(hf_fast_json_path)
    # hf_fast = PreTrainedTokenizerFast(tokenizer_file=hf_fast_json_path)
    # hf_fast = Tokenizer.from_file(hf_fast_json_path)

    if target_files:
        for file in target_files:
            with open(file, "r", encoding="utf8") as fin:
                for line in fin:
                    print_results(sp, hf_slow, hf_fast, line.rstrip("\n"))
    else:
        test_strings = [
            "  これはテストです。\n  これもテストです。  ",
            "  this is x\n  this is x  ",
            "  <unk>  <s>  </s>  <mask>  <pad>  <CLS>  <SEP>  <EOD>  <|endoftext|> <|padding|>"
        ]
        for text in test_strings:
            print_results(sp, hf_slow, hf_fast, text)


if __name__ == "__main__":
    main(*sys.argv[1:])
