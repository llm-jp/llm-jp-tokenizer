import sys

import sentencepiece as spm
from tokenizers import Tokenizer


def print_results(sp, hf, text: str):
    print("input text", f'"{text}"')
    encorded_sp = sp.encode(text, out_type="immutable_proto")
    encorded_hf = hf.encode(text)
    sp_result = [_.piece for _ in encorded_sp.pieces]
    hf_result = encorded_hf.tokens
    print("encoded pieces")
    print("OK" if hf_result == sp_result else "DIFFERENT")
    print("  sp:", sp_result)
    print("  hf:", hf_result)
    sp_result = [_.id for _ in encorded_sp.pieces]
    hf_result = encorded_hf.ids
    print("encoded ids")
    print("OK" if hf_result == sp_result else "DIFFERENT")
    print("  sp:", sp_result)
    print("  hf:", hf_result)
    sp_result = sp.decode([_.id for _ in encorded_sp.pieces])
    hf_result = hf.decode(encorded_hf.ids)
    print("decoded pieces")
    print("OK" if hf_result == sp_result else "DIFFERENT")
    print("  sp:", f'"{sp_result}"')
    print("  hf:", f'"{hf_result}"')
    print()


def main(spm_path: str, hf_json_path: str):
    sp = spm.SentencePieceProcessor(spm_path)
    hf = Tokenizer.from_file(hf_json_path)

    test_strings = [
        "  これはテストです。  これもテストです。  ",
        "  this is x  this is x  ",
        "  <unk>  <s>  </s>  <mask>  <pad>  <CLS>  <SEP>  <EOD>  "
    ]

    for text in test_strings:
        print_results(sp, hf, text)

    while True:
        text = input()
        print_results(sp, hf, text)


if __name__ == "__main__":
    main(spm_path=sys.argv[1], hf_json_path=sys.argv[2])
