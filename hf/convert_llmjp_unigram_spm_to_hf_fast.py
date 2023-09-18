import argparse

from tokenizers import decoders, models, normalizers, processors, Regex, Tokenizer


"""Tokenizer convert tool for llmjp-tokenizer.

You need to install some packages beforehand.

```console
$ pip -r requirements.txt
```

And retain `sentencepiece_model_pb2.py` from official site.

```console
$ curl -O https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py
```

Then you can convert sentence piece model file to huggingface fast tokenizer file (tokenizer.json).

```console
$ python convert_llmjp_unigram_spm_to_hf_fast.py -i ../model/ver2/code10k_en20k_ja30k.ver2.1.model -o ver2/code10k_en20k_ja30k.ver2.1_hf_fast/tokenizer.json
```

After the conversion, you can create fast tokenizer directory from `tokenizer.json`.

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(tokenizer_json_path)
```

We're still trying to create tokenizer instance via `AutoTokenizer.from_pretrained()`. Wait a moment.


This script originates from the following tokenizers codes:

https://github.com/huggingface/tokenizers/blob/v0.14.0/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py
https://github.com/huggingface/tokenizers/blob/v0.14.0/bindings/python/scripts/convert.py
"""


def format_special_token(label: str):
    return f"{label[:-1]}|LLM-jp{label[-1]}"


def get_proto():
    try:
        import sys

        sys.path.append(".")

        import sentencepiece_model_pb2 as model
    except Exception:
        raise Exception(
            "You don't seem to have the required protobuf file, in order to use this function you need to run `pip install protobuf` and `wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py` for us to be able to read the intrinsics of your spm_file. `pip install sentencepiece` is not required."
        )

    m = model.ModelProto()
    return m


def convert_llmjp_unigram_spm_to_hf(input_sp_model_path: str, eod_token: str) -> Tokenizer:
    proto = get_proto()
    proto.ParseFromString(open(input_sp_model_path, "rb").read())
    model_type = proto.trainer_spec.model_type
    assert model_type == 1, f"You're trying to run a `Unigram` model but you're file was trained with a different algorithm ({model_type=})"
    vocab = [(piece.piece, piece.score) for piece in proto.pieces]
    unk_id = proto.trainer_spec.unk_id
    special_tokens = [_ for _, piece in enumerate(proto.pieces) if piece.type > 1]
    for _, token_id in enumerate(special_tokens):
        vocab[token_id] = format_special_token(vocab[token_id][0]), vocab[token_id][1]
        special_tokens[_] = vocab[token_id][0]
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id, byte_fallback=True))
    tokenizer.add_special_tokens(special_tokens)
    normalizer_list = []
    precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
    if precompiled_charsmap:
        normalizer_list.append(normalizers.Precompiled(precompiled_charsmap))
    replacement = "▁"
    """
    # do not use Metaspace pre_tokenizer because all the continuous spaces are divided into single space sequences 
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using normalizer to insert "▁" to the beginning of text and to replace space to "▁"
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("^"), replacement),
            normalizers.Replace(Regex(r"\n" + replacement), "\n"),
            normalizers.Replace(Regex(" "), replacement),
        ]
    )
    eod = format_special_token(eod_token)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=["$0", eod],
        pair=["$A", eod, "$B:1", f"{eod}:1"],
        special_tokens=[
            (eod, tokenizer.get_vocab()[eod]),
        ],
    )
    """
    # do not use Metaspace decoder because all the heading spaces are removed
    tokenizer.decoder = decoders.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using Replace decoders to remove the extra space char at the beginning of text and replace "▁" to space
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.Replace(Regex(replacement), " "),
            decoders.Fuse(),
            decoders.Replace(Regex(r"\n"), "\n "),
            decoders.Replace(Regex(f"^ "), ""),
        ]
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_sp_model_path",
        required=True,
        type=str,
        help="path for input sentencepiece unigram model file",
    )
    parser.add_argument(
        "-o", "--output_hf_tokenizer_json_path",
        required=True,
        type=str,
        help="path for output huggingface tokenizers json file",
    )
    parser.add_argument(
        "-e", "--eod_token",
        default="<EOD>",
        type=str,
        help="the end-of-document token which appended to the results of encode(), default='<EOD>'",
    )
    args = parser.parse_args()
    print("converting", args.input_sp_model_path, "to" ,args.output_hf_tokenizer_json_path)
    tokenizer = convert_llmjp_unigram_spm_to_hf(args.input_sp_model_path, args.eod_token)
    tokenizer.save(args.output_hf_tokenizer_json_path)


if __name__ == "__main__":
    main()
