import json
import sys


def main():
    argv = sys.argv[1:]
    if argv[0] == "--head":
        head = int(argv[1])
        argv = argv[2:]
    else:
        head = None
    new_vocab = {}
    for vocab_json in argv:
        with open(vocab_json, "r", encoding="utf8") as fin:
            vocab = json.load(fin)
            if isinstance(vocab, dict):
                if "model" in vocab and isinstance(vocab["model"], dict) and "vocab" in vocab["model"]:
                    vocab = vocab["model"]["vocab"]
                else:
                    vocab = list(vocab.keys()) # [:13005]
            else:
                assert isinstance(vocab, list), f"{vocab_json} is not valid"
            for _ in vocab:
                if head is not None and len(new_vocab) >= head:
                    break
                assert isinstance(_, str), f"{vocab_json}: {_} is not str"
                new_vocab[_] = None
    json.dump(list(new_vocab.keys()), sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
