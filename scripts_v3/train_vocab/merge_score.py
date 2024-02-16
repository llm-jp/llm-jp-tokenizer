import json
import sys


def main():
    base_vocab_json = sys.argv[1]
    with open(base_vocab_json, "r", encoding="utf8") as fin:
        vocab = json.load(fin)
        if isinstance(vocab, dict):
            if "model" in vocab and isinstance(vocab["model"], dict) and "vocab" in vocab["model"]:
                vocab = vocab["model"]["vocab"]
        else:
            assert isinstance(vocab, list), f"{base_vocab_json} is not valid"
    new_vocab = {_: None for _ in vocab}
    for vocab_score_json in sys.argv[2:]:
        with open(vocab_score_json, "r", encoding="utf8") as fin:
            vocab_score = json.load(fin)
            assert isinstance(vocab_score, dict), f"{vocab_score_json} must be a dict"
        for w, s in vocab_score.items():
            assert isinstance(w, str), f"{vocab_score_json}: {w} is not str"
            assert isinstance(s, float) or isinstance(s, int), f"{vocab_score_json}: {s} is not a number"
            new_vocab[w] = s
    errors = []
    for w, s in new_vocab.items():
        if s is None:
            errors.append(w)
    assert not errors, f"Error: {len(errors)} entries do not have score: {json.dumps(errors, ensure_ascii=False)}"
    json.dump(new_vocab, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
