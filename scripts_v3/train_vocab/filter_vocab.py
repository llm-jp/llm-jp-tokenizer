import json
import re
import sys


def main():
    filter_pattern = re.compile(sys.argv[1])
    if len(sys.argv) > 2:
        max_entries = int(sys.argv[2])
    else:
        max_entries = None
    vocab = json.load(sys.stdin)
    if isinstance(vocab, dict):
        vocab = dict(sorted(vocab.items(), key=lambda _: -_[1]))
    filtered = [_ for _ in vocab if filter_pattern.fullmatch(_)]
    if max_entries is not None:
        filtered = filtered[:max_entries]
    if isinstance(vocab, dict):
        filtered = {_: vocab[_] for _ in filtered}
    json.dump(filtered, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
