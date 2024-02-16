import json
import re
import sys
import unicodedata

from sudachipy import dictionary, tokenizer


class SudachiCharNormalizer:
    def __init__(self, rewrite_def_path="./rewrite.def"):
        self.ignore_normalize_set = set()
        self.replace_char_map = {}
        self.read_rewrite_def(rewrite_def_path)
        
    def read_rewrite_def(self, rewrite_def_path):
        with open(rewrite_def_path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                cols = line.split()
                if len(cols) == 1:
                    if len(cols[0]) != 1:
                        raise Exception("'{}' is not a single character at line {}".format(cols[0], i))
                    self.ignore_normalize_set.add(cols[0])
                elif len(cols) == 2:
                    if cols[0] in self.replace_char_map:
                        raise Exception("'Replacement for '{}' defined again at line {}".format(cols[0], i))
                    self.replace_char_map[cols[0]] = cols[1]
                else:
                    raise Exception("Invalid format '{}' at line {}".format(line, i))
                    
    def rewrite(self, text):
        chars_after = []

        offset = 0
        next_offset = 0
        i = -1
        while True:
            i += 1
            if i >= len(text):
                break
            textloop = False
            offset += next_offset
            next_offset = 0

            # 1. replace char without normalize
            for l in range(len(text) - i, 0, -1):
                replace = self.replace_char_map.get(text[i:i+l])
                if replace:
                    chars_after.append(replace)
                    next_offset += len(replace) - l
                    i += l - 1
                    textloop = True
                    continue
            if textloop:
                continue

            # 2. normalize    
            # 2-1. capital alphabet (not only latin but greek, cyrillic, etc) -> small
            original = text[i]
            lower = original.lower()
            if lower in self.ignore_normalize_set:
                replace = lower
            else:
                # 2-2. normalize (except in ignoreNormalize)
                # e.g. full-width alphabet -> half-width / ligature / etc.
                replace = unicodedata.normalize("NFKC", lower)
            next_offset = len(replace) - 1
            chars_after.append(replace)

        return "".join(chars_after)


def main():
    debug = False
    base_vocab_path = sys.argv[1]
    candidate_vocab_path = sys.argv[2]
    sudachi_mode = sys.argv[3].upper()
    rewrite_def_path = sys.argv[4]
    sudachi_dict_src_paths = sys.argv[5:]
    print("Executing: python", " ".join(sys.argv), file=sys.stderr)
 
    with open(base_vocab_path, "r", encoding="utf8") as fin:
        base_vocab = json.load(fin)
    print(f"{len(base_vocab)=}", file=sys.stderr)

    with open(candidate_vocab_path, "r", encoding="utf8") as fin:
        candidate_vocab = json.load(fin)
    print(f"{len(candidate_vocab)=}", file=sys.stderr)

    assert sudachi_mode in ["A", "B", "C"]

    normalizer = SudachiCharNormalizer(rewrite_def_path)

    sudachi_dict = {}
    for src_path in sudachi_dict_src_paths:
        with open(src_path, "r", encoding="utf8") as fin:
            for _ in fin:
                r = _.rstrip("\n").split(",")
                assert len(r) == 19, f"ERROR]:{r}"
                mode = r[14]
                if mode > sudachi_mode:
                    continue
                s = r[0].encode("unicode-escape").replace(b"\\\\u", b"\\u").decode("unicode-escape")
                r = normalizer.rewrite(s)
                sudachi_dict[r] = (mode, src_path)
    print(f"{len(sudachi_dict)=}", file=sys.stderr)

    new_vocab = []
    for v, score in candidate_vocab.items():
        if v in base_vocab:
            continue
        if v.startswith("▁"):
            r = normalizer.rewrite(v[1:])
        else:
            r = normalizer.rewrite(v)
        if r in sudachi_dict:
            new_vocab.append(v)
        elif debug:
            if True or re.fullmatch("[ぁ-ん]{2,}[、。]?|[ぁ-ん][、。]", v):
                print(f"{score}\t{v}", file=sys.stderr)
    print(f"{len(new_vocab)=}", file=sys.stderr)
    json.dump(new_vocab, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
