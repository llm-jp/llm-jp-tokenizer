import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
parser.add_argument('--only-all-diff', action='store_true')
parser.add_argument('-m', '--model', nargs='+')
args = parser.parse_args()

tknzrs = []

for i, path in enumerate(args.model):
    print('TOKENIZER%d:'%i, path)
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    tknzrs.append(sp)

for line in open(args.data):
    segs = [' '.join(tknzr.encode_as_pieces(line)) for tknzr in tknzrs]

    if not args.only_all_diff or \
        (args.only_all_diff and len(set(segs))==len(tknzrs)):
        print('-'*20)
        print(line)
        for i, seg in enumerate(segs):
            print('TKNZR%d:'%i, seg)

