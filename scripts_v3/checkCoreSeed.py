import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vocab')
parser.add_argument('--coreSeeds', nargs='+')
parser.add_argument('--keyword')
parser.add_argument('--onlyNocore', action='store_true')
args = parser.parse_args()

vocab = [line.strip() for line in open(args.vocab)]
coreSeeds = set([line.strip() for path in args.coreSeeds for line in open(path)])

for line in vocab:
    word = line.split()[0]
    if args.keyword in word:
        if args.onlyNocore and word in coreSeeds:
            continue
        print(word+('★' if word in coreSeeds else ''))

