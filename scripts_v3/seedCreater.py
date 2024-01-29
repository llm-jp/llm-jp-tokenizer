# lm.pyのbuild vocabをモジュール化

import argparse
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--minFreq', type=int)
parser.add_argument('--maxLength', type=int)
parser.add_argument('--headPrefix', default='▁')
parser.add_argument('--output')
parser.add_argument('--spaceSplit', action='store_true')
args = parser.parse_args()

if args.spaceSplit:
    data = [args.headPrefix + word for line in open(args.data) for word in line.strip().split()]
else:
    data = [args.headPrefix + line.replace(' ', args.headPrefix) for line in open(args.data)]

# data is a list of string
unigramFreq = 0
wordCountDict = defaultdict(lambda:0)

for line in tqdm(data):
    for i in range(len(line)):
        for l in range(min(i+1, args.maxLength)):
            w = line[i-l:i+1]
            wordCountDict[w] += 1
    unigramFreq += len(line)

vocab = set(k for k,v in wordCountDict.items() if len(k)==1 or args.minFreq<=v)

# add unk
vocab.add('<unk>')

# remove none
if '' in vocab:
    vocab.remove('')
if '^Z' in vocab:
    vocab.remove('^Z')

vocab = sorted(list(vocab))

with open(args.output, 'w') as f:
    for word in vocab:
        f.write(word + '\n')
