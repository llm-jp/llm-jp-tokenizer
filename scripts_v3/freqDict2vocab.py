import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dicts', nargs='+')
parser.add_argument('--numTokens', type=int)
parser.add_argument('--minFreq', type=int, default=3)
parser.add_argument('--output')
args = parser.parse_args()

freqDict = {}

for path in tqdm(args.dicts):
    d = json.load(open(path))
    for k, v in d.items():
        if k not in freqDict:
            freqDict[k] = 0
        freqDict[k] += v

with open(args.output, 'w') as f:
    c = 0
    for k, v in sorted(freqDict.items(), key=lambda x:x[1], reverse=True):
        if v < args.minFreq:
            break

        print(k, v)
        f.write(k+'\n')

        c += 1
        if c==args.numTokens:
            break
