import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dicts', nargs='+')
parser.add_argument('--output')
args = parser.parse_args()

freqDict = {}

for path in tqdm(args.dicts):
    d = json.load(open(path))
    for k, v in d.items():
        if k not in freqDict:
            freqDict[k] = 0
        freqDict[k] += v

json.dump(freqDict, open(args.output, 'w'), ensure_ascii=False)
