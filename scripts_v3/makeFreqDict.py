import argparse
import json
from tqdm import tqdm
# 頻度をカウントして，順番にソート

parser = argparse.ArgumentParser()
parser.add_argument('--vocab')
parser.add_argument('--text')
parser.add_argument('--output')
args = parser.parse_args()

vocab2freq = {line.strip():0 for line in open(args.vocab)}

data = [line.strip() for line in open(args.text)]

for line in tqdm(data):
    for word in vocab2freq:
        if word in line:
            vocab2freq[word] += line.count(word)

json.dump(vocab2freq, open(args.output, 'w'), ensure_ascii=False)
