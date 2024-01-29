import argparse
import csv
import regex

parser = argparse.ArgumentParser()
parser.add_argument('--dictionary', help='path to sudachi-dict (csv) like core_lex.csv')
parser.add_argument('--output')
args = parser.parse_args()

# read csv

vocab = set()
with open(args.dictionary) as f:
    reader = csv.reader(f)
    for row in reader:
        word = row[0]
        vocab.add(word)

# filtering
# 平仮名，片仮名，漢字にマッチする
reg = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九 ]')

filteredVocab = set()
for word in vocab:
    if reg.search(word):
        filteredVocab.add(word)
        #print('ADD:', word)
    else:
        print('REMOVE:', word)

print('original vocab size:', len(vocab))
print('filtered vocab size:', len(filteredVocab))

with open(args.output, 'w') as f:
    for word in filteredVocab:
        f.write(word+'\n')
