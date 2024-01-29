import argparse
import regex

reg = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九 ]')

parser = argparse.ArgumentParser()
parser.add_argument('--vocab')
args = parser.parse_args()

# read csv
vocab = [line.rstrip().split()[0] for line in open(args.vocab)]

print('SIZE:', len(vocab))

jps = [token for token in vocab if reg.search(token)]
print('JP SIZE:', len(jps))
print('RATIO:', len(jps)/len(vocab))
