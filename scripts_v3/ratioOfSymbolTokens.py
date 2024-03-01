import argparse
import regex

SYMBOLS1='!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
SYMBOLS2='、。，．・：；？！゛゜´｀¨＾￣＿ヽヾゝゞ〃仝々〆〇ー―‐／＼～∥｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋－±×÷＝≠＜＞≦≧∞∴♂♀°′″℃￥＄￠￡％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨￢⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪'

SYMBOLS = set(list(SYMBOLS1) + list(SYMBOLS2))


jpreg = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九 ]')
#syreg = regex.compile(r'[%s]'%SYMBOLS)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab')
args = parser.parse_args()

# read csv
vocab = [line.rstrip().split()[0] for line in open(args.vocab)]

print('SIZE:', len(vocab))

jps = [token for token in vocab if jpreg.search(token)]
print('JP SIZE:', len(jps))
print('JP RATIO:', len(jps)/len(vocab))

def includesSym(token):
    for c in token:
        if c in SYMBOLS:
            return True
    return False

syms = [token for token in vocab if includesSym(token)]
print('SYMBOL SIZE:', len(syms))

jpsyms = [token for token in vocab if jpreg.search(token) and includesSym(token)]
print('JP and SYMBOL SIZE:', len(jpsyms))
