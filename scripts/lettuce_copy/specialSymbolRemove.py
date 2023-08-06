import sys

SYMBOL = '▁'

newVocab = set()
newVocabList = []

newLineFlag = False

for line in open(sys.argv[1]):
    if line == '\n':
        newLineFlag = True
        continue
    elif newLineFlag:
        newVocab.add('\n')
        newVocabList.append('\n')
        newLineFlag = False
        continue

    line = line.rstrip()
    token, score = line.split('\t')
    if set(list(token)) == 1 and token[0] == SYMBOL:
        pass
    elif token.startswith(SYMBOL):
        token = token.replace(SYMBOL, '')
    if token not in newVocab:
        newVocab.add(token)
        newVocabList.append(token)

for line in newVocabList:
    print('%s\t0.0'%line)

#print('size of new vocab:', len(newVocab))
