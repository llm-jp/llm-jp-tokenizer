import numpy as np
from collections import defaultdict
import pickle
import random
from tqdm import tqdm

class MultigramLM:
    def __init__(self, maxLength=5, minFreq=4, data=None, 
                 wordHeadPrefix=None, wordInterPrefix=None, unkToken='<unk>',
                 fixedSeeds=None, initialSeeds=None):
        self.maxLength = maxLength
        self.minFreq = minFreq
        self.theta = None
        self.fixedSeeds = set(fixedSeeds) if fixedSeeds else None
        self.initialSeeds = set(initialSeeds) if initialSeeds else None
        
        self.replaceSpaceMode = False

        self.wordHeadPrefix = wordHeadPrefix    # _th is (sentencepiece like)
        self.wordInterPrefix = wordInterPrefix  # th @@is (word piece like)

        # memo set wordHeadPrefix as '' if you want to recognize whitespace as word boundary.
        self.unkToken = unkToken
        
        if data:
            self.buildVocab(data, fixedSeeds, initialSeeds)

    def buildVocab(self, data, fixedSeeds=None, initialSeeds=None):
        if self.fixedSeeds is None and fixedSeeds is not None:
            print('set fixed seeds')
            self.fixedSeeds = set(fixedSeeds)

        print('build initial vocab...')

        if initialSeeds:
            self.vocab = set(initialSeeds)
            self.unigramFreq = sum(len(line) for line in data)
        else:
            # data is a list of string
            self.unigramFreq = 0
            wordCountDict = defaultdict(lambda:0)

            for line in tqdm(data):
                for i in range(len(line)):
                    for l in range(min(i+1, self.maxLength)):
                        w = line[i-l:i+1]
                        wordCountDict[w] += 1
                self.unigramFreq += len(line)

            self.vocab = set(k for k,v in wordCountDict.items() if len(k)==1 or self.minFreq<=v)
            
        if self.fixedSeeds is not None:
            self.vocab |= self.fixedSeeds
        
        # add unk
        self.vocab.add(self.unkToken)

        # remove none
        if '' in self.vocab:
            self.vocab.remove('')

        # reset maxlength
        self.maxLength = max(len(w) for w in self.vocab)

        self.word2id = {w:i for i,w in enumerate(sorted(list(self.vocab)))}
        self.id2word = {i:w for w,i in self.word2id.items()}

        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        print('>>> BUILD VOCABULARY')
        print('possible n-grams (n=%d):'%self.maxLength, len(self.vocab))

        self.randomizeTheta()
        print('>>> INITIALIZE THETA')\
        
    def randomizeTheta(self):
        self.theta = np.random.rand(len(self.vocab))
        self.theta = self.theta/sum(self.theta)

    def __addLineToVocab(self, line):
        for i in range(len(line)):
            for l in range(min(i+1, self.maxLength)):
                w = line[i-l:i+1]
                self.vocab.add(w)
        self.unigramFreq += len(line)

    def addWordToVocab(self, word, p=0.0):
        if word in self.vocab:
            return False
        
        self.vocab.add(word)
        self.word2id[word] = len(self.word2id)
        self.id2word[self.word2id[word]] = word

        # add theta as its prob=0.0
        self.theta = np.append(self.theta, p)

        return True

    def piece_to_id(self, piece):
        if piece not in self.vocab:
            piece = self.unkToken
        return self.word2id[piece]

    def id_to_piece(self, i):
        return self.id2word[i]

    def setThetaFromTokenizedData(self, segData, smoothingValue=1.0):
        # segData: ['this', 'is', 'a', 'pen', '.']
        if not hasattr(self, 'vocab'):
            wordList = list({w for line in segData for w in line})
            self.setVocabFromWordList(wordList)

        unk = '<unk>' if '<unk>' in self.vocab else '[UNK]'

        self.theta = [smoothingValue]*len(self.vocab)
        for line in segData:
            for w in line:
                i = self.word2id[w] if w in self.word2id else self.word2id[unk]
                self.theta[i] += 1
        self.theta = np.array(self.theta)
        self.theta = self.theta / self.theta.sum()

    def setVocabFromWord2Id(self, w2i):
        # dict = {w:p, w:p, ...}
        self.vocab = set(w2i.keys())
        self.word2id = w2i
        self.id2word = {i:w for w,i in self.word2id.items()}
        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        self.theta = np.array([1 for i in range(len(self.word2id))])
        self.theta = self.theta/self.theta.sum()

        self.maxLength = max([len(w) for w in self.vocab])

    def setVocabFromWordList(self, wordList):
        dummyUnigramDict = {w:1 for w in wordList}
        if '<unk>' not in dummyUnigramDict and '[UNK]' not in dummyUnigramDict:
            dummyUnigramDict['<unk>'] = 1
        self.setVocabFromUnigramDict(dummyUnigramDict)

    def setVocabFromUnigramDict(self, unigramDict, word2id=None, char2id=None):
        # dict = {w:p, w:p, ...}
        
        # word
        self.vocab = set(unigramDict.keys())
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = {w:i for i,w in enumerate(sorted(list(self.vocab)))}
        self.id2word = {i:w for w,i in self.word2id.items()}

        # char
        charVocab = set(w for w in self.vocab if len(w)==1)
        if char2id:
            self.char2id = char2id
        else:
            self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        self.theta = np.array([unigramDict[self.id2word[i]] for i in range(len(self.word2id))])
        self.theta = self.theta/self.theta.sum()

        self.maxLength = max([len(w) for w in self.vocab])

    def setVocabFromBERTVocab(self, vocab):
        self.vocab = set(vocab.keys())
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(vocab):
            assert i==vocab[w], 'index mismatch'
            self.word2id[w] = i
            self.id2word[i] = w
        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        size = len(self.vocab)
        p = 1/size
        self.theta = np.full(size, p)            

        self.maxLength = max([len(w) for w in vocab])

    def setThetaFromSentencePiece(self, path):
        sp = self.__loadSentencePieceModel(path)
        spPiece2Score = {sp.id_to_piece(i):sp.get_score(i) for i in range(sp.get_piece_size())}
        # vocabs which is not included in loaded sentencepiece
        vocabDiff = self.vocab-set(spPiece2Score.keys()) 
        print('vocab diff')
        print(vocabDiff)

        print('set vocab diff as small value, exp(-30)') 
        for w in vocabDiff:
            spPiece2Score[w] = -30
        
        print('set scores of special tokens as exp(-30)')
        for w in spPiece2Score:
            if spPiece2Score[w]==0.0:
                # scores of special tokens (<s>, </s>, <unk>) are 0.0
                spPiece2Score[w] = -30

        self.theta = [spPiece2Score[self.id2word[i]] for i in range(len(self.vocab))]
        self.theta = np.exp(self.theta)
        self.theta = self.theta / self.theta.sum()

    def __loadSentencePieceModel(self, path):
        import sentencepiece as sp
        spp = sp.SentencePieceProcessor()
        if spp.load(path):
            print('>>> LOAD SENTENCEPIECE MODEL')
        else:
            print('>>> FAIL TO LOAD SENTENCE MODEL, EXIT')
            exit()
        return spp

    def convertWords2Ids(self, words, unkIdx=None):
        if unkIdx is None: unkIdx = self.word2id[self.unkToken]
        ids = [self.word2id[word] if word in self.vocab else unkIdx for word in words]
        return ids
    
    def convertIds2Words(self, ids):
        words = [self.id2word[i] for i in ids]
        return words

    def makeLogProbTable(self, line, unkProb=1e-7, idTable=None, lam=1.0):
        if idTable is None:
            idTable = self.makeIdTable(line, paddingIdx=-1, unkCharIdx=self.word2id[self.unkToken])

        # calc theta
        if lam==1.0:
            theta = self.theta
        else:
            theta = self.theta ** lam
            theta = theta / theta.sum()

        I, J = idTable.shape
        probTable = np.zeros((I, J))
        for t in range(I):
            for l in range(min(t+1, J)):
                i = idTable[t,l]
                if i != -1:
                    probTable[t,l] = theta[i]
        logProbTable = np.log(probTable)
        return logProbTable

    def makeIdTable(self, line, paddingIdx=-1, unkCharIdx=None, vocab=None):
        if unkCharIdx is None: unkCharIdx = self.word2id[self.unkToken]
        # specify vocab if you want to limit the vocab for some reasons
        if vocab is None: vocab = self.vocab
       
        if not hasattr(self, 'wordHeadPrefix'):
            if hasattr(self, 'wordPiecePrefix'):
                # compatibility with a version using "wordPiecePrefix" that will be removed in future
                self.wordHeadPrefix = self.wordPiecePrefix
            else:
                self.wordHeadPrefix = None
        if not hasattr(self, 'wordInterPrefix'):
            self.wordInterPrefix = None

        if self.wordHeadPrefix is not None:
            line = self.wordHeadPrefix + line.replace(' ', self.wordHeadPrefix)

        #if (self.wordHeadPrefix is not None) or (self.wordInterPrefix is not None):
        if self.wordInterPrefix is not None:
            # if wordHeadPrefix is '', just skipping whitespace as word boundary.
            heads = {0}
            c = 0
            for a in line:
                if a == ' ':
                    heads.add(c)
                    continue
                c += 1
            # the size of table's column is gained by removing space when using word piece mode
            line = ''.join(line.split())
        else:
            heads = None

        idTable = np.full((len(line), self.maxLength), paddingIdx)
        if unkCharIdx is not None:
            idTable[:,0] = unkCharIdx

        for t in range(len(line)):
            for l in range(min(t+1, self.maxLength)):
                w = line[t-l:t+1]
                if heads:
                    if 0<sum([i in heads for i in range(t-l+1,t+1)]):
                        continue 
                    #if self.wordHeadPrefix and t-l in heads:
                    #    w = self.wordHeadPrefix + w
                    if self.wordInterPrefix and t-l not in heads:
                        w = self.wordInterPrefix + w

                if w in vocab:
                    #if self.wordHeadPrefix and 1<=len(set(range(t-l+1,t+1))&heads):
                    #    continue
                    idTable[t,l] = self.word2id[w]
        return idTable

    def getWordIdsInLine(self, line):
        wordIds = []
        for t in range(len(line)):
            for l in range(min(t+1, self.maxLength)):
                w = line[t-l:t+1]
                if w in self.vocab:
                    wordIds.append(self.word2id[w])
        return wordIds

    def proneVocab(self, thre=None):
        # TODO: ここにseedに準拠した枝刈りを書く

        if thre is None:
            # 1/2 * 1/sum(all 1-gram freq)
            thre = 1/2 * 1/self.unigramFreq
        print('prone thre:', thre)

        # escape unk
        self.theta[self.word2id[self.unkToken]] = thre

        # escape fixed seed
        if self.fixedSeeds is not None:
            for token in self.fixedSeeds:
                i = self.word2id[token]
                self.theta[i] = 1.0

        dropCount = 0
        for i in range(self.theta.shape[0]):
            if self.theta[i] < thre:
                if len(self.id2word[i])==1:
                    self.theta[i] = thre
                else:
                    self.theta[i] = 0
                    dropCount += 1
        print('drop %d tokens'%dropCount)

        # smooth theta
        # self.theta = self.theta/sum(self.theta)

        print('-> unigram distribution was broken, NEED to reestimate probs.')
        # TODO: これをやるとunigram確率が壊れるので，再推定する必要がある

    def shrinkVocab_old(self, size, fixedSeeds=None):
        # get a size, then shrink self.vocab into the size
        print('>>> SHRINK VOCAB')
        size -= len(self.char2id)
        print('char size:', len(self.char2id))
        print('word size:', size)

        sortedTheta = sorted(self.theta, reverse=True)
        thre = sortedTheta[size]
        if thre==sortedTheta[size+1]:
            while thre==sortedTheta[size]:
                size -= 1
            thre = sortedTheta[size]
        print('actrual word size:', size)
        self.proneVocab(thre)

    def shrinkVocab(self, size, hardLimit=False, keepChar=False):
        print('>>> SHRINK VOCAB')
        # add small value for all tokens to distinguish them from discarded tokens (assigned with 0.0)
        self.theta += 1e-30

        # escape unk
        escaped = []
        words = []
        dropped = []
        ties = []
        for word in self.vocab:
            if word==self.unkToken or (self.fixedSeeds is not None and (word in self.fixedSeeds)):
                escaped.append(self.word2id[word])
            elif keepChar and len(word)==1:
                escaped.append(self.word2id[word])
            else:
                words.append(self.word2id[word])

        wordSize = size - len(escaped)

        wordTheta = self.theta[words]
        sortedWordTheta = sorted(wordTheta, reverse=True)
        if wordSize==0:
            self.theta[words] = 0
            print('drop %d tokens'%len(words))
        else:
            thre = sortedWordTheta[wordSize-1]
            print('proning threshold:', thre)
            for word in words:
                score = self.theta[word]
                if score < thre:
                    dropped.append(word)
                elif score == thre:
                    ties.append(word)

            if hardLimit:
                sampledTiesSize = (len(self.vocab) - len(dropped)) - size
                random.shuffle(ties)
                dropped += ties[:sampledTiesSize]
            
            self.theta[dropped] = 0

            print('drop %d tokens'%len(dropped))

        print('-> unigram distribution was broken, NEED to reestimate probs.')

    def reIndex(self):
        nonZeroIdx = np.where(0<self.theta)[0]
        self.theta = self.theta[nonZeroIdx]
        neow2i = {}
        neoi2w = {}
        for i in nonZeroIdx:
            w = self.id2word[i]
            neow2i[w] = len(neow2i)
            neoi2w[neow2i[w]] = w
        self.word2id = neow2i
        self.id2word = neoi2w
        self.vocab = set(self.word2id.keys())

    def makeLengthWiseIdDict(self):
        lengthIdDict = {l+1:set() for l in range(self.maxLength)}
        for w,i in self.word2id.items():
            lengthIdDict[len(w)].add(i)
        return lengthIdDict

    def makeLengthWiseManyHotVector(self):
        lengthIdDict = self.makeLengthWiseIdDict()
        manyhot = [[1 if i in lengthIdDict[l+1] else 0 for i in range(len(self.id2word))] for l in range(self.maxLength)]
        return np.array(manyhot)

    def makeCharIdOfVocab(self, paddingIdx=-1):
        ids = [[self.char2id[self.id2word[i][j]] if j<len(self.id2word[i]) else paddingIdx 
                for j in range(self.maxLength)] 
               for i in range(len(self.word2id))]
        ids = np.array(ids)
        return ids

    def getCharIdSet(self):
        # return ids of chars(=single length tokens) as its set.
        if not hasattr(self, 'charIdSet'):
            self.charIdSet = {i for i,w in self.id2word.items() if len(w)==1}
        return self.charIdSet

    def save(self, path):
        pickle.dump(self.__dict__, open(path, 'wb'))

    def load(self, path):
        self.__dict__ = pickle.load(open(path, 'rb'))

    def saveVocabScore(self, path):
        with open(path, 'w') as f:
            for i in range(len(self.vocab)):
                word = self.id2word[i]
                score = np.log2(self.theta[i])
                f.write('%s\t%f\n'%(word, score))


    def loadBERTTokenizer(self, name):
        import transformers
        print('>>> LOAD BERT TOKENIZER NAMED %s'%name)
        bt = transformers.AutoTokenizer.from_pretrained(name)
        print('>>> DONE')
    
        self.wordInterPrefix = '##'
        self.unkToken = '[UNK]'    

        self.setVocabFromBERTVocab(bt.vocab)

    def loadSentencePieceModel(self, path):
        spp = self.__loadSentencePieceModel(path)
        self.wordHeadPrefix = '▁'
        
        self.word2id = {}
        self.id2word = {}
        maxLength = 0
        theta = []

        # 0,1,2: unk, bos, eos
        size = spp.get_piece_size()
        for i in range(size):
            w = spp.id_to_piece(i)
            s = spp.get_score(i)
            
            #if w==self.unkToken:
            if s==0.0:
                print('set %s as small value (-30)'%w)
                # set p(unk) as small value
                # set specialtoken as small value
                s = -30
            
            self.word2id[w] = i
            self.id2word[i] = w
            theta.append(s)
        
            length = len(w)
            maxLength = max(maxLength, length)

        self.maxLength = maxLength
        
        theta = np.exp(np.array(theta))
        theta = theta / np.sum(theta)

        self.theta = theta
        self.vocab = set(self.word2id.keys())
        self.replaceSpaceMode = True

