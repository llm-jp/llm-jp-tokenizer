#from . import lm
#from . import mdp as dp
#from . import util
import lm
import mdp as dp
import util
from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import os
from time import time

EPS = 1e-30

def createIdTableProc(mlm, lines, unkid):
    idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in lines]
    return idTables

def EMProc(mlm, lines, unkid):
    idTables = createIdTableProc(mlm, lines, unkid)
    iterTheta = np.zeros(mlm.theta.shape)
    for line, idTable in zip(lines, idTables):
        logProbTable = mlm.makeLogProbTable(line, idTable=idTable)

        # dp
        alpha, sumAlpha = dp.calcAlpha(logProbTable)
        sumBeta = dp.calcBeta(logProbTable)
        sumGamma = dp.calcGamma(logProbTable, alpha, sumAlpha)
        
        # posterior
        posterior = dp.calcPosterior(alpha, sumBeta, sumGamma)
        
        # update
        idx = np.where(idTable!=-1)
        
        iterTheta[idTable[idx]] += posterior[idx]
    return iterTheta

def EMTrainMultiThread(mlm, data, maxIter, numThreads=-1, proning=True):
    from concurrent.futures import ProcessPoolExecutor
    import random

    if numThreads==-1:
        numThreads = os.cpu_count()-2 # とりあえず-2

    # thread sizeにデータをスプリットする

    ### PREPARATION FOR EM TRAINING WITH MULTI THREADING ###
    print('MULTI-THREAD: N =', numThreads)
    MAX_SAMPLE_SIZE = 10000
    idx = list(range(len(data)))
    #random.shuffle(idx)          # 長さをある程度揃えるためにシャッフルする？→しない．これをやるとIDTABLEを作るときにバグるので

    if len(data) // numThreads < MAX_SAMPLE_SIZE:
        idBatches = [idx[i::numThreads] for i in range(numThreads)]
    else:
        # 各スレッドに投げるデータが100万件以上なら，100万件ごとに区切る
        idBatches = [idx[i:i+MAX_SAMPLE_SIZE] for i in range(0, len(data), MAX_SAMPLE_SIZE)]
    print('EACH BATCH SIZE:', len(idBatches))

    ### Create ID Tables ###
    unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
    '''
    print('CREATE ID TABLES...') # ここも並列処理にしないといけない

    with tqdm(total=len(idBatches)) as progress:
        with ProcessPoolExecutor(max_workers=numThreads) as executor:
            futures = []
            for batch in idBatches:
                lines = [data[i] for i in batch]
                future = executor.submit(createIdTableProc, mlm, lines, unkid)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            result = [f.result() for f in futures]
    idTables = []
    for r in result:
        idTables += r
    print('DONE')
    '''

    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        iterTheta = np.zeros(mlm.theta.shape)

        with tqdm(total=len(idBatches)) as progress:
            with ProcessPoolExecutor(max_workers=numThreads) as executor:
                futures = []
                for batch in idBatches:
                    ls = [data[i] for i in batch]
                    #its = [idTables[i] for i in batch]
                    #future = executor.submit(EMProc, mlm, ls, its)
                    future = executor.submit(EMProc, mlm, ls, unkid)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                result = [f.result() for f in futures]
                
        for r in result:
            iterTheta += r

        # re-normalize
        iterTheta = iterTheta + EPS
        iterTheta = iterTheta / sum(iterTheta)

        # update
        mlm.theta = iterTheta

        # proning
        if proning: mlm.proneVocab()
        
        #tmpSegs = [mlm.id2word[i] for i in dp.viterbiIdSegmentation(idTables[0],
        #                                         mlm.makeLogProbTable(data[0], idTable=idTables[0]))]
        #print(' '.join(tmpSegs))

    return mlm    

def EMTrain(mlm, data, maxIter=10, proning=True):
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        iterTheta = np.zeros(mlm.theta.shape)
        for j,line in enumerate(tqdm(data)):
            if len(line)==0: continue

            if it==0:
                unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
                idTable = mlm.makeIdTable(line, unkCharIdx=unkid)
                idTables.append(idTable)
            else:
                idTable = idTables[j]
            logProbTable = mlm.makeLogProbTable(line, idTable=idTable)

            # dp
            alpha, sumAlpha = dp.calcAlpha(logProbTable)
            sumBeta = dp.calcBeta(logProbTable)
            sumGamma = dp.calcGamma(logProbTable, alpha, sumAlpha)
            
            # posterior
            posterior = dp.calcPosterior(alpha, sumBeta, sumGamma)
            
            # update
            idx = np.where(idTable!=-1)
            iterTheta[idTable[idx]] += posterior[idx]

        # re-normalize
        iterTheta = iterTheta + EPS
        iterTheta = iterTheta / sum(iterTheta)

        # update
        mlm.theta = iterTheta

        # proning
        if proning: mlm.proneVocab()
        
        tmpSegs = [mlm.id2word[i] for i in dp.viterbiIdSegmentation(idTables[0],
                                                 mlm.makeLogProbTable(data[0], idTable=idTables[0]))]
        print(' '.join(tmpSegs))

    return mlm

def viterbiTrainBatch(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERBI BATCH %d EPOCH TRAINING'%(maxIter))
    batchSize = 256
    shuffle = True
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        iterTheta = np.zeros(mlm.theta.shape)
        
        if it==0:
            print('BUILD IDTABLES')
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in tqdm(data)]

        for b in tqdm(range(0, len(data), batchSize)):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            

            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]
            tables = [idTables[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [i
                       for line, idTable in zip(lines, idTables)
                       for i in dp.viterbiIdSegmentation(idTable,
                                                         mlm.makeLogProbTable(line, idTable=idTable))]

            # re-estimate
            batchTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                batchTheta[k] = v
            batchTheta = batchTheta / tmpVocabSize

            iterTheta += batchTheta

        # re-normalize
        iterTheta = iterTheta + EPS
        iterTheta = iterTheta / sum(iterTheta)

        # update
        mlm.theta = iterTheta

        # proning
        if proning: mlm.proneVocab()
        
    return mlm

def viterbiTrainStepWise(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERBI STEPWISE %d EPOCH TRAINING'%(maxIter))

    decay = 0.8
    batchSize = 256
    shuffle = True

    step = 0

    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        if it==0:
            print('BUILD IDTABLES')
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in tqdm(data)]

        for b in tqdm(range(0, len(data), batchSize)):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            
            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]
            tables = [idTables[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [i
                       for line, idTable in zip(lines, idTables)
                       for i in dp.viterbiIdSegmentation(idTable,
                                                         mlm.makeLogProbTable(line, idTable=idTable))]

            # re-estimate
            eta = (step+2)**(-decay)
            step += 1

            currentTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                currentTheta[k] = v
            currentTheta = currentTheta / tmpVocabSize

            # update
            mlm.theta = (1-eta)*mlm.theta + eta*currentTheta

        # proning
        if proning: mlm.proneVocab()
        
    return mlm

def viterbiTrain(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERBI %d EPOCH TRAINING'%(maxIter))

    prevLH = 0
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))

        if it==0:
            print('BUILD IDTABLES')
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in tqdm(data)]

        # viterbi
        tmpSegs = [i
                   for line, idTable in zip(tqdm(data), idTables)
                   for i in dp.viterbiIdSegmentation(idTable,
                                                     mlm.makeLogProbTable(line, idTable=idTable))]

        # calc loglikelihood
        loglikelihood = np.log([mlm.theta[i] for i in tmpSegs]).sum()
        print('current log-likelihood:', loglikelihood/len(tmpSegs))

        # re-estimate
        tmpVocabSize = len(tmpSegs)
        tmpUnigramCount = Counter(tmpSegs)
        currentTheta = np.zeros(mlm.theta.shape)
        for k,v in tmpUnigramCount.items():
            currentTheta[k] = v
        currentTheta = currentTheta / tmpVocabSize

        # re-normalize
        mlm.theta = currentTheta

        # proning
        if proning: mlm.proneVocab()

        print(' '.join([mlm.id2word[i] for i in tmpSegs[:100]]))        

    return mlm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='text data for training')
    parser.add_argument('-td', 
                        '--testData', 
                        default=None,
                        type=str,
                        help='text data for checking log-prob as test. if not specified, use training set')
    parser.add_argument('-me', 
                        '--maxEpoch', 
                        default=10, 
                        type=int,
                        help='max training epoch (default: 10)')
    parser.add_argument('-ml', 
                        '--maxLength', 
                        default=5, 
                        type=int,
                        help='maximum length of word (default: 5)')
    parser.add_argument('-mf', 
                        '--minFreq', 
                        default=50, 
                        type=int,
                        help='minimum frequency of word (default: 50)')
    parser.add_argument('-os',
                        '--outputSuffix',
                        default='',
                        type=str,
                        help='output is dumped as [timestamp]_suffix.pickle if suffix is given otherwise [timestamp].pickle')
    parser.add_argument('-tm',
                        '--trainMode',
                        default='EM',
                        choices=['viterbi',
                                 'viterbiStepWise',
                                 'viterbiBatch',
                                 'EM',
                                 'EMMT'
                                 ],
                        help='method to train multigram language model (default: EM)')
    parser.add_argument('-rd',
                        '--resultDir',
                        default='./',
                        help='dir to output (default: ./)')
    args = parser.parse_args()

    # make results dir
    if not os.path.isdir(args.resultDir):
        os.makedirs(args.resultDir)
        print('>>> CREATE RESULTS DIR')

    # set time stamp
    timeStamp = util.getTimeStamp()
    dirName = timeStamp + ('_' + args.outputSuffix if args.outputSuffix else '')
    os.mkdir(os.path.join(args.resultDir, dirName))
    
    # dump config
    setattr(args, 'dirName', dirName)
    open(os.path.join(args.resultDir, dirName, 'config.yaml'), 'w').write(yaml.dump(vars(args)))

    # load data
    data = [line.strip() for line in open(args.data)]
    print('>>> LOAD DATA')

    # training
    print('>>> START TRAINING')
    mlm = lm.MultigramLM(maxLength=args.maxLength, minFreq=args.minFreq, data=data)

    if args.trainMode=='EM':
        mlm = EMTrain(mlm, data, args.maxEpoch)
    elif args.trainMode=='EMMT':
        mlm = EMTrainMultiThread(mlm, data, args.maxEpoch, 10)
    elif args.trainMode=='viterbi':
        mlm = viterbiTrain(mlm, data, args.maxEpoch)
    elif args.trainMode=='viterbiStepWise':
        mlm = viterbiTrainStepWise(mlm, data, args.maxEpoch)
    elif args.trainMode=='viterbiBatch':
        mlm = viterbiTrainBatch(mlm, data, args.maxEpoch)
    
    mlm.reIndex() # discard tokens whose probability is 0.0
    print('>>> FINISH TRAINING')

    # inference
    if args.testData is not None:
        print('>>> INFERENCE ON TEST DATA')
        data = [line.strip() for line in open(args.testData)]
    else:
        print('>>> INFERENCE ON TRAIN DATA')
        
    segData = [dp.viterbiSegmentation(line, mlm.makeLogProbTable(line)) for line in data]
    loglikelihood = np.log([mlm.theta[mlm.word2id[seg]] for segLine in segData for seg in segLine]).sum()
    print('log-likelihood:', loglikelihood/sum([len(segLine) for segLine in segData]))

    # dump
    with open(os.path.join(args.resultDir, dirName, 'seg.txt'), 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = os.path.join(args.resultDir, dirName, 'lm.pickle')
    mlm.save(path)
    print('>>> DUMP RESULTS')

if __name__ == '__main__':
    main()
