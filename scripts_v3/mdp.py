import numpy as np
from scipy.special import logsumexp
#from . import util
import util
import random
from multiprocessing import Pool
import multiprocessing as multi
from time import time

import numba as nb
from numba import jit, f8, i8, u1, b1
from numba.typed import List

from heapq import heappop, heappush, heapify

minf = float('-inf')

@jit(nb.types.Tuple((f8[:,:], f8[:,:]))(f8[:,:]), nopython=True, fastmath=True)
def calcAlpha(logProbTable):
    T, L = logProbTable.shape

    ### A
    #'''
    alpha = np.zeros((T, L))
    sumAlpha = np.zeros((T,1))

    for t in range(T):
        for l in range(L):
            pi = t-l-1
            prev = sumAlpha[pi,0] if 0 <= pi else 0
            alpha[t,l] = logProbTable[t,l] + prev
        sumAlpha[t,0] = util._logsumexp(alpha[t])

    return alpha, sumAlpha

def calcBeta(logProbTable):
    T, L = logProbTable.shape
    beta = np.full((T, L), float('-inf'))
    sumBeta = np.zeros(T+1)

    for t in range(T)[::-1]:
        for l in range(L):
            if T < t+l+1:
                break
            prev = sumBeta[t+l+1]
            beta[t,l] = logProbTable[t+l,l] + prev
        sumBeta[t] = util._logsumexp(beta[t])

    return sumBeta

def calcGamma(logProbTable, alpha, sumAlpha):
    gamma = np.zeros(alpha.shape)
    sumGamma = np.zeros(alpha.shape[0])

    for t in range(gamma.shape[0]):
        for l in range(gamma.shape[1]):
            if t-l-1 < 0:
                prev = 0
            else:
                prev = sumGamma[t-l-1]
           
            gamma[t,l] = prev * np.exp(alpha[t,l] - sumAlpha[t])

        sumGamma[t] = 1 + sum(gamma[t])

    return np.log(sumGamma)

def calcPosterior(alpha, sumBeta, sumGamma):
    ab = alpha + sumBeta[1:].reshape(sumBeta.shape[0]-1, 1)
    bg = sumBeta[0]+sumGamma[-1]
    return np.exp(ab-bg)

@jit(nb.types.Tuple((f8[:,:], i8, i8))(f8[:,:]), nopython=True)
def forwardFiltering(logProbTable):
    T, L = logProbTable.shape
    alpha, sumAlpha = calcAlpha(logProbTable)
    dists = alpha - sumAlpha
    dists = np.exp(dists)
    return dists, T, L

@jit(i8(i8[:], f8[:]), nopython=True)
def rand_choice_nb(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def backwardSampling(dists, T, L):
    ls = [0]*T
    p = T-1
    c = 0
    while 0<=p:
        lSize = min(L, p+1)
        dist = dists[p, :lSize]
        l = rand_choice_nb(np.arange(1,lSize+1), dist)       
        ls[c] = l
        c += 1
        p -= l
    ls = ls[c-1::-1]
    return ls

@jit(nopython=True)
def ffbs(logProbTable, n=1):
    dists, T, L = forwardFiltering(logProbTable)
    lss = [backwardSampling(dists, T, L) for m in range(n)]
    return lss

@jit(nb.types.Tuple((f8[:], i8[:]))(f8[:,:]), nopython=True)
def viterbiForward(logProbTable):
    size = logProbTable.shape[0]
    maxScores = np.zeros(size, dtype=np.float64)
    maxLs = np.zeros(size, dtype=np.int64)

    # forward
    for t in range(size):
        lpList = np.zeros(logProbTable.shape[1], dtype=np.float64)
        for l in range(logProbTable.shape[1]):
            lpList[l] = logProbTable[t,l] + (maxScores[t-l-1] if 0<=t-l-1 else 0)
        maxScores[t] = max(lpList)
        maxLs[t] = lpList.argmax()
    return maxScores, maxLs

def viterbi(logProbTable):
    maxScores, maxLs = viterbiForward(logProbTable)
    
    # backward
    ls = []
    t = len(maxLs)-1
    while 0<=t:
        l = maxLs[t] + 1
        ls.append(l)
        t -= l
    ls = ls[::-1]

    return ls

def tokenizeByLength(line, ls):
    segs = ['']*len(ls)
    pointer = 0
    for i,l in enumerate(ls):
        segs[i] = line[pointer:pointer+l]
        pointer += l
    return segs

def tokenize(line, logProbTable, sampling):
    ls = ffbs(logProbTable, 1)[0] if sampling else viterbi(logProbTable)
    segs = tokenizeByLength(line, ls)
    return segs    

def samplingSegmentation(line, logProbTable):
    return tokenize(line, logProbTable, sampling=True)

def samplingIdSegmentation(idTable, logProbTable):
    ls = ffbs(logProbTable, 1)[0]
    ids = getIds(idTable, ls)
    return ids

def wrapSamplingIdSegmentation(xs):
    return samplingIdSegmentation(*xs)

def samplingIdSegmentationMultiProc(idTables, logProbTables):
    p = Pool(multi.cpu_count())
    segIds = p.map(wrapSamplingIdSegmentation, zip(idTables, logProbTables))
    p.close()
    return segIds

def viterbiSegmentation(line, logProbTable):
    return tokenize(line, logProbTable, sampling=False)

def viterbiIdSegmentation(idTable, logProbTable):
    ls = viterbi(logProbTable)
    c = 0
    ids = []
    for l in ls:
        c += l
        ids.append(idTable[c-1, l-1])
    return ids

def nbestSegmentation(line, logProbTable, n, mode='astar'):
    if mode=='astar':
        # forward
        maxScores, _ = viterbiForward(logProbTable)
        # backward
        segs = [tokenizeByLength(line, ls) for ls, _ in nbestAstarBackward(maxScores, logProbTable, n=n)]
        return segs
    elif mode=='point':
        # viterbi
        bestSeg = viterbi(logProbTable)
        # point estimation
        lss = nbestPointEstimation(bestSeg, logProbTable, n=n)
    else:
        print('mode should be {astar, point}')
        exit()

    segs =  [tokenizeByLength(line, ls) for ls in lss]

    return segs

def nbestIdSegmentation(idTable, logProbTable, n, mode='astar'):
    if mode=='astar':
        # forward
        maxScores, _ = viterbiForward(logProbTable)
        # backward
        lss = nbestAstarBackward(maxScores, logProbTable, n=n)
    elif mode=='point':
        # viterbi
        bestSeg = viterbi(logProbTable)
        # point estimation
        lss = nbestPointEstimation(bestSeg, logProbTable, n=n)
    else:
        print('mode should be {aster, point}')
        exit()

    idss = [getIds(idTable, ls[0]) for ls in lss]

    return idss

def mSampleFromNBestSegmentation(line, logProbTable, m, n, mode='astar', lam=1.0):
    if mode!='astar':
        print('mode %s is not implemented'%mode)
        exit()
    assert m<=n, 'mSamplingFromNbestSegmentation requires: m <= n'

    # forward
    maxScores, _ = viterbiForward(logProbTable)

    # nbest backward
    ress = [(tokenizeByLength(line, ls), logP) for ls, logP in nbestAstarBackward(maxScores, logProbTable, n=n)]
    segs, logPs = zip(*ress)

    size = len(segs)
    if size <= m:
        return segs

    # m-sampling
    if lam!=1.0:
        dist = lam * np.array(logPs)
    dist = logPs - logsumexp(logPs)
    dist = np.exp(dist)

    if np.any(np.isnan(dist)):
        # debug
        print(dist)
        print(logProbTable)
        print(line)

    segIdx = np.random.choice(size, m, p=dist, replace=False)
    
    # sort segIdx: nbest dist is descending order
    # segIdx.sort()
    
    segs = [segs[si] for si in segIdx]
    return segs

def getIds(idTable, ls):
    c = 0
    ids = [0]*len(ls)
    for i, l in enumerate(ls):
        c += l
        ids[i] = idTable[c-1, l-1]
    return ids

def mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar', lam=1.0):
    lam = np.array(lam)

    if mode!='astar':
        print('mode %s is not implemented'%mode)
        exit()
    assert m<=n, 'mSamplingFromNbestSegmentation requires: m <= n'

    if m==1 and n==1:
        # viterbi
        return (viterbiIdSegmentation(idTable, logProbTable),)

    # forward
    maxScores, _ = viterbiForward(logProbTable)

    # nbest backward
    ress = [(getIds(idTable, ls), logP) for ls, logP in nbestAstarBackward(maxScores, logProbTable, n=n)]
    isegs, logPs = zip(*ress)

    size = len(isegs)
    if size <= m:
        return isegs

    # m-sampling
    if lam!=1.0:
        logPs = lam * logPs
    dist = logPs - logsumexp(logPs)
    dist = np.exp(dist)

    if np.any(np.isnan(dist)):
        # debug
        print(dist)
        print(logProbTable)
        print(line)

    segIdx = np.random.choice(size, m, p=dist, replace=False)
    
    # sort segIdx: nbest dist is descending order
    #segIdx.sort()

    isegs = tuple(isegs[si] for si in segIdx)
    selectedDist = dist[segIdx]
    
    return isegs

def nbestPointEstimation(bestSegLen, logProbTable, n):
    maxLength = logProbTable.shape[1]

    def vary(seg):
        variations = [seg[:k] 
                      + (1-seg[k],) 
                      + seg[k+1:] for k in range(len(seg))]
        return variations
    
    def calcSegScore(seg):
        score, p, l = 0, 0, 0
        for s in seg+(1,):
            if s==1:
                score += logProbTable[p,l]
                l = 0;  p += 1
            else:
                l += 1; p += 1
                if maxLength <= l:
                    return float('-inf')
        return score

    def seg2len(seg):
        ls = []
        l = 0
        for s in seg+(1,):
            l += 1
            if s==1:
                ls.append(l)
                l = 0 
        return ls

    bestSeg = tuple(s for l in bestSegLen for s in (0,)*(l-1)+(1,))[:-1]
    queue = {bestSeg: calcSegScore(bestSeg)}
    nbests = {}

    for _ in range(n):
        if not queue:
            break
        nbestSeg = sorted(queue.items(), key=lambda x:x[1], reverse=True)[0][0]
        if queue[nbestSeg]==float('-inf'): break
        nbests[nbestSeg] = queue[nbestSeg]
        del queue[nbestSeg]

        for seg in vary(nbestSeg):
            if seg not in queue and seg not in nbests:
                queue[seg] = calcSegScore(seg)

    return [seg2len(seg) for seg, score in sorted(nbests.items(), key=lambda x:x[1], reverse=True)]

def backtrace(ls):
    return tuple(i-j for i, j in zip(ls[1:], ls[:-1]))

def addNextNodes(queue, CACHE, prevIdx, prevScore, path, maxLength, logProbTable, viterbiScores):
    if prevIdx not in CACHE:
        prevIdxM1 = prevIdx-1
        startIdx = max(prevIdx-maxLength, 0)
        
        ids = range(startIdx,prevIdx)
        wordScores = logProbTable[prevIdxM1][prevIdxM1-startIdx::-1]
        nominf = [i for i, ws in enumerate(wordScores) if ws!=minf]

        CACHE[prevIdx] = [(wordScores[i]+viterbiScores[ids[i]],
                           wordScores[i],
                           (ids[i],)) for i in nominf]

    _ = [heappush(queue, 
                  ((prevScore + np)*-1, # nextPriority
                   prevScore + ws, #nextScore
                   i + path)
                 ) 
            for np, ws, i in CACHE[prevIdx]]
    return queue, CACHE

def nbestAstarBackward(viterbiScores, logProbTable, n):

    # add BOS
    viterbiScores = [0]+viterbiScores.tolist()
    
    maxLength = logProbTable.shape[1]
    logProbTable = logProbTable.tolist()
    CACHE = {}

    queue = [(0, 0, (len(viterbiScores)-1,))] # initialized with endnode. requrires: (priority, score, idx to trace+)
    m = 0

    while queue:
        # pop
        _, prevScore, path = heappop(queue)

        prevIdx = path[0]

        # BOS
        if prevIdx==0:
            yield backtrace(path), prevScore
            m += 1
            if n<=m: break
            continue
        
        queue, CACHE = addNextNodes(queue, CACHE, prevIdx, prevScore, path, maxLength, logProbTable, viterbiScores)

def wrapNbestSegmentation(xs):
    return nbestSegmentation(*xs)

def nbestSegmentationMultiProc(lines, logProbTables, n, mode='astar'):
    size = len(lines)
    if size <= 512:
        return [nbestSegmentation(line, logProbTable, n, mode)
                for line, logProbTable in zip(lines, logProbTables)]

    ns = [n] * size
    modes = [mode] * size
    
    p = Pool(multi.cpu_count())
    segs = p.map(wrapNbestSegmentation, zip(lines, logProbTables, ns, modes))
    p.close()
    
    return segs

def mSamplingFromNbestSegmentationMultiProc(lines, logProbTables, m, n, mode='astar'):
    size = len(lines)
    if size <= 512:
        return [mSampleFromNBestSegmentation(line, logProbTable, m, n, mode='astar')
                for line, logProbTable in zip(lines, logProbTables)]

    ns = [n] * size
    modes = [mode] * size
    
    p = Pool(multi.cpu_count())
    segs = p.map(wrapNbestSegmentation, zip(lines, logProbTables, ns, modes))
    p.close()
    
    return segs

def nSampleWithFFBS(line, logProbTable, n):
    lss = ffbs(logProbTable, n)
    segs = [tokenizeByLength(line, ls) for ls in lss]
    return segs

