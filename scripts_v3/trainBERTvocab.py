from . import train
from . import lm
from transformers import *
import argparse
from tqdm import tqdm
from . import util
import os
from . import mdp as dp
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', 
                        '--data',
                        required=True,
                        help='data for training lm')
    parser.add_argument('-p', 
                        '--pretrain',
                        required=True,
                        help='pretrained shortcut, such as bert-base-cased')
    parser.add_argument('-me', 
                        '--maxEpoch', 
                        default=10, 
                        type=int,
                        help='maximum training epoch')
    parser.add_argument('-os',
                        '--outputSuffix',
                        default='',
                        type=str,
                        help='output is dumped as timestamp_suffix.pickle if suffix is given otherwise timestamp.pickle')
    parser.add_argument('-tm',
                        '--trainMode',
                        default='EM',
                        choices=['viterbi',
                                 'viterbiStepWise',
                                 'viterbiBatch',
                                 'EM'],
                        help='method to train multigram language model')
    parser.add_argument('-rd',
                        '--resultDir',
                        default='./',
                        help='dir to output (default: ./)')
    args = parser.parse_args()

    # make results dir
    if not os.path.isdir(args.resultDir):
        os.makedirs(args.resultDir)
        print('>>> CREATE RESULT DIR')

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

    # load tokenizer
    mlm = lm.MultigramLM()
    mlm.loadBERTTokenizer(args.pretrain)
    print('>>> INITIALIZE MLM WITH BERT TOKENIZER')

    if args.trainMode=='EM':
        mlm = train.EMTrain(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbi':
        mlm = train.viterbiTrain(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbiStepWise':
        mlm = train.viterbiTrainStepWise(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbiBatch':
        mlm = train.viterbiTrainBatch(mlm, data, args.maxEpoch, proning=False)

    # reindexing is not required because word ids should be match b/w bert and mlm

    print('>>> FINISH TRAINING')

    idTables = [mlm.makeIdTable(line, unkCharIdx=mlm.word2id[mlm.unkToken]) for line in data]
    segData = [[mlm.id2word[i] 
                        for i in dp.viterbiIdSegmentation(idTable,
                                                           mlm.makeLogProbTable(line, idTable=idTable))]
                for line, idTable in zip(data, idTables)]
    loglikelihood = np.log([mlm.theta[mlm.word2id[seg]] for segLine in segData for seg in segLine]).sum()
    print('log-likelihood:', loglikelihood/sum([len(segLine) for segLine in segData]))

    # dump
    with open(os.path.join(args.resultDir, dirName, 'seg.txt'), 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = os.path.join(args.resultDir, dirName, 'lm.pickle')
    mlm.save(path)
    print('>>> DUMP RESULTS')

if __name__=='__main__':
    main()
