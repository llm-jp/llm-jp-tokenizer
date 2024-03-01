# Quick Usage
## 用語
- シード (seed): 学習後の語彙に必ず含まれるべきトークンのリスト
- 初期シード（initialSeed）: 学習に使うトークン候補のリスト．学習中にスコアが低いものから削除される．
    - 初期シードを指定しない場合，データから可能なn-gramを自動で初期シードとして設定

## 基本的な使い方（シード，初期シードともに設定しない場合）
- `--data`: 初期シードの作成，学習に使用するテキストデータ．
- `--maxEpoch 2`: inner-loopの学習ステップ数
- `--maxLength`: 初期シードを作成するときのトークンの最大長
- `--minFreq`: 初期シードを作成するときの最小出現頻度（`--data`でカウント）
- `--resultDir ../models`: 学習した結果の出力先
- `--outputSuffix model`: `--resultDir`に保存するフォルダにつける接尾辞
- `--trainMode EMMT`: ユニグラムスコア推定のinner-loopに，EMアルゴリズム（並列ver）を使用
- `--vocabSize 300000`: 最終的な語彙の規模．
- `--shriknRatio 0.5`: 語彙の枝刈りステップごとに削除する語彙の規模の割合．0.5なら，ステップごとに5割を削除
- `--spaceSplit`: 入力テキストをスペース区切りで分割し，先頭にheadPrefixを付与する．

```
$ python train_spstyle.py \
    --data path/to/train.text \
    --maxEpoch 2 \
    --maxLength 16 \
    --minFreq 30 \
    --resultDir ../models/ \
    --outputSuffix model \
    --trainMode EMMT \
    --vocabSize 300000 \
    --shrinkRatio 0.5 \
    --spaceSplit
```

- 出力ファイル
    - この場合，`../models/{timestamp}-{randomID}_{outputSuffix}`以下に学習済みの語彙などが保存される
        - `config.yml`: 学習の設定ファイル
        - `lm.pickle`: multigramlmで読み込めるファイル（今回は使わない）
        - `lm.pickle.vocab`: トークンとスコアを並べたTSVファイル．`vocab2model.py`で読み込める．

## シードを指定する場合
- `--fixedSeedsPath seed1.txt seed2.txt`: シードとして使用するトークンリスト（改行区切りのテキストファイル）
    - シードに指定したトークンは，必ず完成した語彙に含まれる

```
$ python train_spstyle.py \
    --data path/to/train.text \
    --maxEpoch 2 \
    --maxLength 16 \
    --minFreq 30 \
    --resultDir ../models/ \
    --outputSuffix model \
    --trainMode EMMT \
    --vocabSize 300000 \
    --shrinkRatio 0.5 \
    --fixedSeedsPath seed1.txt seed2.txt \
    --spaceSplit
```

## 初期シードを指定する場合
- `--initialSeedsPath initSeed1.txt initSeed2.txt`: 初期シードとして使用するトークンリスト（改行区切りのテキストファイル）
    - 初期シードに指定したトークンリストがトークンの候補としてそのまま使用される
    - このオプションを使用した場合，`--data`で読み込んだテキストから自動でn-gramを収集する処理はスキップ

```
$ python train_spstyle.py \
    --data path/to/train.text \
    --maxEpoch 2 \
    --maxLength 16 \
    --minFreq 30 \
    --resultDir ../models/ \
    --outputSuffix model \
    --trainMode EMMT \
    --vocabSize 300000 \
    --shrinkRatio 0.5 \
    --initialSeedsPath initSeed1.txt initSeed2.txt \
    --spaceSplit
```

## 引数リスト
```
$ python train_spstyle.py -h
usage: train_spstyle.py [-h] -d DATA [-td TESTDATA] [-me MAXEPOCH]
                        [-ml MAXLENGTH] [-mf MINFREQ] [-os OUTPUTSUFFIX]
                        [-tm {viterbi,viterbiStepWise,viterbiBatch,EM,EMMT}]
                        [-hp HEADPREFIX] [-vs VOCABSIZE] [-rd RESULTDIR]
                        [-sr SHRINKRATIO]
                        [--fixedSeedsPath FIXEDSEEDSPATH [FIXEDSEEDSPATH ...]]
                        [--initialSeedsPath INITIALSEEDSPATH [INITIALSEEDSPATH ...]]
                        [--spaceSplit] [--withInference] [--keepChar]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  text data for training
  -td TESTDATA, --testData TESTDATA
                        text data for checking log-prob as test. if not
                        specified, use training set
  -me MAXEPOCH, --maxEpoch MAXEPOCH
                        max training epoch of each training step (default: 10)
  -ml MAXLENGTH, --maxLength MAXLENGTH
                        maximum length of word (default: 5)
  -mf MINFREQ, --minFreq MINFREQ
                        minimum frequency of word (default: 3)
  -os OUTPUTSUFFIX, --outputSuffix OUTPUTSUFFIX
                        output is dumped as [timestamp]_suffix.pickle if
                        suffix is given otherwise [timestamp].pickle
  -tm {viterbi,viterbiStepWise,viterbiBatch,EM,EMMT}, --trainMode {viterbi,viterbiStepWise,viterbiBatch,EM,EMMT}
                        method to train multigram language model (default: EM)
  -hp HEADPREFIX, --headPrefix HEADPREFIX
                        special prefix for subwords indicating head of word
                        (default: ▁)
  -vs VOCABSIZE, --vocabSize VOCABSIZE
                        vocabulary size (default 32000)
  -rd RESULTDIR, --resultDir RESULTDIR
                        dir to output (default: ./)
  -sr SHRINKRATIO, --shrinkRatio SHRINKRATIO
                        ratio for shrinking in each proning step (default:
                        0.8, which means 20 perc of vocab are discarded in
                        each step)
  --fixedSeedsPath FIXEDSEEDSPATH [FIXEDSEEDSPATH ...]
                        path/to/fixedSeedsList.txt, which is splited by \n.
                        multiple paths can be specified.
  --initialSeedsPath INITIALSEEDSPATH [INITIALSEEDSPATH ...], -is INITIALSEEDSPATH [INITIALSEEDSPATH ...]
                        path/to/initialSeed.txt, which is splited by \n.
                        multiple paths can be specified.
  --spaceSplit          split text by whitespaces if true
  --withInference       tokenize the training data after training
  --keepChar            do not discard character when shrinking vocabulary if
                        true
```

## シードの作成方法
### LLaMA2
- vocabの表層をそのままdump

### Sudachi
- Aモードの辞書をSudachi-coreから抽出
    - DictioinaryPrinterなどを使って，CSV形式の辞書words.csvを作成
- ただし，以下のスクリプトを使って，平仮名・片仮名・漢字のいずれかが含まれる単語に限定した

```
$ python filterSudachiDict.py --dictionary words.csv --output path/to/sudachi_A.vocab
```


## 初期シードを作成する場合
### テキストデータから作成

```
$ python seedCreater.py \
    --data path/to/text \
    --maxLength 16 \
    --minFreq 30 \
    --output path/to/output \
    --spaceSplit
```

### Sudachi辞書から初期シードを作る場合

```
$ python makeFreqDict.py --vocab path/to/sudachi_A.vocab --text path/to/text --output freqDict.json
# textでの頻出上位20,000件を使用する場合
$ python freqDict2vocab.py --dicts freqDict.json --numTokens 20,000 --text path/to/text --output initSeedSudachi_A.vocab
```