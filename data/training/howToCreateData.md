# Tokenizer学習用データの作成方法
## Ver1, 2, 2.1, 2.2
### データ
LLMを学習するコーパスの一部を利用して作成します．
**ただし，現時点ではコーパスは公開されていないため，外部利用につきましてはコーパスの公開をお待ちください．**
- `nii-llm-corpora-ver1` の以下のデータから各1GBずつサンプリング（合計5GB程度）
    - en_pile/train.jsonl
    - en_wiki/train.jsonl
    - ja_wiki/train.jsonl
    - ja_cc/train_[0-9].jsonl
    - code_stack/train.jsonl

### 準備
1. jsonlをテキストに変換
```
$ python jsonl2text.py nii-llm-corpora-ver1/en_pile/data/train.jsonl > en_pile.text
$ python jsonl2text.py nii-llm-corpora-ver1/en_wiki/data/train.jsonl > en_wiki.text
$ python jsonl2text.py nii-llm-corpora-ver1/ja_wiki/data/train.jsonl > ja_wiki.text
$ python jsonl2text.py nii-llm-corpora-ver1/code_stack/data/train.jsonl > code_stack.text
$ for i in `seq 0 9`; do
$   python jsonl2text.py nii-llm-corpora-ver1/ja_cc/data/train_${i}.jsonl > ja_cc_${i}.text
$ done
```

2. データを抽出
```
$ python extractLines.py en_pile.text ids/en_pile_1GB.ids > en_pile_1GB.text
$ python extractLines.py en_wiki.text ids/en_wiki_1GB.ids > en_wiki_1GB.text
$ python extractLines.py ja_wiki.text ids/ja_wiki_1GB.ids > ja_wiki_1GB.text
$ python extractLines.py code_stack.text ids/code_stack_1GB.ids > code_stack_1GB.text
$ for i in `seq 0 9`; do
$   python extractLines.py ja_cc_${i}.text ids/ja_cc_1GB_${i}.ids >> ja_cc_1GB.text
$ done
```