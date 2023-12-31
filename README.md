# LLM-jp Tokenizer
[LLM勉強会（LLM-jp）](https://llm-jp.nii.ac.jp/)で開発しているLLM用のトークナイザー関連をまとめたリポジトリです．

## What's New
### Release ver2.1
#### Hugging Face Fast Tokenizerで使う場合
- https://github.com/llm-jp/llm-jp-tokenizer/releases/tag/v2.1
- 必須ライブラリ
  - `transformers>=4.34.0`
  - `tokenizers>=0.14.0`
- 使い方
```Python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v1.0")
```

#### SentencePieceで使う場合
- 必須ライブラリ
  - sentencepiece>=0.1.99
  - protobuf<3.21.0
- 使い方
```Python
from sentencepiece import SentencePieceProcessor
sp = SentencePieceProcessor("models/ver2.1/code10k_en20k_ja30k.ver2.1.model")
```

## 特徴
[SentencePiece (Unigramモード)](https://github.com/google/sentencepiece)をベースに，以下のような工夫を盛り込んでいます．

1. 日本語テキストを人間にも分かりやすい単位に区切るための，辞書分割に準拠した語彙の構築
2. 言語やドメインごとに使用する語彙の規模を柔軟に設定するための，言語ごとのモデル構築と語彙のマージ，スコアの再推定

## モデル
各モデルは`/models`以下に配置してあります．
ここでは特に，Hugging Face Hubで公開中の1.3Bモデルおよび13Bモデルで使用されているver2.1(50k)と，
ABCI第2回大規模言語モデル構築支援プログラムで構築中の175Bモデルで使用されているver2.2(100k)について紹介します．


| モデル名 | 語彙の規模 (コード) | 語彙の規模 (英語) | 語彙の規模 (日本語) | 語彙の規模 (マージ後) |
| --- | --- | --- | --- | --- |
| code10k_en20k_ja30k.ver2.1(50k) | 10,000 | 20,000 | 30,000 | 50,572 |
| code20k_en40k_ja60k.ver2.2(100k) | 20,000 | 40,000 | 60,000 | 96,869 |

マージ後の語彙の規模より，code10k_en20k_ja30kのモデルを「だいたい50K」，code20k_en40k_ja60kを「だいたい100K」規模のトークナイザーモデルとして，主に利用しています(※1)．

Megatron-DeepSpeedでの事前学習では[code10k_en20k_ja30k.ver2.1.model](https://github.com/llm-jp/llm-jp-tokenizer/tree/main/models/ver2.1)や[code20k_en40k_ja60k.ver2.2.model](https://github.com/llm-jp/llm-jp-tokenizer/tree/main/models/ver2.2)のSentencePieceモデルをそのまま使用しています．
SentencePieceモデルからHugging Face Fast Tokenizerへのコンバートは[こちらのツール](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/hf/convert_llmjp_unigram_spm_to_hf_fast.py)を使用しています．

※1 各バージョンには50k・100k以外の語彙数のバリエーションもあります。


## 作成方法
### データ
LLM-jpで構築している以下のデータより，一部をサンプリングしたデータを利用しています．
現時点ではいずれのデータの未公開ですので，再現を行う場合はデータの公開をお待ちください．

- `en_pile/train.jsonl`
- `en_wiki/train.jsonl`
- `ja_wiki/train.jsonl`
- `ja_cc/train_[0-9].jsonl`
- `code_stack/train.jsonl`

学習に用いたデータを再現するための手順については，[こちらのドキュメント](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/data/training/howToCreateData.md)をご覧ください．

### 手順
トークナイザーのモデルの作成手順の概要は以下の通りです．

1. データの前処理（日本語テキストの辞書分割など）
2. 言語ごとにSentencePieceのモデル・語彙を作成
3. 語彙の後処理（Prefixの削除や重複語彙の削除など）
4. 各言語で作成した語彙をマージ
5. 全言語のテキストを用いて，マージ後の語彙のスコアを再推定

本リポジトリ内の[手順書](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/scripts/howToCreateModel_ver2.md)をご覧ください．
あるいは，[第3回LLM勉強会の資料のpp3-6](https://drive.google.com/file/d/1Nj4P5NDMvYEy8juQwe6uSqgfYsCYa_E_/edit)も参照いただけます．
