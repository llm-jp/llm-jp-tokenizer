# LLM-jp Tokenizer
[LLM勉強会（LLM-jp）](https://llm-jp.nii.ac.jp/)で開発しているLLM用のトークナイザー関連をまとめたリポジトリです．

## What's New
### Release ver3.0b1
#### Hugging Face Fast Tokenizerで使う場合 (**FIXME**)
（作業中）

#### SentencePieceで使う場合
- 必須ライブラリ
  - sentencepiece>=0.1.99
  - protobuf<3.21.0
- 使い方
```Python
from sentencepiece import SentencePieceProcessor
sp = SentencePieceProcessor("models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model")
```

## 特徴
[SentencePiece (Unigramモード)](https://github.com/google/sentencepiece)をベースに，以下のような工夫を盛り込んでいます．

1. 既存の大規模言語モデル（Mistral）を日本語・中国語・韓国語を対象に拡張した語彙
2. 言語ごとに含めるべき「必須語彙」を指定した段階的な語彙の拡張
3. 多言語設定に拡張しやすいスケーラブルな学習枠組み

## モデル
各モデルは`/models`以下に配置してあります．
v3.0b1とv2.2の各モデルの語彙の規模は以下の通りです．

v3.0b1のコードと英語の語彙は，Mistralの語彙を借用しています．


| モデル名 | 語彙の規模 | 対応言語 |
| --- | --- | --- |
| code10K_en20K_ja30K.ver2.2        | 48,588 | コード，英語，日本語 |
| code20K_en40K_ja60K.ver2.2        | 96,869 | コード，英語，日本語 |
| **llm-jp-tokenizer-100k.ver3.0b1** | 99,487 | コード，英語，日本語，中国語，韓国語 |


v3.0a2とv2.2の各モデルの分割性能を以下にまとめました．
値は `文字数/分割後のトークン数` で，値が大きいほど圧縮率が高く分割性能が高いと言えます．
後述の各言語のテキストデータに対して分割を行った結果を表示しています．

|モデル名|コード|英語|日本語|中国語|韓国語|
|--|--|--|--|--|--|
|code10K_en20K_ja30K.ver2.2  |2.5742|3.6677|1.4782|0.8757|0.4689|
|code20K_en40K_ja60K.ver2.2  |2.6715|3.8826|1.5263|0.8845|0.4697|
|**llm-jp-tokenizer-100k.ver3.0b1**|2.7450|3.9467|2.0186|1.2370|2.0428|

## 作成方法
### データ
LLM-jpで構築している以下のデータより，一部をサンプリングしたデータを利用しています．
括弧内はサンプリング後のデータサイズです．
現時点ではいずれのデータも未公開ですので，再現を行う場合はデータの公開をお待ちください．

- コード (0.5GB)
  - Stack
- 英語（1GB）
  - Wikipedia
  - Falcon RefinedWeb
- 日本語 (1.5GB)
  - Wikipedia
  - SlimPajama (Books, Github以外)
  - Common Crawl
- 中国語 (0.5GB)
  - Wikipedia
- 韓国語 (0.5GB)
  - Wikipedia



### Tokenizerモデルの作成手順
（追記予定）
