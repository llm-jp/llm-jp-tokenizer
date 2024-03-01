# LLM-jp Tokenizer
[LLM勉強会（LLM-jp）](https://llm-jp.nii.ac.jp/)で開発しているLLM用のトークナイザー関連をまとめたリポジトリです．

## What's New
### Release ver3.0a2
#### Hugging Face Fast Tokenizerで使う場合 (**FIXME**)
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
sp = SentencePieceProcessor("models/ver3.0/llm-jp-tokenizer-100k.ver3.0a2.model")
```

## 特徴
[SentencePiece (Unigramモード)](https://github.com/google/sentencepiece)をベースに，以下のような工夫を盛り込んでいます．

1. 既存の大規模言語モデル（Mistral）を日本語・中国語・韓国語を対象に拡張した語彙
2. 言語ごとに含めるべき「必須語彙」を指定した段階的な語彙の拡張
3. 多言語設定に拡張しやすいスケーラブルな学習枠組み

## モデル
各モデルは`/models`以下に配置してあります．
v3.0a2とv2.2の各モデルの語彙の規模は以下の通りです．

v3.0a2のコードと英語の語彙は，Mistralの語彙を借用しています．


| モデル名 | 語彙の規模 | 対応言語 |
| --- | --- | --- |
| code10K_en20K_ja30K.ver2.2.model        | 48,588 | コード，英語，日本語 |
| code20K_en40K_ja60K.ver2.2.model        | 96,869 | コード，英語，日本語 |
| **llm-jp-tokenizer-100k.ver3.0a2.model** | **99,487** | コード，英語，日本語，中国語，韓国語 |


v3.0a2とv2.2の各モデルの分割性能を以下にまとめました．
値は `文字数/分割後のトークン数` で，値が大きいほど圧縮率が高く分割性能が高いと言えます．
後述の各言語のテキストデータに対して分割を行った結果を表示しています．

| モデル名 | 日本語 | |
| --- | --- | --- |
| code10K_en20K_ja30K.ver2.2        | 48,588 | 1.4781 |
| code20K_en40K_ja60K.ver2.2        | 96,869 | 1.5263 |
| **llm-jp-tokenizer-100k.ver3.0a2** | **99,487** | **2.0186** |


## 作成方法
### データ
LLM-jpで構築している以下のデータより，一部をサンプリングしたデータを利用しています．
括弧内はサンプリング後のデータサイズです．
現時点ではいずれのデータの未公開ですので，再現を行う場合はデータの公開をお待ちください．

- コード (0.5GB)
  - Stack
- 英語（1GB）
  - Wikipedia
  - Falcon RefinedWeb
- 日本語 (1.5GB)
  - Wikipedoa
  - SlimPajama (Books, Github以外)
  - Common Crawl
- 中国語 (0.5GB)
  - Wikipedia
- 韓国語 (0.5GB)
  - Wikipedia



### 手順
トークナイザーのモデルの作成手順の概要は以下の通りです．

1. データの前処理（日本語テキストの辞書分割など）
2. 言語ごとにSentencePieceのモデル・語彙を作成
3. 語彙の後処理（Prefixの削除や重複語彙の削除など）
4. 各言語で作成した語彙をマージ
5. 全言語のテキストを用いて，マージ後の語彙のスコアを再推定

本リポジトリ内の[手順書](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/scripts/howToCreateModel_ver2.md)をご覧ください．
あるいは，[第3回LLM勉強会の資料のpp3-6](https://drive.google.com/file/d/1Nj4P5NDMvYEy8juQwe6uSqgfYsCYa_E_/edit)も参照いただけます．
