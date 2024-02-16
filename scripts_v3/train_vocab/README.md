```console
$ cd scripts_v3/train_vocab/
$ pip install -U -r requirements.txt
$ tar zxf small_lex.csv.tgz
$ tar zxf core_lex.csv.tgz

# 最初にMistral＋日本語常用文字を日本語Unigram学習のシードファイルとする(35,216件) ⇒ 作成済み(skip可)
$ python merge_vocab.py \
  Mistral-7B-v0.1_tokenizer.json \
  seed_vocab_leading-spaces.json \
  seed_vocab_regulars-ja.json \
  seed_vocab_ordinals-ja.json \
  > seed_vocab_mistral_regulars-ja_ordinals-ja.json

# 上記シードファイルで日本語のUnigram学習（vocab=150k）を実施した結果を unigram_output_ja-150k.json として次を実行
$ python is_single_word_in_sudachi.py \
  seed_vocab_mistral_regulars-ja_ordinals-ja.json \
  unigram_output_ja-150k.json \
  B rewrite.def small_lex.csv core_lex.csv \
  > unigram_output_ja-150k_filtered_by_core_dict_B.json

# 上記ファイルの件数が45,000件程度であることを確認した上で日本語までの範囲のシードファイルを作成する(約8万件)
$ python merge_vocab.py \
  seed_vocab_mistral_regulars-ja_ordinals-ja.json \
  unigram_output_ja-150k_filtered_by_core_dict_B.json \
  > seed_vocab_mistral_all-ja.json

# 上記ファイル＋簡体字一級・二級（対外訴求としてシードに加えたい）＋中国語Wikipedia頻出文字(30回以上ぐらい)を合わせて韓国語Unigram学習のシードファイルとする(9万件弱で頻度のきりがよいところに調整する)
$ python merge_vocab.py --head 88000 \
  seed_vocab_mistral_all-ja.json \
  seed_vocab_regulars-zh-[12].json \
  zh_wiki_char_freqDict_filtered.json \
  > seed_vocab_mistral_all-ja_all-zh.json

# 上記シードファイルで韓国語のUnigram学習（vocab=100k）を実施した結果を unigram_output_ko-100k.json として次を実行（フィルタを語彙調整学習に合わせる）
$ egrep -v ' "[^"]*([0-9"#$%&()*+/:-@^_`{|}~「」（）]|-|\[|\]|\\)[^"]*"[:,]' \
  unigram_output_ko-100k.json \
  > unigram_output_ko-100k_exclude_symbols.json

# 上記ファイルから語彙調整学習用のシードファイルを作成する(99,000件程度)
$ python merge_vocab.py \
  seed_vocab_mistral_all-ja_all-zh.json \
  unigram_output_ko-100k_exclude_symbols.json \
  > seed_vocab_mistral_all-ja_all-zh_ko.json

# 上記シードファイルで語彙調整学習(vocab=100k)を日本語コーパスで実施した結果を ungiram_output_final-ja-100k.json として次を実行
$ egrep -v ' "[^"]*([0-9"#$%&()*+/:-@^_`{|}~「」（）]|-|\[|\]|\\)[^"]*"[:,]' \
  ungiram_output_final_ja-100k.json \
  > ungiram_output_final_ja-100k_exclude_symbols.json

# 上記ファイル＋異体字セレクタを合わせてスコア再学習用のシードファイル(10万件弱)と語彙コーパスファイル(2文字以上)を作成する
$ python merge_vocab.py \
  seed_vocab_mistral_all-ja_all-zh_ko.json \
  ungiram_output_all-100k_exclude_symbols.json \
  seed_vocab_unicode-svs-ivs.json \
  > seed_vocab_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.json
$ python create_vocab_corpus.py \
  seed_vocab_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.json
  > seed_vocab_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.corpus.txt

# 上記シードファイルを用いて語彙コーパスファイルを加えた全言語のコーパスでスコア再学習を実施した結果を unigram_output_final-all-100k.json として次を実行
$ python merge_vocab.py \
  seed_vocab_special-tokens.json \
  seed_vocab_leading-spaces.json \
  seed_vocab_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.json \
  > seed_vocab_special_leading-spaces_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.json
$ python merge_score.py \
  seed_vocab_special_leading-spaces_mistral_all-ja_all-zh_ko_final-ja_svs-ivs.json \
  unigram_output_final-all-100k.json \
  seed_vocab_special-tokens.json \
  seed_vocab_leading-spaces.json \
  > seed_vocab_special_leading-spaces_mistral_all-ja_all-zh_ko_final-ja_svs-ivs_reestimated.json

# 上記シードファイルからSentencePieceモデルを作成
```