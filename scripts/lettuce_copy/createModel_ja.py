import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/mnt/tamanegi/home1/tathi//tokenizer_training_data/cc100ja_3GB.txt.mecab.splitsymbols', 
    model_prefix='cc100ja40K_3GB', 
    vocab_size=40000, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols='<br>',
    max_sentencepiece_length=8,
    split_digits=True,
    byte_fallback=True
)
