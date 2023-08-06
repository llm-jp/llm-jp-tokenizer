import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/mnt/tamanegi/home1/tathi//tokenizer_training_data/cc100en_3GB.txt.splitsymbols', 
    model_prefix='cc100en20K_3GB', 
    vocab_size=20000, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols='<br>',
    max_sentencepiece_length=16,
    split_digits=True,
    byte_fallback=True
)
