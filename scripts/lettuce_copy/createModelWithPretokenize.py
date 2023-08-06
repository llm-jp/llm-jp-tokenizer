import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/mnt/tamanegi/home1/tathi/cc100/downsampled_ja_7GB.txt.mecab', 
    model_prefix='sp', 
    vocab_size=32000, 
    num_threads=72,
    train_extremely_large_corpus=True,
    pretokenization_delimiter="||||",
    seed_sentencepiece_size=1000000, #*100,
    max_sentencepiece_length=8,
    split_digits=True,
    byte_fallback=True
)
