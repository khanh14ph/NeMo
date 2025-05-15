import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="/home4/khanhnd/improved_unimatch/corpus/large_corpus.txt",
    model_prefix="tokenizer/nemo",
    model_type="bpe",
    user_defined_symbols=[],
    vocab_size=1024,
    input_sentence_size=10000000,
    train_extremely_large_corpus=True,
    shuffle_input_sentence=True,
    # character_coverage=1,
    # treat_whitespace_as_suffix=True,
    # unk_surface="<unk>",
    # pad_id=0,
    # unk_id=1,
    # bos_id=2,
    # eos_id=3,
)