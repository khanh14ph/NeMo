python NeMo/scripts/tokenizers/process_asr_text_tokenizer.py\
  --data_file /home4/khanhnd/Ezspeech/data/vlsp2020.jsonl,/home4/khanhnd/Ezspeech/ezspeech/resource/corpus/librispeech_upper.txt \
  --data_root /home4/khanhnd/nemo_dev/tokenizer/mix\
  --vocab_size 2024\
  --tokenizer spe\
  --spe_type bpe\
  --spe_character_coverage 1