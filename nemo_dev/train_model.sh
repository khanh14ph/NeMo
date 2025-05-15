#!/bin/bash
# export CUDA_VISIBLE_DEVICES=3
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
        --config-path="/home4/khanhnd/NeMo/nemo_dev/config" --config-name="fastconformer_hybrid_tdt_ctc_bpe" \
        ++model.train_ds.manifest_filepath="/home4/khanhnd/Ezspeech/data/vn.jsonl" \
        ++model.validation_ds.manifest_filepath="/home4/khanhnd/Ezspeech/data/test.jsonl" \