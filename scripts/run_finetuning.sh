GPU=0
DATA_DIR=../data/sample
MODEL_DIR=../models/sample
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/finetuning/finetune.py > ${DATA_DIR}/finetune.log 2>&1 \
  --gpus 1 \
  --data_dir ${DATA_DIR} --output_dir=${MODEL_DIR} \
  --model_name_or_path facebook/bart-base  --cache_dir ${MODEL_DIR}/tmp\
  --num_workers 4  --train_batch_size 64 --eval_batch_size 64 --gradient_accumulation_steps 4 \
  --max_source_length 128 --max_target_length 128 --val_max_target_length 128 --test_max_target_length 128 \
  --do_train --task translation --num_train_epochs 5  --learning_rate 0.0001 --weight_decay 0.01 --n_val=-1 \
  --do_predict --beam_size 20 --length_penalty 2.0 --ngram_size 3 \
  --gpus 1 &
