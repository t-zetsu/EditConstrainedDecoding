GPU=0
DATA_DIR=../data/sample
MODEL_DIR=../models/sample
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/decoder/decode.py > ${DATA_DIR}/decode.log 2>&1 \
	--model_name ${MODEL_DIR} \
    --input_file ${DATA_DIR}/test.jsonl \
    --prune_factor 80 --min_tgt_length 5 --max_tgt_length 128 \
	--beam_size 20 --bs 64 --length_penalty 0.0001 \
	--save_path ${DATA_DIR}/test.pred.txt &

