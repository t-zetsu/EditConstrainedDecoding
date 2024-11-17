export LC_ALL=C.UTF-8
export USER=Anything
GPU=0
DATA_DIR=../data/sample
MODEL_DIR=../models/sample
MOSES=../models/mosesdecoder

# Dictionary
python3 ../src/constraint_generator/generate_dictionary.py \
    --input ${DATA_DIR}/train.jsonl \
    --output ${DATA_DIR}/dic.train.json 

# Word Alignment
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/constraint_generator/semi_crf.py > ${DATA_DIR}/semi_crf.log 2>&1 \
    --input_dir ${DATA_DIR} \
    --checkpoint ../models/semi_crf/Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt \
    --split train \
    --batchsize 1 &

# Edit Labels
python3 ../src/constraint_generator/generate_label.py
    --input_dir ${DATA_DIR}
    --split train

# Finetune
pip3 install -r ../requirements.const.txt
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/constraint_generator/edit_label_finetune.py > ${DATA_DIR}/edit_label_finetune.log 2>&1\
    --input_dir ${DATA_DIR} \
    --model_dir ${MODEL_DIR}/edit_label &

# Predict
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/constraint_generator/edit_label_predict.py > ${DATA_DIR}/edit_label_predict.log 2>&1\
    --input_file ${DATA_DIR}/test.jsonl \
    --output_file ${DATA_DIR}/test.label.pred.txt \
    --model_dir ${MODEL_DIR}/edit_label/bert-base-uncased_${DATA}/version_0 &

# Moses
perl ${MOSES}/scripts/training/clean-corpus-n.perl ${DATA_DIR}/train src dst ${DATA_DIR}/train.clean 1 500

# Language Model
mkdir -p /work/models/moses/${DATA}
${MOSES}/bin/lmplz -o 3 -S 80% -T /tmp < ${DATA_DIR}/train/train.clean.dst > ${DATA_DIR}/moses/train.dst.arpa

${MOSES}/bin/build_binary ${DATA_DIR}/moses/train.dst.arpa ${DATA_DIR}/moses/train.dst.blm
echo "is this an English sentence ?" | ${MOSES}/bin/query ${DATA_DIR}/moses/train.dst.blm

# Moses Training
cp -p ../models/giza-pp/GIZA++-v2/GIZA++ ${MOSES}/tools/ 
cp -p ../models/giza-pp/GIZA++-v2/snt2cooc.out ${MOSES}/tools/ 
cp -p ../models/giza-pp/mkcls/mkcls ${MOSES}/tools/ 

${MOSES}/scripts/training/train-model.perl \
    --root-dir ${DATA_DIR}/moses \
    --external-bin-dir ${MOSES}/tools  \
    --corpus ${DATA_DIR}/train --f src --e dst \
    --lm 0:3:${DATA_DIR}/moses/train.dst.blm:0 

cp ${DATA_DIR}/train.align.txt ${DATA_DIR}/moses/aligned.semi-CRF
mv ${DATA_DIR}/moses/lex.e2f ${DATA_DIR}/moses/lex.e2f.default
mv ${DATA_DIR}/moses/lex.f2e ${DATA_DIR}/moses/lex.f2e.default

${MOSES}/scripts/training/train-model.perl \
    --root-dir ${DATA_DIR}/moses \
    --external-bin-dir /${MOSES}/tools  \
    --corpus ${DATA_DIR}/train --f src --e dst \
    --lm 0:3:${DATA_DIR}/moses/train.dst.blm:0 \
    --alignment semi-CRF \
    --first-step 4 --last-step 4

grep wearing ${DATA_DIR}/moses/lex.f2e | sort -nrk 3 | head

# Generating
CUDA_VISIBLE_DEVICES=${GPU} python3 ../src/constraint_generator/generate_constraint.py > ${DATA_DIR}/generate_constraint.log 2>&1 \
    --input ${DATA_DIR}/test.jsonl \
    --src_labels ${DATA_DIR}/test.label.pred.txt \
    --dict ${DATA_DIR}/dic.train.json  \
    --table ${DATA_DIR}/moses/lex.f2e \
    --output ${DATA_DIR}/test.constraint.pred.jsonl  &
    
