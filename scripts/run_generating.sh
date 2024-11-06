# ---------------------------------------------------------------             
#  辞書作成
# ---------------------------------------------------------------
# DATA=Newsela
# python3 scripts/generate_dictionary.py \
#     --input /work/data/datasets/${DATA}/train.jsonl \
#     --output /work/data/datasets/${DATA}/train.tf.json 

# ---------------------------------------------------------------
# src-dst間の単語アラインメントをとる
# ---------------------------------------------------------------
# pip3 install transformers==3.0.2

# GPU=1
# DATA=TURK/first
# SPLIT=valid

# mkdir -p /work/data/outputs/${DATA}/log
# mkdir -p /work/data/outputs/${DATA}/generations
# # wiki train約1日 
# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/semi_crf.py > /work/data/outputs/${DATA}/log/semi_crf_${SPLIT}.log 2>&1\
#     --input_dir /work/data/datasets/${DATA} \
#     --checkpoint /work/models/semi_crf/Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt \
#     --split ${SPLIT} \
#     --batchsize 1 &

# ---------------------------------------------------------------     
# 単語アラインメントから編集操作ラベルの作成
# ---------------------------------------------------------------  
# DATA=TURK/first
# SPLIT=test
# python3 scripts/generate_label.py --input_dir /work/data/datasets/${DATA} --split ${SPLIT}

# ---------------------------------------------------------------     
# 編集操作予測モデル
# ---------------------------------------------------------------  
# pip3 install transformers==4.22.2
# transformers==4.22.2
# pytorch-lightning==1.5.0
# protobuf==3.20.0
# numpy==1.20
# sklearn

# GPU=0
# DATA=Newsela

# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/edit_label_finetune.py > /work/data/outputs/${DATA}/log/edit_label_finetune.log 2>&1\
#     --input_dir /work/data/datasets/${DATA} --model_dir /work/models/edit_label &

# GPU=0
# DATA=Newsela
# SPLIT=test
# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/edit_label_predict.py > /work/data/outputs/${DATA}/log/edit_label_predict.log 2>&1\
#     --input_file /work/data/datasets/${DATA}/${SPLIT}.jsonl --output_file /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.label.txt \
#     --model_dir  /work/models/edit_label/bert-base-uncased_${DATA}/version_0 &

# ---------------------------------------------------------------     
# Moses
# ---------------------------------------------------------------  
# export LC_ALL=C.UTF-8
# export USER=Anything
# MOSES=/work/gen_lex_constraints/mosesdecoder
# DATA=Newsela

# テキストクリーニング
# perl ${MOSES}/scripts/training/clean-corpus-n.perl /work/data/datasets/${DATA}/train src dst /work/data/datasets/${DATA}/train/train.clean 1 500

# 言語モデル
# mkdir -p /work/models/moses/${DATA}
# ${MOSES}/bin/lmplz -o 3 -S 80% -T /tmp < /work/data/datasets/${DATA}/train/train.clean.dst > /work/models/moses/${DATA}/train.dst.arpa

# ${MOSES}/bin/build_binary /work/models/moses/${DATA}/train.dst.arpa /work/models/moses/${DATA}/train.dst.blm
# echo "is this an English sentence ?" | ${MOSES}/bin/query /work/models/moses/${DATA}/train.dst.blm

# 訓練
# cd ${MOSES}
# mkdir tools
# cp -p /work/gen_lex_constraints/giza-pp/GIZA++-v2/GIZA++ tools/ 
# cp -p /work/gen_lex_constraints/giza-pp/GIZA++-v2/snt2cooc.out tools/ 
# cp -p /work/gen_lex_constraints/giza-pp/mkcls/mkcls tools/ 

# ${MOSES}/scripts/training/train-model.perl \
#     --root-dir /work/models/moses/${DATA} \
#     --external-bin-dir /${MOSES}/tools  \
#     --corpus /work/data/datasets/${DATA}/train --f src --e dst \
#     --lm 0:3:/work/models/moses/${DATA}/train.dst.blm:0 

# cp /work/data/datasets/${DATA}/train/train.align.txt /work/models/moses/${DATA}/model/aligned.semi-CRF
# mv /work/models/moses/Newsela/model/lex.e2f /work/models/moses/Newsela/model/lex.e2f.default
# mv /work/models/moses/Newsela/model/lex.f2e /work/models/moses/Newsela/model/lex.f2e.default

# ${MOSES}/scripts/training/train-model.perl \
#     --root-dir /work/models/moses/${DATA} \
#     --external-bin-dir /${MOSES}/tools  \
#     --corpus /work/data/datasets/${DATA}/train --f src --e dst \
#     --lm 0:3:/work/models/moses/${DATA}/train.dst.blm:0 \
#     --alignment semi-CRF \
#     --first-step 4 --last-step 4

# grep wearing /work/models/moses/${DATA}/model/lex.f2e | sort -nrk 3 | head

# ---------------------------------------------------------------             
#  制約作成
# ---------------------------------------------------------------
# -----------------
# CON-3
# -----------------
# DATA=Wiki-Auto
# SPLIT=valid
# GPU=2
# mkdir -p /work/data/datasets/${DATA}/CON-3
# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/generate_constraint_CON-3.py > /work/data/outputs/${DATA}/log/generate_constraint_${SPLIT}_CON-3.log 2>&1\
#     --input /work/data/datasets/${DATA}/${SPLIT}.jsonl \
#     --src_labels /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.label.txt \
#     --dict /work/data/datasets/${DATA}/train.tf.json \
#     --scorer /work/models/roberta_scorer \
#     --output /work/data/datasets/${DATA}/CON-3/${SPLIT}.pred.constraint.jsonl &

# -----------------
# CON-5
# -----------------
# DATA=Newsela
# GPU=0
# SPLIT=valid
# mkdir -p /work/data/datasets/${DATA}/CON-5

# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/generate_constraint_CON-5.py > /work/data/outputs/${DATA}/log/generate_constraint_${SPLIT}.log 2>&1\
#     --input /work/data/datasets/${DATA}/${SPLIT}.jsonl \
#     --src_labels /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.label.txt \
#     --dict /work/data/datasets/${DATA}/train.tf.json \
#     --table /work/models/moses/${DATA}/model/lex.f2e \
#     --output /work/data/datasets/${DATA}/CON-5/${SPLIT}.CON-5.constraint.jsonl &

# -----------------
# CON-6
# -----------------
# DATA=Wiki-Medical
# SPLIT=test.after_del
# GPU=0
# mkdir -p /work/data/datasets/${DATA}/CON-6
# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/generate_constraint_CON-6.py > /work/data/outputs/${DATA}/log/generate_constraint_${SPLIT}_CON-6.log 2>&1\
#     --input /work/data/datasets/${DATA}/${SPLIT}.jsonl \
#     --src_labels /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.label.txt \
#     --dict /work/data/datasets/${DATA}/train.tf.json \
#     --scorer /work/models/roberta_scorer \
#     --output /work/data/datasets/${DATA}/CON-6/${SPLIT}.pred.constraint.jsonl &

# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/generate_constraint_RoBERTa_multiple.py > /work/data/outputs/${DATA}/log/generate_constraint_${SPLIT}_${VERSION}.log 2>&1\
#     --input /work/data/datasets/${DATA}/${SPLIT}.after_del.jsonl \
#     --src_labels /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.after_del.label.txt \
#     --dict /work/data/datasets/${DATA}/train.tf.json \
#     --scorer /work/models/roberta_scorer \
#     --output /work/data/datasets/${DATA}/${VERSION}/${SPLIT}.pred.after_del.constraint.jsonl &

# CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/generate_constraint.py > /work/data/outputs/${DATA}/log/generate_constraint_${SPLIT}_${VERSION}.log 2>&1\
#     --input /work/data/datasets/${DATA}/${SPLIT}.after_del.jsonl \
#     --src_labels /work/data/datasets/${DATA}/${SPLIT}/${SPLIT}.pred.label.after_del.txt \
#     --dict /work/data/datasets/${DATA}/train.tf.json \
#     --table /work/models/moses/${DATA}/model/lex.f2e \
#     --output /work/data/datasets/${DATA}/${VERSION}/${SPLIT}.pred.after_del.constraint.jsonl &
# CUDA_VISIBLE_DEVICES=7 python3 scripts/generate_constraint.py > data/outputs/log/generate_constraint_Newsela-Auto_sample.log 2>&1 &

