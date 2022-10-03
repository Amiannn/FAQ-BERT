TRAIN_SET="./datasets/taipei_qa_train.tsv"
DEV_SET="./datasets/taipei_qa_eval.tsv"
LABEL_PATH="./datasets/taipei_qa_labels.tsv"
OUTPUT_DIR="./exp"

MODEL_TYPE="./assets/transformers/bert-base-chinese"
BATCH_SIZE=8
EPOCHS=3
NGPU=1

python3 src/train.py \
    --train_set $TRAIN_SET \
    --dev_set $DEV_SET \
    --label_path $LABEL_PATH \
    --batch_size $BATCH_SIZE \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --n_gpu $NGPU 