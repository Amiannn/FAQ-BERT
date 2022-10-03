INPUT_PATH="./data/taipei_qa_test.txt"
LABEL_PATH="./datasets/taipei_qa_labels.tsv"
# trained model
MODEL_TYPE="./exp/best_model"

python3 src/inference.py \
    --input_path $INPUT_PATH \
    --label_path $LABEL_PATH \
    --model_type $MODEL_TYPE \