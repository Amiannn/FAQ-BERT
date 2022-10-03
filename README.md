# Simple-HMM-GMM
FAQ系統-使用BERT預訓練語言模型

## Installation
```bash
# Step 1 使用 git 下載專案
git clone https://github.com/Amiannn/FAQ-BERT.git
cd FAQ-BERT

# Step 2 使用 conda 建立虛擬 python 環境
conda create --name faq python=3.7
conda activate faq

# Step 3 安裝套件
pip3 install -r requirements.txt
```

## Dataset
- [Taipei QA](https://github.com/p208p2002/taipei-QA-BERT)
- 使用`preprocess/create_dataset_taipei_qa.py`處理訓練資料。

## Pre-trained Model 與訓練模型
- 可以到[Huggingface](https://huggingface.co/models?language=zh&sort=downloads&search=bert)網站上找適合的預訓練模型。
- 使用`src/download_model.py`將模型下載下來。

## Train 訓練模型
```bash
python3 src/train.py \
    --train_set "./datasets/taipei_qa_train.tsv" \
    --dev_set "./datasets/taipei_qa_eval.tsv" \
    --label_path "./datasets/taipei_qa_labels.tsv" \
    --model_type "./assets/transformers/bert-base-chinese" \
    --output_dir "./exp" \
    --batch_size 8 \
    --epochs 3 \
    --n_gpu 1 
```
或是直接執行
```bash
./run_train.sh
```

## Inference 測試模型
```bash
python3 src/inference.py \
    --input_path "./data/taipei_qa_test.txt" \
    --label_path "./datasets/taipei_qa_labels.tsv" \
    --model_type "./exp/best_model" \
```
或是直接執行
```bash
./run_inference.sh
```

## Reference
- [Simple Transformer](https://simpletransformers.ai/docs/classification-specifics/)
- [台北QA問答機器人(with BERT or ALBERT)](https://github.com/p208p2002/taipei-QA-BERT)
- [李弘毅老師 BERT介紹](https://www.youtube.com/watch?v=UYPa347-DdE&ab_channel=Hung-yiLee)
