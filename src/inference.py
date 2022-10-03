import os
import logging
import argparse
import pandas as pd

from simpletransformers.classification import (
    ClassificationModel, 
    ClassificationArgs
)

def load_data(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            datas.append(data)
    return datas

def load_labels(path):
    labels, lab2ans = [], {}
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split('\t')
            label= int(data[0])
            ans  = data[1]
            labels.append(label)
            lab2ans[label] = ans

    return labels, lab2ans

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description="Inference a BERT model on a FAQ task")
parser.add_argument("--input_path"    , type=str,  default=None, help="train_input_path.")
parser.add_argument("--label_path"    , type=str,  default=None, help="label_path.")
parser.add_argument("--model_type"    , type=str,  default='ckiplab/bert-base-chinese', help="huggingface model path or saved model path.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model args
model_args = ClassificationArgs()

# Loading test datas
input_datas = load_data(args.input_path)

# Optional model configuration
labels, lab2ans = load_labels(args.label_path)

model_args.labels_list = labels

# Create a ClassificationModel
model = ClassificationModel(
    "bert", args.model_type, num_labels=len(labels), args=model_args, use_cuda=False
)

# Make predictions with the model
predictions, raw_outputs = model.predict(input_datas)

for i, data in enumerate(input_datas):
    pred = lab2ans[predictions[1]]
    print(f'Question: {data}, Answer: {pred}')