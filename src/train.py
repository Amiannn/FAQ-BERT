import os
import logging
import argparse
import pandas as pd

from simpletransformers.classification import (
    ClassificationModel, 
    ClassificationArgs
)

def load_dataset(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split('\t')
            data[1] = int(data[1])
            datas.append(data)

    df = pd.DataFrame(datas)
    df.columns = ["text", "labels"]
    return df

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
parser = argparse.ArgumentParser(description="Finetune a BERT model on a FAQ task")
parser.add_argument("--train_set"     , type=str,  default=None, help="train_input_path.")
parser.add_argument("--dev_set"       , type=str,  default=None, help="test_input_path.")
parser.add_argument("--label_path"    , type=str,  default=None, help="label_path.")
parser.add_argument("--batch_size"    , type=int,  default=8   , help="batch size.")
parser.add_argument("--model_type"    , type=str,  default='ckiplab/bert-base-chinese', help="huggingface model path or saved model path.")
parser.add_argument("--epochs"        , type=int,  default=3 , help="training epochs.")
parser.add_argument("--n_gpu"         , type=int,  default=1 , help="numbers of gpu.")
parser.add_argument("--output_dir"    , type=str,  default="outputs/", help="output path for the model.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model args
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.num_train_epochs = args.epochs
model_args.train_batch_size = args.batch_size
model_args.no_save = False
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True
model_args.output_dir = args.output_dir
model_args.best_model_dir = os.path.join(args.output_dir, 'best_model')
model_args.n_gpu = args.n_gpu


# Preparing train data
train_df = load_dataset(args.train_set)

# Preparing eval data
eval_df  = load_dataset(args.dev_set)

# Optional model configuration
labels, lab2ans = load_labels(args.label_path)

model_args.labels_list = labels

# Create a ClassificationModel
model = ClassificationModel(
    "bert", args.model_type, num_labels=len(labels), args=model_args
)

# Train the model
model.train_model(train_df, eval_df=eval_df)
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])