import os
import json
import random

def load_data(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(' ')
            datas.append(data)
    return datas

def create_FAQ_dataset(path, datas, ans2lab):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(f'{data[1]}\t{ans2lab[data[0]]}\n')

def create_labels_file(path, labels, ans2lab):
    with open(path, 'w', encoding='utf-8') as fr:
        for label in labels:
            fr.write(f'{ans2lab[label]}\t{label}\n')

if __name__ == '__main__':
    data_path = './data/taipei_qa.txt'
    datas   = load_data(data_path)
    answers = list(set([d[0] for d in datas]))
    ans2lab = dict([[ans, i] for i, ans in enumerate(answers)])
    

    random.shuffle(datas)

    filename  = (data_path.split('/')[-1]).split('.')[0]

    train_output_path = f'./datasets/{filename}_train.tsv'
    create_FAQ_dataset(train_output_path, datas[:-200], ans2lab)

    eval_output_path  = f'./datasets/{filename}_eval.tsv'
    create_FAQ_dataset(eval_output_path, datas[-200:], ans2lab)

    label_output_path = f'./datasets/{filename}_labels.tsv'
    create_labels_file(label_output_path, answers, ans2lab)
