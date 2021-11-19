import re
import csv
from normalize_text import normalize_neologd

import random
from tqdm import tqdm


def remove_brackets(text):
    text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
    return text

def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.encode().decode("utf-8").strip()
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

all_data = []

with open("content/raw_data/nucc_dataset.tsv") as f:
  reader = csv.reader(f, delimiter='\t')
  for row in reader:
    utterance = row[0]
    response = row[1]
    utterance = normalize_text(utterance)
    response = normalize_text(response)

    if len(utterance) > 0 and len(response) > 0:
      all_data.append({
          "utterance": utterance,
          "response": response
      })

# データセットを90% : 5%: 5% の比率でtrain/dev/testに分割します。
random.seed(1234)
random.shuffle(all_data)

def to_line(data):
    utterance = data["utterance"]
    response = data["response"]

    assert len(utterance) > 0 and len(response) > 0
    return f"{utterance}\t{response}\n"

data_size = len(all_data)
train_ratio, dev_ratio, test_ratio = 0.9, 0.05, 0.05

with open(f"content/data/train.tsv", "w", encoding="utf-8") as f_train, \
    open(f"content/data/dev.tsv", "w", encoding="utf-8") as f_dev, \
    open(f"content/data/test.tsv", "w", encoding="utf-8") as f_test:
    
    for i, data in tqdm(enumerate(all_data)):
        line = to_line(data)
        if i < train_ratio * data_size:
            f_train.write(line)
        elif i < (train_ratio + dev_ratio) * data_size:
            f_dev.write(line)
        else:
            f_test.write(line)

