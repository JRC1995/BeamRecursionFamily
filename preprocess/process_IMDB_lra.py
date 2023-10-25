import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import tensorflow_datasets as tfds

SEED = 101
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

data = tfds.load('imdb_reviews')
train_raw = data['train']
valid_raw = data['test']
test_raw = data['test']


def adapt_example(example):
    return {'Source': example['text'], 'Target': example['label']}


train = train_raw.map(adapt_example)
valid = valid_raw.map(adapt_example)
test = test_raw.map(adapt_example)

tokenizer = tfds.deprecated.text.ByteTextEncoder()

labels2idx = {"Positive": 1, "Negative": 0}
vocab2idx = {i: i for i in range(tokenizer.vocab_size)}

Path('../processed_data/IMDB_lra').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/IMDB_lra/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/IMDB_lra/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/IMDB_lra/test_{}.jsonl'.format(key))
test_save_path["contrast_pair"] = Path('../processed_data/IMDB_lra/test_{}.jsonl'.format("contrast_pair"))
metadata_save_path = fspath(Path("../processed_data/IMDB_lra/metadata.pkl"))

def process_data(dataset):
    global labels2idx

    sequences = []
    labels = []

    for x in dataset:
        text = x["Source"].numpy().decode("utf-8")
        label = int(x["Target"].numpy())
        sequences.append(text)
        labels.append(label)

    return sequences, labels


train_sequences, train_labels = process_data(train)

test_sequences = {}
test_labels = {}
test_sequences["normal"], test_labels["normal"] = process_data(test)

dev_sequences = {}
dev_labels = {}
dev_sequences["normal"], dev_labels["normal"] = process_data(valid)



def vectorize_data(sequences, labels):
    data_dict = {}
    sequences_vec = [tokenizer.encode(sequence)[0:4096] for sequence in sequences]
    data_dict["sequence"] = [sequence.split(" ") for sequence in sequences]
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences, train_labels)

dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences[key], test_labels[key])

jsonl_save(filepath=train_save_path,
           data_dict=train_data)

for key in dev_keys:
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

for key in test_keys:
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

metadata = {"labels2idx": labels2idx,
            "vocab2idx": vocab2idx,
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
