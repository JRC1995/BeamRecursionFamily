import copy
import math
import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
import csv
import tensorflow_datasets as tfds
from preprocess_tools.process_utils import jsonl_save

csv.field_size_limit(100000000)
SEED = 101
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

tokenizer = tfds.deprecated.text.ByteTextEncoder()
labels2idx = {"Positive": 1, "Negative": 0}
vocab2idx = {i: i for i in range(tokenizer.vocab_size)}

train_path = Path('../data/AAN/new_aan_pairs.train.tsv')
dev_path = Path('../data/AAN/new_aan_pairs.eval.tsv')
test_path = {}
test_path["normal"] = Path('../data/AAN/new_aan_pairs.test.tsv')

Path('../processed_data/AAN_lra').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/AAN_lra/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/AAN_lra/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/AAN_lra/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/AAN_lra/metadata.pkl"))




def process_data(filename):

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0

    with open(filename, encoding="utf8") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            label = int(float(row[0]))
            sequence1 = row[3]
            sequence2 = row[4]
            sequences1.append(sequence1)
            sequences2.append(sequence2)
            labels.append(label)

            count += 1
            if count % 1000 == 0:
                print("sequence1: ", sequence1)
                print("sequence2: ", sequence2)
                print("label: ", label)
                print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels


dev_sequences1 = {}
dev_sequences2 = {}
dev_labels = {}
dev_sequences1["normal"], dev_sequences2["normal"], dev_labels["normal"] = process_data(dev_path)


test_sequences1 = {}
test_sequences2 = {}
test_labels = {}
test_sequences1["normal"], test_sequences2["normal"], test_labels["normal"] = process_data(test_path["normal"])


train_sequences1, train_sequences2, train_labels = process_data(train_path)


def vectorize_data(sequences1, sequences2, labels):
    data_dict = {}
    sequences1_vec = [tokenizer.encode(sequence)[0:4096] for sequence in sequences1]
    sequences2_vec = [tokenizer.encode(sequence)[0:4096] for sequence in sequences2]

    data_dict["sequence1"] = [sequence.split(" ")[0:1000] for sequence in sequences1]
    data_dict["sequence2"] = [sequence.split(" ")[0:1000] for sequence in sequences2]
    sequences_vec = [sequence1 + sequence2 for sequence1, sequence2 in
                     zip(sequences1_vec, sequences2_vec)]
    data_dict["sequence1_vec"] = sequences1_vec
    data_dict["sequence2_vec"] = sequences2_vec
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences1, train_sequences2, train_labels)

dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences1[key], dev_sequences2[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences1[key], test_sequences2[key], test_labels[key])

print("hello")

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