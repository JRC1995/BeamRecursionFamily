import copy
import math
import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import load_glove, jsonl_save
import csv
import os
import nltk

embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")
MAX_VOCAB = 50000
MIN_FREQ = 5
WORDVECDIM = 300
filter_size = 150

SEED = 101
dev_keys = ["normal"]
test_keys = ["normal",
             "PAWS_QQP",
             "PAWS_WIKI"]
np.random.seed(SEED)
random.seed(SEED)

data_path = Path('../data/QQP/quora_duplicate_questions.tsv')
test_paths = {}
test_paths["PAWS_QQP"] = Path('../data/QQP/PAWS_QQP/dev_and_test.tsv')
test_paths["PAWS_WIKI"] = Path('../data/QQP/PAWS_WIKI/test.tsv')

Path('../processed_data/QQP').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/QQP/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/QQP/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/QQP/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/QQP/metadata.pkl"))

labels2idx = {'Duplicate': 1, "Not-Duplicate": 0}

vocab2count = {}


def tokenize(sequence):
    return sequence.split()


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0) + 1

def process_data(filename, update_vocab=True):
    global labels2idx

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0

    with open(filename, encoding="utf8") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i > 0:
                sequence1 = tokenize(row[3].lower())
                sequence2 = tokenize(row[4].lower())
                if len(sequence1) <= filter_size and len(sequence2) <= filter_size:
                    sequences1.append(sequence1)
                    sequences2.append(sequence2)
                    labels.append(int(row[5]))
                    count += 1
                    if update_vocab:
                        for word in sequence1:
                            updateVocab(word)

                        for word in sequence2:
                            updateVocab(word)

                    if count % 1000 == 0:
                        print("sequence1: ", sequences1[-1])
                        print("sequence2: ", sequences2[-1])
                        print("labels: ", labels[-1])
                        print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels

def process_data2(filename, update_vocab=True):
    global labels2idx

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0

    with open(filename, encoding="utf8") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i > 0:
                sequence1 = tokenize(row[1].lower())
                sequence2 = tokenize(row[2].lower())
                sequences1.append(sequence1)
                sequences2.append(sequence2)
                labels.append(int(row[3]))
                count += 1
                if update_vocab:
                    for word in sequence1:
                        updateVocab(word)

                    for word in sequence2:
                        updateVocab(word)

                if count % 1000 == 0:
                    print("sequence1: ", sequences1[-1])
                    print("sequence2: ", sequences2[-1])
                    print("labels: ", labels[-1])
                    print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels

sequences1, sequences2, labels = process_data(data_path, update_vocab=True)

idx = [i for i in range(len(labels))]
dup_idx = [i for i in idx if labels[i] == 1]
nodup_idx = [i for i in idx if labels[i] == 0]
random.shuffle(dup_idx)
random.shuffle(nodup_idx)

dev_sequences1 = {}
dev_sequences2 = {}
dev_labels = {}
test_sequences1 = {}
test_sequences2 = {}
test_labels = {}

dev_idx = dup_idx[0:5000] + nodup_idx[0:5000]
test_idx = dup_idx[5000:10000] + nodup_idx[5000:10000]
train_idx = dup_idx[10000:] + nodup_idx[10000:]

dev_sequences1["normal"] = [sequences1[i] for i in dev_idx]
dev_sequences2["normal"] = [sequences2[i] for i in dev_idx]
dev_labels["normal"] = [labels[i] for i in dev_idx]

test_sequences1["normal"] = [sequences1[i] for i in test_idx]
test_sequences2["normal"] = [sequences2[i] for i in test_idx]
test_labels["normal"] = [labels[i] for i in test_idx]

train_sequences1 = [sequences1[i] for i in train_idx]
train_sequences2 = [sequences2[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]

test_sequences1["PAWS_QQP"], test_sequences2["PAWS_QQP"], test_labels["PAWS_QQP"] = process_data2(test_paths["PAWS_QQP"], update_vocab=True)
test_sequences1["PAWS_WIKI"], test_sequences2["PAWS_WIKI"], test_labels["PAWS_WIKI"] = process_data2(test_paths["PAWS_WIKI"], update_vocab=True)

print("training_size: ", len(train_sequences1))


counts = []
vocab = []
for word, count in vocab2count.items():
    if count > MIN_FREQ:
        vocab.append(word)
        counts.append(count)

vocab2embed = load_glove(embedding_path, vocab=vocab2count, dim=WORDVECDIM)

sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx if vocab[id] in vocab2embed]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]

vocab += ["<PAD>", "<UNK>", "<SEP>"]

print(vocab)

vocab2idx = {word: id for id, word in enumerate(vocab)}

vocab2embed["<PAD>"] = np.zeros((WORDVECDIM), np.float32)
b = math.sqrt(3 / WORDVECDIM)
vocab2embed["<UNK>"] = np.random.uniform(-b, +b, WORDVECDIM)
vocab2embed["<SEP>"] = np.random.uniform(-b, +b, WORDVECDIM)

embeddings = []
for id, word in enumerate(vocab):
    embeddings.append(vocab2embed[word])


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


def vectorize_data(sequences1, sequences2, labels):
    data_dict = {}
    sequences1_vec = [text_vectorize(sequence) for sequence in sequences1]
    sequences2_vec = [text_vectorize(sequence) for sequence in sequences2]
    data_dict["sequence1"] = sequences1
    data_dict["sequence2"] = sequences2
    sequences_vec = [sequence1 + [vocab2idx["<SEP>"]] + sequence2 for sequence1, sequence2 in
                     zip(sequences1_vec, sequences2_vec)]
    data_dict["sequence1_vec"] = sequences1_vec
    data_dict["sequence2_vec"] = sequences2_vec
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences1, train_sequences2, train_labels)
"""
for item in train_data["sequence1"]:
    print(item)
print("\n\n")
"""
dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences1[key], dev_sequences2[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences1[key], test_sequences2[key], test_labels[key])

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
            "embeddings": np.asarray(embeddings, np.float32),
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)