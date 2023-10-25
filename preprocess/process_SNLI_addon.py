import math
import pickle
import random
from os import fspath
from pathlib import Path
import csv
import jsonlines
import nltk
import numpy as np

from preprocess_tools.process_utils import load_glove, jsonl_save

metadata_path = Path("../processed_data/SNLI/metadata.pkl")
with open(metadata_path, 'rb') as fp:
    metadata = pickle.load(fp)

vocab2idx = metadata["vocab2idx"]
labels2idx = metadata["labels2idx"]
dev_keys = ["normal"]
test_keys = ["normal", "hard", "break", "counterfactual"]

metadata["test_keys"] = test_keys


test_path = Path('../data/SNLI/revised_combined/test.tsv')
test_save_path = Path('../processed_data/SNLI/test_counterfactual.jsonl')

metadata_save_path = fspath(Path("../processed_data/SNLI/metadata.pkl"))
with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def process_data(filename):
    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0
    with open(filename, encoding="utf8") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            print(row)
            if i > 0:
                sequence1 = tokenize(row[0].lower())
                sequence2 = tokenize(row[1].lower())
                label = row[2]
                label_id = labels2idx[label]

                sequences1.append(sequence1)
                sequences2.append(sequence2)
                labels.append(label_id)

                count += 1

                if count % 1000 == 0:
                    print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels


test_sequences1, \
test_sequences2, \
test_labels = process_data(test_path)


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


test_data = vectorize_data(test_sequences1, test_sequences2, test_labels)

jsonl_save(filepath=test_save_path,
           data_dict=test_data)