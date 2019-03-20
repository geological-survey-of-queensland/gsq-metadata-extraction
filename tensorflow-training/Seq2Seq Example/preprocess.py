import os
import pickle
import copy
import numpy as np
import tensorflow as tf
from collections import Counter


# special codes
CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


def main():
    source_path = 'small_vocab_en.txt'
    target_path = 'small_vocab_fr.txt'
    source_text = load_data(source_path)
    target_text = load_data(target_path)
    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_data = text_to_ints(source_text, source_vocab_to_int)
    target_data = text_to_ints(target_text, target_vocab_to_int, append="<EOS>")

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab),
        (source_data, target_data)), open('preprocess.p', 'wb'))


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data


def create_lookup_tables(text):
    vocab = set(text.split())

    vocab_to_int = copy.copy(CODES)

    # string => int
    for i, v in enumerate(vocab, (len(CODES))):
        vocab_to_int[v] = i

    # int => string
    int_to_vocab = {i: v for v, i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def text_to_ints(text, vocab_to_ints, append=""):
    
    data = []

    for sentence in text.split('\n'):
        sentence = sentence + " " + append
        data.append([vocab_to_ints[s] for s in sentence.split()])
        
    return data


if __name__ == "__main__": main()