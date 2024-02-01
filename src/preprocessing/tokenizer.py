
from transformers import BertTokenizer
import tensorflow as tf
import numpy as np

def label_expected_result(labels, max_input_len):
    label_to_index = {"O": 0, "B-Student": 1, "I-Student": 2}
    for label in labels:
        for index, word in enumerate(label):
            label[index] = label_to_index[word]

    encoded_labels = np.zeros((len(labels), max_input_len), dtype=np.int32)
    return encoded_labels

def generate_token(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

    return tokenized_inputs
