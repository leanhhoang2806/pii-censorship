# import tensorflow as tf
# import numpy as np
# from transformers import BertTokenizer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import json
# from tqdm import tqdm 
# import random

# # Load data
# with open("pii-detection-removal-from-educational-data/train.json") as file:
#     json_data = json.load(file)

# # Sample a portion of the data for faster testing
# sampled_data = random.sample(json_data, int(0.01 * len(json_data)))
# print(f"Total number of data to train: {len(sampled_data)}")

# # Extract features and labels
# documents = []
# expected_output = []
# all_labels = set()

# for item in sampled_data:
#     documents.append(item["full_text"])
#     expected_output.append(item["labels"])
#     for i in item["labels"]:
#         if i != 'O':
#             all_labels.add(i)
# all_data = zip(documents, expected_output)
# label_to_index = {}

# for index, item in enumerate(list(all_labels)):
#     label_to_index[item] = index + 1

# label_to_index['O'] = 0
# for item in expected_output:
#     for index, i in enumerate(item):
#         item[index] = label_to_index[i]

# # Split the data into train and test sets
# train_documents, test_documents, train_labels, test_labels = train_test_split(
#     documents, expected_output, test_size=0.2, random_state=42
# )

# neurons = 32
# dropout = 0.2
# output_categories = len(label_to_index)

# # Using GPU, assuming TensorFlow is configured to use GPU
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# with strategy.scope():
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     tokenized_train_inputs = tokenizer(train_documents, padding=True, truncation=True, return_tensors='tf')
#     max_input_len = tf.shape(tokenized_train_inputs['input_ids'])[1]
#     input_shape = tokenized_train_inputs['input_ids'].shape

#     # Encode labels to match the model's output shape
#     encoded_train_labels = np.zeros((len(train_labels), max_input_len), dtype=np.int32)
    
#     model = Sequential([
#         Embedding(input_dim=tokenizer.vocab_size, output_dim=neurons, input_shape=(input_shape[1],)),
#         Bidirectional(LSTM(neurons*2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, implementation=2)),
#         Dense(output_categories, activation='softmax')
#     ])

#     # Compile the model with class weights
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Train the model with tqdm progress bar
#     epochs = 1
#     batch_size = 1
#     steps_per_epoch = len(tokenized_train_inputs['input_ids']) // batch_size

#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
#         progress_bar = tqdm(total=steps_per_epoch, position=0, leave=True)  # Create tqdm progress bar

#         for step in range(steps_per_epoch):
#             start_idx = step * batch_size
#             end_idx = (step + 1) * batch_size
#             batch_inputs = tokenized_train_inputs['input_ids'][start_idx:end_idx]
#             batch_labels = encoded_train_labels[start_idx:end_idx]

#             # Training step
#             model.train_on_batch(batch_inputs, batch_labels)

#             # Update progress bar
#             progress_bar.update(1)
#             progress_bar.set_description(f"Batch {step}/{steps_per_epoch}")

#         progress_bar.close()

#     # Print the summary of the model
#     model.summary()

#     # Tokenize test documents
#     tokenized_test_inputs = tokenizer(test_documents, padding=True, truncation=True, return_tensors='tf')

#     # Predict labels for the tokenized test documents
#     predicted_labels = model.predict(tokenized_test_inputs['input_ids'])
#     print(test_labels[0])
#     print("================")
#     print(predicted_labels[0])

from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

# Define your labeled training data (replace this with your own dataset)
training_data = [
    {"sentence": "Apple Inc. is a technology company.", "labels": [0, 0, 0, 0, 1, 1, 1, 2]},
    {"sentence": "Microsoft is based in Redmond.", "labels": [0, 0, 0, 1, 2, 2]},
    # Add more labeled examples as needed
]

# Define the labels (replace these with your own labels)
labels = ["O", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER"]

# Set up the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels)
)

# Tokenize and prepare the training data
tokenized_data = tokenizer(
    [example["sentence"] for example in training_data],
    truncation=True,
    padding=True,
    return_tensors="pt",
    is_split_into_words=True,
)

# Extract labels and convert them to tensor
labels_tensor = torch.tensor(
    [example["labels"] for example in training_data], dtype=torch.long
)