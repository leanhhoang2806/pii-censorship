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

# # parameters Testing: is_small_sample = True
# # small training batches: single_GPU = True

# # real training: is_small_sample = False, single_GPU = False
# small_sample = None
# strategy = None
# epochs = 1
# batch_size = 1


# is_small_sample = True
# single_GPU = True


# if is_small_sample:
#     small_sample = 0.001
# else:
#     small_sample = 1
# # Sample a portion of the data for faster testing
# sampled_data = random.sample(json_data, int(small_sample * len(json_data)))
# print(f"Total number of data to train: {len(sampled_data)}")

# # Extract features and labels
# documents = []
# expected_output = []
# all_labels = set()

# for item in sampled_data:
#     documents.append(item["full_text"])
#     expected_output.append(item["labels"])
#     for i in item["labels"]:
#         if i != "O":
#             all_labels.add(i)
# all_data = zip(documents, expected_output)
# label_to_index = {}

# for index, item in enumerate(list(all_labels)):
#     label_to_index[item] = index + 1

# label_to_index["O"] = 0
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
# if single_GPU:
#     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
# else:
#     strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# with strategy.scope():
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     tokenized_train_inputs = tokenizer(
#         train_documents, padding=True, truncation=True, return_tensors="tf"
#     )
#     max_input_len = tf.shape(tokenized_train_inputs["input_ids"])[1]
#     input_shape = tokenized_train_inputs["input_ids"].shape

#     # Encode labels to match the model's output shape
#     encoded_train_labels = np.zeros((len(train_labels), max_input_len), dtype=np.int32)

#     model = Sequential(
#         [
#             Embedding(
#                 input_dim=tokenizer.vocab_size,
#                 output_dim=neurons,
#                 input_length=max_input_len,
#             ),
#             Bidirectional(
#                 LSTM(
#                     neurons * 2,
#                     return_sequences=True,
#                     dropout=dropout,
#                     recurrent_dropout=dropout,
#                     implementation=2,
#                 )
#             ),
#             Dense(output_categories, activation="softmax"),
#         ]
#     )

#     # Compile the model with class weights
#     model.compile(
#         optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
#     )

#     # Train the model with tqdm progress bar
#     steps_per_epoch = len(tokenized_train_inputs["input_ids"]) // batch_size

#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
#         progress_bar = tqdm(
#             total=steps_per_epoch, position=0, leave=True
#         )  # Create tqdm progress bar

#         for step in range(steps_per_epoch):
#             start_idx = step * batch_size
#             end_idx = (step + 1) * batch_size
#             batch_inputs = tokenized_train_inputs["input_ids"][start_idx:end_idx]
#             batch_labels = encoded_train_labels[start_idx:end_idx]

#             # Training step
#             model.train_on_batch(batch_inputs, batch_labels)

#             # Update progress bar
#             progress_bar.update(1)
#             progress_bar.set_description(f"Batch {step}/{steps_per_epoch}")

#         progress_bar.close()

#     # Print the summary of the model
#     # model.summary()

#     # Tokenize test documents
#     tokenized_test_inputs = tokenizer(
#         test_documents, padding=True, truncation=True, return_tensors="tf"
#     )

#     encoded_test_labels = np.zeros((len(test_labels), max_input_len), dtype=np.int32)

#     for i, labels in enumerate(test_labels):
#         for j, label in enumerate(labels):
#             if j < max_input_len:  # Ensure not to cross the maximum allowed length
#                 encoded_test_labels[i][j] = label

#     # Update encoded_test_labels according to the number of samples after tokenization
#     num_test_samples_after_tokenization = tokenized_test_inputs["input_ids"].shape[0]
#     encoded_test_labels = encoded_test_labels[:num_test_samples_after_tokenization]

#     # Predict labels for the tokenized test documents
#     predicted_labels = model.predict(tokenized_test_inputs["input_ids"])
#     # Transform model's output to label id
#     predicted_labels_id = []

#     for pred_array in predicted_labels:
#         max_value_index = np.argmax(pred_array, axis=1)
#         predicted_labels_id.append(max_value_index.tolist())

#     # check for data consistency
#     assert len(test_documents) == len(predicted_labels_id)
#     print(test_labels[0])
#     print(predicted_labels_id[0])
#     for i in range(len(predicted_labels_id)):
#         print(f"length of predicted labels: {len(predicted_labels_id[i])}, len of test labels: {len(test_labels[i])}")
#         assert len(predicted_labels_id[i]) == len(test_labels[i])

#     # Converting predicted labels ids to make it flat because the classification report
#     # expects a 1d y_pred and y_true.
#     predicted_labels_1d_flat = [
#         label for prediction in predicted_labels_id for label in prediction
#     ]

#     # Flatten the test_labels without padding
#     test_labels_1d_flat = [
#         label for labels in test_labels for label in labels
#     ]

#     print(
#         classification_report(
#             test_labels_1d_flat, predicted_labels_1d_flat, digits=4, zero_division=1
#         )
#     )

#     # Print three samples of text, true labels, and predicted labels
#     # for i in range(3):
#     #     print(f"Sample {i + 1}")
#     #     print("Text:", test_documents[i])
#         # print("True Labels:", test_labels[i])
#         # print("Predicted Labels:", predicted_labels_id[i])
#         # print(predicted_labels_1d[i])
#         # print("\n")


# =========================================
import numpy as np
import tensorflow as tf

# Sample training data
X_train = np.array([
    [1, 2, 3, 4, 0, 0],  # "Apple is a technology company."
    [5, 6, 7, 8, 9, 0],  # "Tim Cook is the CEO of Apple."
    [10, 11, 12, 13, 14, 15]  # "San Francisco is located in California."
])
y_train = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]],  # ORG, ORG, O, O, O, O
    [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],  # ORG, O, O, O, ORG, O
    [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]  # LOC, O, O, O, LOC, LOC
])

# Define and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=16, output_dim=32, input_length=6),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: ORG, LOC, O
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Example prediction
test_sentence = np.array([[1, 2, 3, 16, 0, 0]])  # "Apple is headquartered in <unknown>."
predictions = model.predict(test_sentence)
predicted_labels = [np.argmax(pred) for pred in predictions[0]]
print(predicted_labels)  # Output: [0, 2, 2, 2, 2, 2] -> Predicted labels for each word

