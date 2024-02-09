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

# ==================
# import json
# import numpy as np
# import tensorflow as tf
# from transformers import BertTokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import classification_report
# import random
# from tqdm import tqdm 

# # physical_devices = tf.config.list_physical_devices('GPU')
# # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# single_GPU = True
# small_sample = 0.05

# # Load data from JSON file
# with open("pii-detection-removal-from-educational-data/train.json") as file:
#     json_data = json.load(file)
# sampled_data = random.sample(json_data, int(small_sample * len(json_data)))

# if single_GPU:
#     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
# else:
#     strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()



# with strategy.scope():

#     documents = []
#     expected_output = []
#     all_labels = set()

#     for item in sampled_data:
#         documents.append(item["full_text"])
#         expected_output.append(item["labels"])
#         for i in item["labels"]:
#             if i != "O":
#                 all_labels.add(i)
#     all_data = zip(documents, expected_output)
#     label_to_index = {}

#     for index, item in enumerate(list(all_labels)):
#         label_to_index[item] = index + 1

#     label_to_index["O"] = 0
#     for item in expected_output:
#         for index, i in enumerate(item):
#             item[index] = label_to_index[i]
    
#     # Initialize BERT tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Tokenize train sentences
#     train_encodings = tokenizer(documents, padding="max_length", truncation=True, return_tensors='np')
#     Y_train = [[label for label in sent] for sent in expected_output]
#     Y_train = pad_sequences(Y_train, padding="post", maxlen=train_encodings['input_ids'].shape[1])

#     # Convert labels to one-hot encoding
#     num_classes = len(label_to_index)
#     Y_train = [to_categorical(i, num_classes=num_classes) for i in Y_train]

#     # Model Architecture
#     model = Sequential([
#         Embedding(input_dim=len(tokenizer.get_vocab()), output_dim=50, input_length=train_encodings['input_ids'].shape[1]),
#         Bidirectional(LSTM(units=50, return_sequences=True)),
#         TimeDistributed(Dense(num_classes, activation='softmax'))
#     ])

#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     epochs = 1
#     with tqdm(total=epochs, desc="Epochs") as pbar_epochs:
#         for epoch in range(epochs):
#             with tqdm(total=len(train_encodings['input_ids']), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
#                 for i in range(len(train_encodings['input_ids'])):
#                     # Batch training here
#                     model.train_on_batch(train_encodings['input_ids'][i:i+1], np.array([Y_train[i]]))
#                     pbar.update(1)
#             pbar_epochs.update(1)


#     # Select a random example from the training set
#     random_index = random.randint(0, len(sampled_data) - 1)

#     # Extract the text and labels from the selected example
#     test_text = documents[random_index]
#     test_labels = expected_output[random_index]

#     # Tokenize the test text
#     test_encoding = tokenizer(test_text, padding="max_length", truncation=True, return_tensors='np')

#     # Predict labels using the trained model
#     predictions = model.predict(test_encoding['input_ids'])

#     # Decode predictions
#     predicted_labels = np.argmax(predictions, axis=-1)
#     idx2label = {v: k for k, v in label_to_index.items()}
#     decoded_predicted_labels = [idx2label[label] for label in predicted_labels[0]]

#     # Convert test labels to indices
#     Y_test = [label for label in test_labels]
#     Y_test = pad_sequences([Y_test], padding="post", maxlen=test_encoding['input_ids'].shape[1])[0]

#     # Flatten the arrays
#     Y_test_flat = Y_test.flatten()
#     predicted_labels_flat = predicted_labels.flatten()

#     print("Count of non-zero Y_test_flat:", len([i for i in Y_test_flat if i != 0]))
#     # Print the classification report
#     print("Classification Report:")
#     print(classification_report(Y_test_flat, predicted_labels_flat, zero_division=1))

#     # Calculate and print accuracy metrics
#     general_accuracy = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i]]) / len(predicted_labels_flat)
#     print("General Accuracy:", general_accuracy)

#     accuracy_without_0 = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i] and predicted_labels_flat[i] != 0]) / len(predicted_labels_flat)
#     print("Accuracy without class 0:", accuracy_without_0)


#     # Select a random example from the training set
#     random_index = random.randint(0, len(documents) - 1)
#     test_text = documents[random_index]
#     test_labels = expected_output[random_index]

#     # Tokenize the test text
#     test_encoding = tokenizer(test_text, padding="max_length", truncation=True, return_tensors='np')

#     # Predict labels using the trained model
#     predictions = model.predict(test_encoding['input_ids'])

#     # Decode predictions
#     predicted_labels = np.argmax(predictions, axis=-1)
#     idx2label = {v: k for k, v in label_to_index.items()}
#     decoded_predicted_labels = [idx2label[label] for label in predicted_labels[0]]

#     # Split the text into words
#     words = test_text.split()

#     # Map each word to its corresponding predicted label
#     word_predictions = list(zip(words, decoded_predicted_labels))

#     # Print the word predictions
#     # for word, prediction in word_predictions:
#     #     print(f"Word: {word}, Prediction: {prediction}")


# ================================
import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from tensorflow.keras import layers

single_GPU = True
small_sample = 1
epochs = 10
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Load data from JSON file
with open("pii-detection-removal-from-educational-data/train.json") as file:
    json_data = json.load(file)
sampled_data = random.sample(json_data, int(small_sample * len(json_data)))

if single_GPU:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():

    documents = []
    expected_output = []
    all_labels = set()

    for item in sampled_data:
        documents.append(item["full_text"])
        expected_output.append(item["labels"])
        for i in item["labels"]:
            if i != "O":
                all_labels.add(i)
    all_data = zip(documents, expected_output)
    label_to_index = {}

    for index, item in enumerate(list(all_labels)):
        label_to_index[item] = index + 1

    label_to_index["O"] = 0
    for item in expected_output:
        for index, i in enumerate(item):
            item[index] = label_to_index[i]

    # Split the data into train and test sets
    train_documents, test_documents, train_expected_output, test_expected_output = train_test_split(
        documents, expected_output, test_size=0.1, random_state=42
    )

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize train sentences
    train_encodings = tokenizer(train_documents, padding="max_length", truncation=True, return_tensors='np')
    Y_train = [[label for label in sent] for sent in train_expected_output]
    Y_train = pad_sequences(Y_train, padding="post", maxlen=train_encodings['input_ids'].shape[1])

    # Convert labels to one-hot encoding
    num_classes = len(label_to_index)
    Y_train = [to_categorical(i, num_classes=num_classes) for i in Y_train]

    # Model Architecture
    model = Sequential([
        Embedding(input_dim=len(tokenizer.get_vocab()), output_dim=50, input_length=train_encodings['input_ids'].shape[1]),
        Bidirectional(LSTM(units=128, return_sequences=True)),
        layers.Dropout(0.5),  # Adding dropout for regularization
        Bidirectional(LSTM(units=64, return_sequences=True)),
        layers.Dropout(0.5),
        TimeDistributed(Dense(num_classes, activation='softmax'))
    ])

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Convert class weights to a dictionary
    class_weight = {value: 50. for _, value in label_to_index.items()}
    class_weight[0] = 1.

    # Train the model with class weights
    model.fit(train_encodings['input_ids'], np.array(Y_train), class_weight=class_weight, batch_size=1, epochs=epochs)

    test_encodings = tokenizer(test_documents, padding="max_length", truncation=True, return_tensors='np')
    test_predictions = model.predict(test_encodings['input_ids'])

    # Decode predictions
    test_predicted_labels = np.argmax(test_predictions, axis=-1)
    idx2label = {v: k for k, v in label_to_index.items()}
    decoded_test_predicted_labels = [[idx2label[label] for label in sent] for sent in test_predicted_labels]

    # Convert test labels to indices
    Y_test = [[label for label in sent] for sent in test_expected_output]
    Y_test = pad_sequences(Y_test, padding="post", maxlen=test_encodings['input_ids'].shape[1])

    # Flatten the arrays
    Y_test_flat = Y_test.flatten()
    predicted_labels_flat = test_predicted_labels.flatten()

    print("Count of non-zero Y_test_flat:", len([i for i in Y_test_flat if i != 0]))
    # Print the classification report
    print("Classification Report:")
    print(classification_report(Y_test_flat, predicted_labels_flat, zero_division=1))

    # Calculate and print accuracy metrics
    general_accuracy = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i]]) / len(predicted_labels_flat)
    print("General Accuracy:", general_accuracy)

    accuracy_without_0 = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i] and predicted_labels_flat[i] != 0]) / len(predicted_labels_flat)
    print("Accuracy without class 0:", accuracy_without_0)

    random_index = random.randint(0, len(test_documents) - 1)
    test_text = test_documents[random_index]
    test_labels = test_expected_output[random_index]

    # Tokenize the test text
    test_encoding = tokenizer(test_text, padding="max_length", truncation=True, return_tensors='np')

    # Predict labels using the trained model
    predictions = model.predict(test_encoding['input_ids'])

    # Decode predictions
    predicted_labels = np.argmax(predictions, axis=-1)
    idx2label = {v: k for k, v in label_to_index.items()}
    decoded_predicted_labels = [idx2label[label] for label in predicted_labels[0]]

    # Split the text into words
    words = test_text.split()

    # Map each word to its corresponding predicted label
    word_predictions = list(zip(words, decoded_predicted_labels))

    # Print the word predictions
    for word, prediction in word_predictions:
        if prediction != "O": print(f"Word: {word}, Prediction: {prediction}")

