import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Data
train_sentences = [
    "John lives in New York City.",
    "He works for Microsoft in Seattle.",
    "Mary went to Paris last summer."
]
train_labels = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"],
    ["B-PER", "O", "O", "O", "B-ORG", "O", "B-LOC"],
    ["B-PER", "O", "O", "B-LOC", "O", "O", "O"]
]

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize train sentences
train_encodings = tokenizer(train_sentences, padding="max_length", truncation=True, return_tensors='np')

# Convert labels to indices
label2idx = {"O": 0, "B-PER": 1, "B-LOC": 2, "B-ORG": 3, "I-LOC": 4}
Y_train = [[label2idx[label] for label in sent] for sent in train_labels]
Y_train = pad_sequences(Y_train, padding="post", maxlen=train_encodings['input_ids'].shape[1])

# Convert labels to one-hot encoding
num_classes = len(label2idx)
Y_train = [to_categorical(i, num_classes=num_classes) for i in Y_train]

# Model Architecture
model = Sequential([
    Embedding(input_dim=len(tokenizer.get_vocab()), output_dim=50, input_length=train_encodings['input_ids'].shape[1]),
    Bidirectional(LSTM(units=50, return_sequences=True)),
    TimeDistributed(Dense(num_classes, activation='softmax'))
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_encodings['input_ids'], np.array(Y_train), batch_size=1, epochs=10)

# Test sentences
test_sentences = [
    "Peter visited London.",
    "The company Apple is located in California."
]

# Tokenize test sentences
test_encodings = tokenizer(test_sentences, padding="max_length", truncation=True, return_tensors='np')

# Predictions
predictions = model.predict(test_encodings['input_ids'])


# Test labels
test_labels = [
    ["B-PER", "O", "B-LOC", "O"],
    ["O", "O", "B-ORG", "O", "O", "B-LOC"]
]

# Convert test labels to indices
Y_test = [[label2idx[label] for label in sent] for sent in test_labels]
Y_test = pad_sequences(Y_test, padding="post", maxlen=test_encodings['input_ids'].shape[1])


# Decode predictions
predicted_labels = np.argmax(predictions, axis=-1)

# Flatten the arrays
Y_test_flat = Y_test.flatten()
predicted_labels_flat = predicted_labels.flatten()
# Decode predictions
idx2label = {v: k for k, v in label2idx.items()}
for i, sentence in enumerate(test_sentences):
    decoded_labels = [idx2label[np.argmax(pred)] for pred in predictions[i]]


def map_labels_to_words(sentences, predicted_labels):
    mapped_predictions = []
    for i, sentence in enumerate(sentences):
        words = sentence.split()  # Split the sentence into words
        labels = predicted_labels[i * len(words):(i + 1) * len(words)]  # Get the predicted labels for this sentence
        mapped_sentence = [(word, labels[j]) for j, word in enumerate(words)]  # Map each word to its predicted label
        mapped_predictions.append(mapped_sentence)
    return mapped_predictions

# Convert predicted labels to mapped tuples
mapped_predictions = map_labels_to_words(test_sentences, predicted_labels_flat)

# Print mapped predictions
for i, sentence in enumerate(mapped_predictions):
    print(f"Sentence: {test_sentences[i]}")
    print(f"Mapped Predictions: {sentence}")
    print()


# Calculate and print classification report
print(classification_report(Y_test_flat, predicted_labels_flat, zero_division=1))
general_accuracy = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i]]) / len(predicted_labels_flat)

print("General accuracy: ", general_accuracy)

accuracy_without_0 = len([i for i in range(len(predicted_labels_flat)) if predicted_labels_flat[i] == Y_test_flat[i] and predicted_labels_flat[i] != 0]) / len(predicted_labels_flat)

print("Accuracy without class 0: ", accuracy_without_0)