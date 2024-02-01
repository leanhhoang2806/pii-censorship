import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
import json

# Load data
with open("pii-detection-removal-from-educational-data/train.json") as file:
    json_data = json.load(file)

documents = []
expected_output = []
all_labels = set()
for item in json_data:
    documents.append(item["full_text"])
    expected_output.append(item["labels"])
    for i in item["labels"]:
        if i != 'O':
            all_labels.add(i)

label_to_index = {}
for index, item in enumerate(list(all_labels)):
    label_to_index[item] = index + 1

label_to_index['O'] = 0
for item in expected_output:
    for index, i in enumerate(item):
        item[index] = label_to_index[i]

neurons = 32
dropout = 0.2
output_categories = len(all_labels)

# Create MirroredStrategy for distributed training
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy

with strategy.scope():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors='tf')
    max_input_len = tf.shape(tokenized_inputs['input_ids'])[1]
    input_shape = tokenized_inputs['input_ids'].shape

    # Encode labels to match the model's output shape
    encoded_labels = np.zeros((len(expected_output), max_input_len), dtype=np.int32)
    model = Sequential([
        Embedding(input_dim=tokenizer.vocab_size, output_dim=neurons, input_shape=(input_shape[1],)),
        Bidirectional(LSTM(neurons*2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, implementation=2)),
        Bidirectional(LSTM(neurons, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, implementation=2)),
        Dense(output_categories, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(tokenized_inputs['input_ids'], encoded_labels, epochs=10, batch_size=1)

    # Print the summary of the model
    model.summary()

    new_texts = ["Another John joined the class."]

    # Tokenize the new texts
    new_tokenized_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='tf')

    # Make predictions using the trained model
    predictions = model.predict(new_tokenized_inputs['input_ids'])

    # Print the predicted labels
    predicted_labels = tf.argmax(predictions, axis=-1).numpy()
    print("Predicted Labels:")
    print(predicted_labels)
