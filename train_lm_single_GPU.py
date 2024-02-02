import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
import json
from tqdm import tqdm 

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

# Using GPU, assuming TensorFlow is configured to use GPU
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

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

    # Train the model with tqdm progress bar
    epochs = 10
    batch_size = 1
    steps_per_epoch = len(tokenized_inputs['input_ids']) // batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(total=steps_per_epoch, position=0, leave=True)  # Create tqdm progress bar

        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size
            batch_inputs = tokenized_inputs['input_ids'][start_idx:end_idx]
            batch_labels = encoded_labels[start_idx:end_idx]

            # Training step
            model.train_on_batch(batch_inputs, batch_labels)

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_description(f"Batch {step}/{steps_per_epoch}")

        progress_bar.close()

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
