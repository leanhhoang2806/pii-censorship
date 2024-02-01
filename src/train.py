from data.data_generator import generated_data
from src.preprocessing.tokenizer import generate_token, label_expected_result
from src.models.model import build_model
from transformers import BertTokenizer
import tensorflow as tf


def main():
    texts, labels = generated_data()

    tokenized_inputs = generate_token(texts)
    max_input_len = tf.shape(tokenized_inputs['input_ids'])[1]
    encoded_labels = label_expected_result(labels, max_input_len)

    input_shape = tokenized_inputs['input_ids'].shape
    output_categories = 3
    model = build_model(input_shape, output_categories)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(tokenized_inputs['input_ids'], encoded_labels, epochs=10, batch_size=1)
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


if __name__ == "__main__":
    main()