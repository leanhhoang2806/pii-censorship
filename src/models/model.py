from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from transformers import BertTokenizer

def build_model(input_shape: int, output_categories: int):
    neurons = 32
    dropout = 0.2
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = Sequential([
        Embedding(input_dim=tokenizer.vocab_size, output_dim=neurons, input_shape=(input_shape[1],)),
        Bidirectional(LSTM(neurons*2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, implementation=2)),
        Dense(output_categories, activation='softmax')
    ])
    
    return model
