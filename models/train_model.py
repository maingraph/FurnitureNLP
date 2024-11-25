import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


with open('data.json', 'r') as f:
    data = json.load(f)

texts = [item['description'] for item in data]
labels = [item['labels'] for item in data]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

max_len = 20
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

mlb = MultiLabelBinarizer()
y_labels = mlb.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y_labels, test_size=0.2, random_state=42
)


model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(mlb.classes_), activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

def predict_products(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded)
    predicted_labels_indices = np.where(prediction > 0.5)[1]
    product_names = mlb.classes_[predicted_labels_indices]
    return product_names.tolist()


test_description = "https://qualitywoods.com/products/qw-amish-mckee-6pc-dining-set?variant=31793708105810"
predicted_products = predict_products(test_description)
print(f"Predicted products for '{test_description}': {predicted_products}")