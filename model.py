import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pickle

# Load training data
train_data = pd.read_csv("C:\\IE\\email-cl\\train.csv")
train_data['text'] = train_data['Subject'] + " " + train_data['Body']
train_data['text'] = train_data['text'].fillna('')

# Encode target labels
label_encoder = LabelEncoder()
train_data['Category'] = label_encoder.fit_transform(train_data['Category'])

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to sequences
sequences = tokenizer.texts_to_sequences(train_data['text'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split data
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, train_data['Category'], test_size=0.2, random_state=42)

# Define model
embedding_dim = 100
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SpatialDropout1D(0.3),  # Adjusted dropout
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.01))),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.0005)  # Adjusted learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64, callbacks=[early_stopping])

# Save model
model.save('model.h5')

# Save tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save label encoder
with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save max_length
with open('max_length.pkl', 'wb') as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Display accuracy and loss
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

print("Model, tokenizer, label encoder, and max_length saved successfully.")
