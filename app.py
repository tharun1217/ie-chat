from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Load model, tokenizer, and label encoder
model = load_model('model.h5')  # or 'model_updated.h5'

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load max_length
with open('max_length.pkl', 'rb') as handle:
    max_length = pickle.load(handle)

# Function to load and preprocess email data
def load_and_preprocess_emails():
    emails_data = pd.read_csv("C:\\IE\\email-cl\\Output\\emails.csv")
    emails_data['text'] = emails_data['Subject'] + " " + emails_data['Body']
    emails_data['text'] = emails_data['text'].fillna('')

    email_sequences = tokenizer.texts_to_sequences(emails_data['text'])
    padded_email_sequences = pad_sequences(email_sequences, maxlen=max_length, padding='post')
    
    predicted_probabilities = model.predict(padded_email_sequences)
    predicted_categories = np.argmax(predicted_probabilities, axis=1)
    predicted_categories = label_encoder.inverse_transform(predicted_categories)
    emails_data['Predicted_Category'] = predicted_categories

    return emails_data

# Load initial emails data
emails_data = load_and_preprocess_emails()

@app.route('/')
def index():
    categories_count = emails_data['Predicted_Category'].value_counts().to_dict()
    return render_template('index.html', categories=categories_count)

@app.route('/category/<category_name>')
def category(category_name):
    category_emails = emails_data[emails_data['Predicted_Category'] == category_name]
    return jsonify(category_emails.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
