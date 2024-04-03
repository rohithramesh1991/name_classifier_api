from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import re
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


def clean_name(name):

    """Remove numbers.
    Removes hyphens (-), en dashes (–), em dashes (—), parentheses
    and any other special characters that are not required.
    Retained apostrophe and diacritics"""

    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[-–—(){}[\]]', ' ', name)
    name = re.sub(r'[<>:;,_\"!@#$%^&*=+|\\/?~`]', '', name)
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name


def load_model_and_tokenizer(model_dir):
    # Load the model
    model_path = os.path.join(model_dir, 'model.h5')
    model = load_model(model_path)

    # Load the tokenizer and max_seq_length
    tokenizer_max_seq_length_path = os.path.join(model_dir, 'tokenizer_and_max_seq_length.pkl')
    with open(tokenizer_max_seq_length_path, 'rb') as file:
        tokenizer_data = pickle.load(file)

    return model, tokenizer_data['tokenizer'], tokenizer_data['max_seq_length']


model, tokenizer, max_seq_length = load_model_and_tokenizer('./model')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', predicted_label=None, confidence_score=None)


@app.route('/', methods=['POST'])
def predict():
    name = request.form['name']
    cleaned_name = clean_name(name)
    sequence = tokenizer.texts_to_sequences([cleaned_name])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post')
    prediction = model.predict(padded_sequence)

    # Assuming your model outputs a probability
    predicted_label = int(prediction > 0.5)
    confidence_score = np.max(prediction)  # Confidence score of the prediction

    return render_template('index.html',
                           name=name,
                           predicted_label=predicted_label,
                           confidence_score=confidence_score)


if __name__ == "__main__":
    app.run(debug=True)

