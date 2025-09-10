from flask import Flask, render_template, request
import numpy as np
import os, re, pickle
from pathlib import Path
from typing import Tuple, Any, cast

from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def clean_name(name: str) -> str:
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[-–—(){}[\]]', ' ', name)
    name = re.sub(r'[<>:;,_\"!@#$%^&*=+|\\/?~`]', '', name)
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def load_model_and_tokenizer(model_dir: str | os.PathLike) -> Tuple[Model, Any, int]:
    model_dir = Path(model_dir)
    model_path = model_dir / "model.h5"
    tok_path = model_dir / "tokenizer_and_max_seq_length.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")

    mdl = cast(Model, load_model(model_path))
    assert mdl is not None, f"Failed to load model from {model_path}"

    with open(tok_path, "rb") as f:
        tok_data = pickle.load(f)

    # Support either dict or tuple pickle formats
    if isinstance(tok_data, dict):
        tokenizer = tok_data["tokenizer"]
        max_seq_length = int(tok_data["max_seq_length"])
    else:
        tokenizer, max_seq_length = tok_data  # (tokenizer, max_len)

    return mdl, tokenizer, max_seq_length

# Use a path relative to this file, not the CWD
BASE_DIR = Path(__file__).resolve().parent
model, tokenizer, max_seq_length = load_model_and_tokenizer(BASE_DIR / "model")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", predicted_label=None, confidence_score=None)

@app.route("/", methods=["POST"])
def predict():
    name = request.form["name"]
    cleaned = clean_name(name)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding="post")

    pred = model.predict(padded)  # shape (1,1) or (1,) typically
    prob = float(np.squeeze(pred))  # make it a plain float

    predicted_label = int(prob > 0.5)
    confidence_score = prob

    return render_template(
        "index.html",
        name=name,
        predicted_label=predicted_label,
        confidence_score=confidence_score,
    )

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
