import re
import pickle
import numpy as np
from pathlib import Path

# Keras 2.15 import style (matches your current code)
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import gradio as gr

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_PATH = MODEL_DIR / "model.h5"
TOKENIZER_PKL = MODEL_DIR / "tokenizer_and_max_seq_length.pkl"

def clean_name(name: str) -> str:
    """Remove numbers, special chars (keep apostrophes/diacritics), normalize spaces."""
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[-–—(){}[\]]', ' ', name)
    name = re.sub(r'[<>:;,_\"!@#$%^&*=+|\\/?~`]', '', name)
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

# ---- Load model & tokenizer once (on startup) ----
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not TOKENIZER_PKL.exists():
    raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PKL}")

model = load_model(MODEL_PATH)

with open(TOKENIZER_PKL, "rb") as f:
    tok_data = pickle.load(f)

# Support dict or tuple pickle formats
if isinstance(tok_data, dict):
    tokenizer = tok_data["tokenizer"]
    max_seq_length = int(tok_data["max_seq_length"])
else:
    tokenizer, max_seq_length = tok_data  # (tokenizer, max_len)

def predict_fn(name: str):
    if not name or not name.strip():
        return {"error": "Please enter a name."}

    cleaned = clean_name(name)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding="post")

    # Model usually returns [[prob]] or [prob]; squeeze to float
    prob = float(np.squeeze(model.predict(padded)))  # type: ignore
    label = int(prob > 0.5)

    return {
        "input": name,
        "cleaned": cleaned,
        "predicted_label": label,
        "probability": prob
    }

demo = gr.Interface(
    fn=predict_fn,
    inputs=gr.Textbox(label="Enter a name"),
    outputs=gr.JSON(label="Result"),
    title="Name Classifier",
    description="Type a name to get the prediction (binary label with probability)."
)

if __name__ == "__main__":
    # For local testing: python app.py
    demo.launch()
