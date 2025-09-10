# People Name Classifier Web Application using character level RNN model

This interactive web application is an extension of the People Name Classifier project, which utilizes a character-level RNN model to discern if a given string is a person's name. Building upon the core functionality of the [character level model repository](https://github.com/rohithramesh1991/name_classifier_challenge_Char_Rnn.git), 
this application provides a user-friendly interface for real-time predictions.

## Overview

This repository contains the code for a web application that interfaces with the People Name Classifier model. Users can enter strings, and the application will classify them as a person's name or not.

# Name Classifier (Gradio on Hugging Face Spaces)

Type a name and get a binary prediction from a Keras model.

## Files
- `app.py` — Gradio UI + model inference
- `model/model.h5` — Keras model
- `model/tokenizer_and_max_seq_length.pkl` — tokenizer + max length (dict or tuple)

## Run locally
```bash
pip install -r requirements.txt
python app.py
# open the local Gradio URL
```

# 5) Deploy on Hugging Face Spaces (free)

1. Create a free account at **huggingface.co**.
2. Go to **Spaces → Create new Space**:
   - **SDK**: Gradio
   - **Visibility**: Public (so anyone can test)
   - **Hardware**: CPU Basic (free)
3. Upload these files (`app.py`, `requirements.txt`, `README.md`, and the `model/` folder).
   - You can drag-and-drop in the Space UI.
   - If `model.h5` is big, uploading via web UI is easiest. (Git LFS also works.)
4. The Space will **auto-build** and then show a **public URL**. Share that link!

---

## Tips & gotchas

- **Model size / memory**: CPU Basic is fine for most TF/Keras CPU models. If it crashes while loading, reduce model size or contact me to optimize (or switch to `tensorflow-cpu`).
- **Cold start**: First request after sleeping may take a few seconds while the model loads. It stays warm when there’s some traffic.
- **Private to public**: If you set the Space to private first, you can switch it to **Public** later so “anyone can test it.”
