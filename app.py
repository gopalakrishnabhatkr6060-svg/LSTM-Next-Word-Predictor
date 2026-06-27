import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="📝",
    layout="centered"
)
st.title("📝 LSTM Next Word Prediction")
st.write("Enter a sentence and let the LSTM model predict the next words.")

# ----------------------------
# Required files
# ----------------------------
MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN_PATH = "max_len.pkl"


def _require_file(path: str):
    if not os.path.exists(path):
        st.error(f"Required file not found: `{path}`. Make sure it's in the app's working directory.")
        st.stop()


# ----------------------------
# Load Model / Tokenizer / Max Len
# ----------------------------
@st.cache_resource
def load_lstm_model():
    _require_file(MODEL_PATH)
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


@st.cache_resource
def load_tokenizer():
    _require_file(TOKENIZER_PATH)
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        st.stop()


# max_len is a plain value, not a heavy resource -> cache_data is the
# semantically correct decorator here (cache_resource is meant for
# models/connections that shouldn't be hashed/serialized).
@st.cache_data
def load_max_len():
    _require_file(MAX_LEN_PATH)
    try:
        with open(MAX_LEN_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load max_len: {e}")
        st.stop()


model = load_lstm_model()
tokenizer = load_tokenizer()
max_len = load_max_len()

# Build index -> word lookup once, instead of scanning word_index every call
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


# ----------------------------
# Sampling
# ----------------------------
def sample_with_temperature(preds, temperature: float = 1.0) -> int:
    """Sample an index from a probability distribution.

    temperature == 0 falls back to plain argmax (greedy/deterministic).
    Higher temperature -> more random/creative output.
    """
    if temperature <= 0:
        return int(np.argmax(preds))

    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    return int(np.random.choice(len(probs), p=probs))


# ----------------------------
# Predict Next Word
# ----------------------------
def predict_next_word(model, tokenizer, text, max_len, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        # None of the seed words are in the tokenizer's vocabulary.
        return None, "oov"

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )
    predicted = model.predict(token_list, verbose=0)[0]
    predicted_index = sample_with_temperature(predicted, temperature)

    # Index 0 is the padding token and is never a real word in
    # tokenizer.word_index (which starts at 1) -> treat it as "no word".
    output_word = index_to_word.get(predicted_index)
    if output_word is None:
        return None, "pad"

    return output_word, None


# ----------------------------
# Generate Text
# ----------------------------
def generate_text(model, tokenizer, seed_text, n_words, max_len, temperature=1.0):
    warning = None
    for _ in range(n_words):
        next_word, reason = predict_next_word(
            model,
            tokenizer,
            seed_text,
            max_len,
            temperature
        )
        if next_word is None:
            if reason == "oov" and warning is None:
                warning = "None of the words in your seed text are in the model's vocabulary, so generation stopped early."
            break
        seed_text += " " + next_word
    return seed_text, warning


# ----------------------------
# UI
# ----------------------------
seed = st.text_area(
    "Enter starting sentence",
    placeholder="Example: once upon a time"
)

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider(
        "Number of words to generate",
        min_value=1,
        max_value=20,
        value=5
    )
with col2:
    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.5,
        value=0.0,
        step=0.1,
        help="0 = deterministic (always picks the most likely word). Higher = more varied/creative, but less predictable."
    )

if st.button("🚀 Generate"):
    if seed.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            generated, warning = generate_text(
                model,
                tokenizer,
                seed,
                num_words,
                max_len,
                temperature
            )
        if warning:
            st.warning(warning)
        st.success("Generated Text")
        st.text_area(
            "",
            value=generated,
            height=150
        )
