import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pandas as pd
import re
import random
from sklearn.metrics.pairwise import cosine_similarity

import os
import gdown

# --- Download FAISS index from Google Drive if not present ---
FAISS_FILE_ID = "1jUZ9929E6aWMdTJaLU87L2ca_4MQG9Si"
FAISS_DEST = "therapy_faiss.index"

if not os.path.exists(FAISS_DEST):
    print("Downloading FAISS index from Google Drive...")
    url = f"https://drive.google.com/uc?id={FAISS_FILE_ID}"
    gdown.download(url, FAISS_DEST, quiet=False)
# --- Streamlit Setup ---
st.set_page_config(
    page_title="AI Virtual Therapist (Demo)",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Load FAISS Index ---
index = faiss.read_index('therapy_faiss.index')

# --- Load Therapist Responses ---
with open('therapist_responses.pkl', 'rb') as f:
    therapist_responses = pickle.load(f)

# --- Response Cleaning ---
def clean_response(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^\W+|\W+$', '', text)
    return text

# --- Fallback Responses ---
def fallback_response():
    fallback_responses = [
        "It sounds like you're carrying a lot right now. Would you like to share more about that?",
        "It's okay to feel this way. Let's take it one small step at a time.",
        "Emotions can feel overwhelming. What’s one thing that helps ground you?",
        "You are not alone in this. I’m here to listen.",
        "Take a deep breath. What’s on your mind right now?",
        "It's perfectly normal to feel this way sometimes. Can you tell me more?",
        "I’m here with you. Would you like to explore this feeling together.",
    ]
    return random.choice(fallback_responses)

# --- Response Variation ---
def vary_response(response):
    variations = [
        "",  # no change
        " Let’s explore that together.",
        " Take your time to share more.",
        " It’s okay, we can talk through it.",
        " You’re doing well by opening up.",
        " Your feelings are valid.",
        " I'm here for you.",
    ]
    variation = random.choice(variations)
    return response + variation

# --- Similarity Search ---
def find_best_response(user_input, threshold=0.40):
    user_embedding = model.encode([user_input])
    faiss.normalize_L2(user_embedding)

    K = 5
    D, I = index.search(np.array(user_embedding).astype('float32'), k=K)

    top_responses = [therapist_responses[idx] for idx in I[0]]
    top_embeddings = model.encode(top_responses)
    sims = cosine_similarity(user_embedding, top_embeddings)[0]

    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    best_response = top_responses[best_idx]

    if best_sim >= threshold:
        response = best_response
    else:
        response = fallback_response()

    response = clean_response(response)
    response = vary_response(response)
    return response

# --- Context Window ---
def build_context_window(current_prompt):
    context_parts = []

    prev_user_msg = None
    prev_bot_msg = None

    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user" and prev_user_msg is None:
            prev_user_msg = msg["content"]
        elif msg["role"] == "assistant" and prev_bot_msg is None:
            prev_bot_msg = msg["content"]
        if prev_user_msg and prev_bot_msg:
            break

    if prev_user_msg:
        context_parts.append(f"Previous User: {prev_user_msg}")
    if prev_bot_msg:
        context_parts.append(f"Previous Assistant: {prev_bot_msg}")

    context_parts.append(f"Current User: {current_prompt}")

    context_str = " | ".join(context_parts)
    return context_str

# --- App UI ---
st.title("🧠 AI Virtual Therapist (Demo)")

# --- Disclaimer ---
st.info("""
**Disclaimer:** This is an AI-powered *virtual therapy assistant demo*.
It is not a substitute for professional therapy, counseling, or medical advice.
If you are experiencing a crisis or mental health emergency, please seek help from qualified professionals.

*This is an educational project using public and synthetic data. It is not intended for real therapeutic use.*
""")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("How are you feeling today?"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context window
    context_str = build_context_window(prompt)

    # Get best response
    response = find_best_response(context_str)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
