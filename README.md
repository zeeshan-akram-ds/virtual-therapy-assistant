# 🧠 AI Virtual Therapist (Demo)

This is a simple AI-powered Virtual Therapy Assistant demo.

**Important:** This project is for educational and demonstration purposes only.  
It is *not* a substitute for professional therapy or counseling.

## How it works

- It combines:
  - Sentence-Transformer embeddings (`all-MiniLM-L6-v2`)
  - FAISS similarity search index
  - Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- The dataset is a merge of:
  - EmpatheticDialogues
  - DailyDialog
  - Small sample therapist Q/A pairs
- The app uses a context window to maintain limited conversation memory.

## Limitations

- Responses are retrieved based on vector similarity — **topic mismatch can occasionally occur**.
- The system is *not fine-tuned* and does not perform reasoning.
- It is a proof-of-concept for retrieval-augmented conversational AI.

## Try it

1. Enter your feelings, concerns, or emotional questions.
2. The system will respond with an empathetic message from its response index.
3. You can continue the conversation.

# Disclaimer
This is a demo project built for learning and practice.  
It is not intended for real therapeutic use. If you are in need of support, please seek help from qualified professionals.

# Author
Zeeshan Akram — Software Engineering Student  

Guided and optimized with help from AI mentors.
