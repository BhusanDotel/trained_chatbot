# Chatbot with Local SBERT + FAISS

A fast and scalable chatbot that answers questions using semantic similarity.  
It uses ** (SBERT)** to encode questions and answers, and stores embeddings in a **FAISS index** for near-instant retrieval. Fully local, no API calls required.

---

## Features

- Fast question-answer retrieval using **semantic embeddings**
- Works with large datasets
- Fully local, no external API
- Logs progress during training

---

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`:

Train the model first
python train_model.py

Run the chatbot
python chatbot.py
