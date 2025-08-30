# chatbot.py
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------- Paths ----------
MODEL_DIR = "SEBRT"
INDEX_FILE = "faiss_index.bin"
ANS_FILE = "answers.pkl"

# ---------- Load model ----------
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR)

# ---------- Load FAISS index and answers ----------
index = faiss.read_index(INDEX_FILE)
answers = joblib.load(ANS_FILE)

# ---------- Chatbot function ----------
def chatbot(query, k=1):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)  # returns k nearest neighbors
    return [answers[i] for i in I[0]]

# ---------- Interactive loop ----------
print("Chatbot ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = chatbot(user_input)
    print("Bot:", response[0])
