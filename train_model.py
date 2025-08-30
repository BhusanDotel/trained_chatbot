# train_model.py
import json
import os
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # <-- for progress bar

# ---------- Paths ----------
DATA_PATH = "data/large_data.json"
MODEL_DIR = "SEBRT"
INDEX_FILE = "faiss_index.bin"
ANS_FILE = "answers.pkl"

# ---------- Load dataset ----------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

questions, answers = [], []
for item in dataset["data"]:
    for q, a in zip(item["questions"], item["answers"]):
        if "input_text" in q and "input_text" in a and a["input_text"].lower() != "unknown":
            questions.append(q["input_text"])
            answers.append(a["input_text"])


# ---------- Load SBERT model locally ----------
os.makedirs(MODEL_DIR, exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR)

# ---------- Compute embeddings ----------
print("Computing embeddings...")
batch_size = 512
question_embeddings = []
for i in tqdm(range(0, len(questions), batch_size)):
    batch = questions[i:i+batch_size]
    emb = model.encode(batch, convert_to_numpy=True)
    question_embeddings.append(emb)
question_embeddings = np.vstack(question_embeddings)

# ---------- Build FAISS index ----------
dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(question_embeddings)

# ---------- Save index + answers ----------
faiss.write_index(index, INDEX_FILE)
joblib.dump(answers, ANS_FILE)
print("Training complete!")
