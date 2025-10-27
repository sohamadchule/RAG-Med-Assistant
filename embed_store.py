import psycopg2
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import google.generativeai as genai
import os

# ---------------- CONFIG ----------------
# Connection and API key
conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_F1T2wBfntjkZ@ep-dawn-surf-a43p87u0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
)
cur = conn.cursor()


# Table names must match those you actually created!
cur.execute("""
CREATE TABLE IF NOT EXISTS medical_qa (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT
);
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS medical_embeddings (
    id SERIAL PRIMARY KEY,
    qa_id INT REFERENCES medical_qa(id),
    embedding BYTEA
);
""")
conn.commit()


# Set Gemini API key
genai.configure(api_key="AIzaSyConSej9K_7OG6dwyeiJuL9JqMQ2_hKAwI")
# ----------------------------------------

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 1️⃣ Fetch QA data in small chunks
cur.execute("SELECT id, question, answer FROM medical_qa;")
rows = cur.fetchall()

# 2️⃣ Create embeddings table
cur.execute("""
CREATE TABLE IF NOT EXISTS medical_embeddings (
    id SERIAL PRIMARY KEY,
    qa_id INT REFERENCES medical_qa(id),
    embedding BYTEA
);
""")
conn.commit()

# 3️⃣ Generate embeddings and insert
for row in tqdm(rows, desc="Generating embeddings"):
    qa_id, q, a = row
    text = (q + " " + a).strip()
    # optional chunking if text very long
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    chunk_embeds = [model.encode(c) for c in chunks]
    emb = np.mean(chunk_embeds, axis=0)  # average of chunks
    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
    cur.execute("INSERT INTO medical_embeddings (qa_id, embedding) VALUES (%s, %s)", (qa_id, emb_bytes))

conn.commit()
print("✅ Embeddings stored successfully!")

