import numpy as np
import psycopg2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure API key for Gemini
genai.configure(api_key="AIzaSyConSej9K_7OG6dwyeiJuL9JqMQ2_hKAwI")

# Connect to Neon
conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_F1T2wBfntjkZ@ep-dawn-surf-a43p87u0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
)
cur = conn.cursor()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieval Function ---
def get_similar_docs(query, top_k=3):
    query_emb = model.encode(query)
    cur.execute("SELECT qa_id, embedding FROM medical_embeddings")
    all_embs = cur.fetchall()

    sims = []
    for qa_id, emb_bytes in all_embs:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        sims.append((qa_id, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
    ids = tuple([s[0] for s in sims])

    cur.execute("SELECT question, answer FROM medical_qa WHERE id IN %s;", (ids,))
    return cur.fetchall()

# --- Gemini Answer Function ---
def generate_answer(query):
    docs = get_similar_docs(query, top_k=3)
    if not docs:
        return "No relevant context found in the database."

    context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in docs])
    prompt = f"""
You are a knowledgeable medical assistant.
Use the context below to provide an accurate, concise answer.

Context:
{context}

Question: {query}
Answer:
"""
    model_gemini = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model_gemini.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else "No answer generated."

# --- Example Run ---
if __name__ == "__main__":
    q = "How to diagnose Gout"
    ans = generate_answer(q)
    print("\nUser:", q)
    print("\nAssistant:", ans)

