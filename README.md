# RAG-Based Medical Q&A System

🩺 Project Overview:

This project is my first hands-on attempt at building a Retrieval-Augmented Generation (RAG) system using the MedQuAD dataset — a medical question–answer dataset about diseases and symptoms.

---

## 🧱 How It Works:

Data Loading – Load the MedQuAD dataset and clean the Q&A data.

Database Setup – Store the questions and answers in Neon PostgreSQL.

Embeddings Generation – Create vector embeddings for each question and answer pair.

Retrieval Step – When a user asks a query, the system retrieves the most relevant medical answers.

Response Generation – The gemini LLM uses the retrieved context to generate an accurate medical response.

Automated New Data Ingestion – A background job checks extra_inputs.txt every few minutes, adds new Q&A pairs by creating embeddings and updating the med_data table, then clears the file automatically.

---

## 🖼️ Some Screenshots of Output :
<img width="1665" height="947" alt="Screenshot 2025-10-27 160704" src="https://github.com/user-attachments/assets/c56028cc-9d61-4487-bc63-e90f9cf16834" />

<img width="1667" height="930" alt="Screenshot 2025-10-27 160941" src="https://github.com/user-attachments/assets/02a3de79-567f-47e7-8dbe-ac7e5ae0d144" />

<img width="1687" height="938" alt="Screenshot 2025-10-27 161224" src="https://github.com/user-attachments/assets/81093e67-24ec-4fa5-b544-1d787554b3bc" />

---

## 🚀 Future Improvements:

✅ Streamlit Web Interface – Add a front-end where users can ask medical questions interactively.

✅ Model Fine-Tuning – Improve the Gemma LLM with domain-specific medical text.

✅ Dockerization & Deployment – Package the app for easy cloud deployment.

✅ Prompt Optimization – Experiment with few-shot examples for more reliable RAG responses.

---

Data Set Link : https://www.kaggle.com/datasets/jpmiller/layoutlm

The goal is to explore how RAG helps improve the accuracy of Large Language Models (LLMs) by retrieving relevant data from a structured database before generating an answer

