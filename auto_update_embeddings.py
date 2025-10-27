import time
import psycopg2
from sentence_transformers import SentenceTransformer
from apscheduler.schedulers.background import BackgroundScheduler


# DB connection
conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_F1T2wBfntjkZ@ep-dawn-surf-a43p87u0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
)
cursor = conn.cursor()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize consecutive no data counter and stop flag
consec_no_data = 0
stop_scheduler = False

# Scheduler initialization
scheduler = BackgroundScheduler()

def update_embeddings():
    global consec_no_data, stop_scheduler
    print("🔁 Checking for new data...")
    with open("extra_inputs.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("✅ No new data found.")
        consec_no_data += 1
        if consec_no_data >= 2:
            print("❌ No new data found twice consecutively. Stopping the job.")
            stop_scheduler = True  # Set flag instead of shutdown here
        return

    consec_no_data = 0  # reset counter if data found

    for line in lines:
        try:
            question, answer = line.strip().split("||")  # Q and A separated by ||
        except ValueError:
            continue  # skip invalid format lines

        # Check if question already exists
        cursor.execute("SELECT COUNT(*) FROM med_data WHERE question = %s", (question,))
        if cursor.fetchone()[0] == 0:
            embedding = model.encode(question + " " + answer).tolist()
            cursor.execute(
                "INSERT INTO med_data (question, answer, embedding) VALUES (%s, %s, %s)",
                (question, answer, embedding)
            )
            conn.commit()
            print(f"🆕 Added new QA: {question[:50]}...")

    # Clear the file after processing
    open("extra_inputs.txt", "w").close()
    print("✅ Update complete.\n")

# Schedule the update job every 5 minutes
scheduler.add_job(update_embeddings, "interval", minutes=0.5)
scheduler.start()

print("🚀 Auto update process started. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(10)
        if stop_scheduler:
            print("🛑 Stopping scheduler from main loop.")
            break
except KeyboardInterrupt:
    pass

# Shutdown scheduler and DB connection safely outside the job thread
scheduler.shutdown()
conn.close()
print("🛑 Scheduler stopped and connection closed.")

