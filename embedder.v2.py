import sqlite3
import sqlite_vec
import torch
from sentence_transformers import SentenceTransformer

db = sqlite3.connect("database.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)
cursor = db.cursor()

print(f"Is ROCm/CUDA available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

enable_drop = input("Enable DROP TABLE? [y/n] ")
if enable_drop == "y":
    cursor.execute("""
        DROP TABLE IF EXISTS reviews_vec
    """)

    cursor.execute("""
        CREATE VIRTUAL TABLE reviews_vec USING vec0(embedding float[768])
    """)

cursor.execute("SELECT rowid, tags, chunk FROM review_tags")
rows = cursor.fetchall()

vectors = model.encode(
        ["Tags: " + str(rows[i][1]) + "\n" + "Review: " + str(rows[i][2]) 
         for i in range(len(rows))],
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)

data = [(rows[i][0], vectors[i].tobytes()) for i in range(len(rows))]

cursor.executemany("INSERT INTO reviews_vec(rowid, embedding) VALUES (?, ?)", data)

db.commit()
db.close()
