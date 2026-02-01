import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

db = sqlite3.connect("database.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)
cursor = db.cursor()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")


cursor.execute("""
    DROP TABLE IF EXISTS reviews_vec
""")

input = input("continue ?")
if input == "n":
    exit(1)

cursor.execute("""
    CREATE VIRTUAL TABLE reviews_vec USING vec0(embedding float[768])
""")

cursor.execute("SELECT DISTINCT reviewid, content FROM content")
rows = cursor.fetchall()

print(rows[0][1])

vectors = model.encode(
    [rows[i][1] for i in range(len(rows))],
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)

data = [(rows[i][0], vectors[i].tobytes()) for i in range(len(rows))]

cursor.executemany("INSERT INTO reviews_vec(rowid, embedding) VALUES (?, ?)", data)

db.commit()
db.close()
