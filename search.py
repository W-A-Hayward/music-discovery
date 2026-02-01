import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer

db = sqlite3.connect("database.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)
cursor = db.cursor()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
query = "electronic dance music that is reminiscent of childhood"
query_embedded = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
neighbour_count = 8

cursor.execute(
    """
    SELECT
        rv.distance,
        rv.rowid,
        r.title,
        r.artist,
        r.score,
        r.url
    FROM reviews_vec AS rv
    INNER JOIN content AS c ON rv.rowid = c.reviewid
    LEFT JOIN reviews AS r ON c.reviewid = r.reviewid
    WHERE rv.embedding MATCH ?
    AND k = ?
    ORDER BY rv.distance
    """,
    (query_embedded.tobytes(), neighbour_count),
)

rows = cursor.fetchall()
for row in rows:
    print(row)

db.close()  # Added: close the connection
