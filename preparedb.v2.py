# This script is a modified version of the original script preparedb.py
# It is needed to prepare the database for accurate and fast vector embedding and search
# We use the GPU to chunk the reviews text and generate the tags in parallel.

import sqlite3
import asyncio
import time
from ollama import AsyncClient

# Database & Model Config
DB_PATH = 'database.sqlite'
MODEL_NAME = "qwen2.5:7b"
BATCH_SIZE = 10 
CHUNK_SIZE = 1200
OVERLAP = 200

async def get_tags(client, text):
    """Fetches tags for a specific text segment."""
    prompt = (
        "Summarize the music review segment below. "
        "Return ONLY a comma-separated list of: Emotions, Feelings, Instruments, and effects. "
        f"Review segment: {text}"
    )
    try:
        response = await client.generate(
            model=MODEL_NAME, 
            prompt=prompt, 
            options={"temperature": 0, "num_ctx": 4096}
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

async def get_tags_handler(client, text):
    """Splits text into chunks and returns tags + start positions."""
    all_chunks_results = []
    chunk_start_indexes = []

    # If text is long, chunk it
    if len(text) > CHUNK_SIZE:
        step = max(1, CHUNK_SIZE - OVERLAP)
        for i in range(0, len(text), step):
            chunk = text[i : i + CHUNK_SIZE]
            if len(chunk) < 100 and i > 0: break
            
            tags = await get_tags(client, chunk)
            if tags:
                all_chunks_results.append(tags)
                chunk_start_indexes.append(i)
    else:
        # For small reviews, just do one chunk starting at index 0
        tags = await get_tags(client, text)
        if tags:
            all_chunks_results.append(tags)
            chunk_start_indexes.append(0)

    return all_chunks_results, chunk_start_indexes

async def process_review(client, db, row):
    """Generates tags and saves chunk text + tags to the database."""
    review_id, content = row
    
    # Unpack tags and their corresponding starting positions
    tags_list, start_indexes = await get_tags_handler(client, content)
    
    if tags_list:
        # Prepare all chunk rows for this review for a bulk insert
        db_rows = []
        for tags, start_idx in zip(tags_list, start_indexes):
            chunk_text = content[start_idx : start_idx + CHUNK_SIZE]
            db_rows.append((review_id, tags, chunk_text))
        
        # Fast bulk insert
        cursor = db.cursor()
        cursor.executemany(
            "INSERT INTO review_tags (reviewid, tags, chunk) VALUES (?, ?, ?)", 
            db_rows
        )
        db.commit()
    
    return review_id

def stream_reviews(db, batch_size):
    """Generator to pull unprocessed reviews in batches."""
    cursor = db.cursor()
    query = """
        SELECT reviewid, content FROM content 
        WHERE reviewid NOT IN (SELECT DISTINCT reviewid FROM review_tags)
    """
    cursor.execute(query)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows: break
        yield rows

async def main():
    db = sqlite3.connect(DB_PATH)
    client = AsyncClient()
    total_processed = 0
    
    print(f"--- 6800 XT Batch Processing Started ---")
    
    try:
        for batch in stream_reviews(db, BATCH_SIZE):
            start_time = time.perf_counter()
            
            # Parallel process the batch
            tasks = [process_review(client, db, row) for row in batch]
            await asyncio.gather(*tasks)
            
            elapsed = time.perf_counter() - start_time
            total_processed += len(batch)
            print(f"Batch Done | Total: {total_processed} | {len(batch)/elapsed:.2f} reviews/s")
            
    except KeyboardInterrupt:
        print("\n[!] Stopping safely...")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
