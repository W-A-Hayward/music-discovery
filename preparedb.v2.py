import sqlite3
import asyncio
import time
from ollama import AsyncClient

# Database setup
DB_PATH = 'database.sqlite'
MODEL_NAME = "qwen2.5:14b"
BATCH_SIZE = 4  

async def summarize_review(client, text):
    """Summarizes text, handling long reviews by chunking."""
    if len(text) > 6000:
        # Simple chunking for very long reviews
        text = text[:6000] # For speed, we'll just trim to the 'meat' of the review
        
    prompt = f"Analyze this music review. Return 100 word a comma-separated list of: Genres and sub-genres, Emotions, and Key Instruments. Add a 50 word summary of the review at the end. Review: {text}"
    
    try:
        response = await client.generate(
            model=MODEL_NAME, 
            prompt=prompt, 
            options={"temperature": 0, "num_ctx": 8192}
        )
        return response['response']
    except Exception as e:
        return f"Error: {str(e)}"

async def process_review(client, db, row):
    """Processes a single review: AI Gen -> DB Update."""
    print(f"Processing review of len {len(row[1])}")
    review_id, content = row
    tags = await summarize_review(client, content)
    
    # Use a standard synchronous update since SQLite is fast
    cursor = db.cursor()
    cursor.execute("UPDATE content SET tags = ? WHERE reviewid = ?", (tags, review_id))
    db.commit()
    return review_id

async def main():
    db = sqlite3.connect(DB_PATH)
    client = AsyncClient()
    count = 0
    
    print(f"--- Starting Parallel Processing (Batch Size: {BATCH_SIZE}) ---")
    
    try:
        while True:
            # Wait for a bit to let the GPU cool down
            await asyncio.sleep(10)

            # 1. Fetch a batch of reviews that haven't been tagged yet
            cursor = db.cursor()
            cursor.execute(
                "SELECT reviewid, content FROM content WHERE tags IS NULL LIMIT ?", 
                (BATCH_SIZE,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                print("All reviews processed!")
                break
            
            start_time = time.perf_counter()
            
            # 2. Run the whole batch in parallel on the GPU
            tasks = [process_review(client, db, row) for row in rows]
            completed_ids = await asyncio.gather(*tasks)
            
            elapsed = time.perf_counter() - start_time
            print(f"Batch {completed_ids} finished in {elapsed:.2f}s ({(len(rows)/elapsed):.2f} reviews/sec)")
            count += 4
    except KeyboardInterrupt:
        print("\n[!] Stopping safely...")
    finally:
        db.commit()
        db.close()
        print("Database closed.")

if __name__ == "__main__":
    asyncio.run(main())
