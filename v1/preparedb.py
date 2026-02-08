import sqlite3
import ollama
import time

db = sqlite3.connect('database.sqlite')
cursor = db.cursor()

def summarize_review(text):
    responses = []
    # If text is very long, chunk it
    if len(text) > 6000:
        chunk_size = 5000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        for chunk in chunks:
            prompt = f"Analyze this music review. Return ONLY a comma-separated list of: Genres, Emotions, Feelings, and Key Instruments. Review: {chunk}"
            response = ollama.generate(model="qwen2.5:7b", prompt=prompt, 
                                       options={"temperature": 0, "num_ctx": 8192})
            responses.append(response['response'])
        return ', '.join(responses)
    else:
        prompt = f"Analyze this music review. Return ONLY a comma-separated list of: Genres, Emotions, Feelings, and Key Instruments. Review: {text}"
        response = ollama.generate(model="qwen2.5:7b", prompt=prompt, 
                                   options={"temperature": 0, "num_ctx": 8192})
        return response['response']

def add_summary_to_db(summary, review_id):
    cursor.execute("UPDATE content SET tags = ? WHERE reviewid = ?", (summary, review_id))
    db.commit() # IMPORTANT: Save to disk
    return cursor.rowcount > 0

def fetch_next_review():
    # FIXED: Return both ID and Content
    cursor.execute("SELECT reviewid, content FROM content WHERE tags IS NULL LIMIT 1")
    return cursor.fetchone()

if __name__ == "__main__":
    i = 0
    print("Starting processing... Press Ctrl+C to stop safely.")
    
    try:
        while True:
            start_time = time.perf_counter()
            row = fetch_next_review()
            
            if not row:
                print("\nAll reviews processed!")
                break
                
            review_id, review_text = row
            tags = summarize_review(review_text)
            add_summary_to_db(tags, review_id)
            
            elapsed = time.perf_counter() - start_time
            print(f"[{i}] ID {review_id} updated in {elapsed:.2f}s")
            i += 1

    except KeyboardInterrupt:
        print("\n\n[!] Stop signal received (Ctrl+C).")
        print("Finishing current task and closing database...")
    
    finally:
        # This block runs NO MATTER WHAT (error, interrupt, or success)
        db.commit()
        db.close()
        print("Database connection closed. Progress saved.")
