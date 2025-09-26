import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from utils.db import SessionLocal

load_dotenv()
EMBEDDER = os.getenv('EMBEDDER','sentence-transformers/all-MiniLM-L6-v2')
DATA_DIR = os.getenv('DATA_DIR', './data')

def fetch_titles(sess, limit=1000):
    rows = sess.execute(text("SELECT id, COALESCE(title,'') AS title FROM articles ORDER BY id DESC LIMIT :lim"), {'lim': limit}).all()
    return [(r.id, r.title) for r in rows]

def save_event(sess, centroid, label=None):
    sql = text("""
        INSERT INTO events (label, centroid_embed, time_window_start, time_window_stop, size)
        VALUES (:label, :centroid, NOW(), NOW(), :size)
        RETURNING id;
    """)
    return sess.execute(sql, {'label': label, 'centroid': centroid.tolist(), 'size': 0}).scalar()

def main():
    model = SentenceTransformer(EMBEDDER)
    sess = SessionLocal()
    data = fetch_titles(sess)
    ids, titles = zip(*data) if data else ([], [])
    if not titles:
        print("No titles to embed.")
        return
    X = model.encode(list(titles), normalize_embeddings=True)
    # Save embeddings temporarily to disk for clustering step
    os.makedirs(DATA_DIR, exist_ok=True)
    ids_path = os.path.join(DATA_DIR, 'title_ids.npy')
    embeds_path = os.path.join(DATA_DIR, 'title_embeds.npy')
    np.save(ids_path, np.array(ids))
    np.save(embeds_path, X.astype(np.float32))
    print(f"Embedded {len(ids)} titles to {embeds_path}")
    sess.close()

if __name__ == '__main__':
    main()
