import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import text
from utils.db import SessionLocal

N_CLUSTERS = 20  # adjust or use elbow method later

def main():
    ids = np.load('/mnt/data/title_ids.npy')
    X = np.load('/mnt/data/title_embeds.npy')
    if len(ids) == 0:
        print('Nothing to cluster.')
        return
    kmeans = KMeans(n_clusters=min(N_CLUSTERS, len(ids)), n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    sess = SessionLocal()
    # Create events and map articles
    label_to_event = {}
    for lbl in np.unique(labels):
        centroid = kmeans.cluster_centers_[lbl]
        event_id = sess.execute(text("""
            INSERT INTO events (label, centroid_embed, time_window_start, time_window_stop, size)
            VALUES (:label, :centroid, NOW(), NOW(), :size)
            RETURNING id;
        """), {'label': f'kmeans_{lbl}', 'centroid': centroid.tolist(), 'size': int((labels==lbl).sum())}).scalar()
        label_to_event[int(lbl)] = event_id
    sess.commit()
    # Bridge table inserts
    for art_id, lbl in zip(ids.tolist(), labels.tolist()):
        sess.execute(text("""
            INSERT INTO article_events (article_id, event_id)
            VALUES (:aid, :eid)
            ON CONFLICT DO NOTHING;
        """), {'aid': int(art_id), 'eid': int(label_to_event[int(lbl)])})
    sess.commit()
    sess.close()
    print(f"Created {len(label_to_event)} events and linked {len(ids)} articles.")

if __name__ == '__main__':
    main()
