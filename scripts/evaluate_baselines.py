# Week 1 baseline evaluation scaffold.
# Replace TODOs with your dataset loaders (e.g., FNC-1, SemEval Task 6, MFC).
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'

def load_fake_dataset():
    # Placeholder: titles + stance labels in {'pro','neutral','anti'}
    X_text = [
        'Policy boosts economy, says minister',
        'Critics slam policy over risks',
        'Debate continues on policy impact',
        'Experts support new measures',
        'Opposition warns of consequences'
    ]
    y = np.array(['pro','anti','neutral','pro','anti'])
    return X_text, y

def encode(texts):
    model = SentenceTransformer(EMBEDDER)
    return model.encode(texts, normalize_embeddings=True)

def run_stance_baseline():
    X_text, y = load_fake_dataset()
    X = encode(X_text)
    clf = LogisticRegression(max_iter=200, class_weight='balanced', n_jobs=None)
    clf.fit(X, y)
    yhat = clf.predict(X)
    print('Macro-F1:', f1_score(y, yhat, average='macro'))
    print(classification_report(y, yhat))
    print(confusion_matrix(y, yhat))

if __name__ == '__main__':
    run_stance_baseline()
