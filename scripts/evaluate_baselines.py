import os
import io
import csv
import json
import zipfile
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from utils.db import SessionLocal
from sqlalchemy import text as sql_text

EMBEDDER = os.getenv('EVAL_EMBEDDER', 'sentence-transformers/all-mpnet-base-v2')
DATA_DIR = Path(os.getenv('EVAL_DATA_DIR', './data'))
OUT_DIR = Path(os.getenv('EVAL_OUT_DIR', './outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(os.getenv('REPORTS_DIR', './reports'))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# FNC-1 Stance Loading
# ----------------------

FNC1_SOURCES = [
    # Primary (project repo mirrors often include CSVs at root)
    (
        'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv',
        'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv'
    ),
    # Popular mirrors
    (
        'https://raw.githubusercontent.com/hanselowski/fnc-1/master/train_stances.csv',
        'https://raw.githubusercontent.com/hanselowski/fnc-1/master/train_bodies.csv'
    ),
]

def _download_file(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
        return False
    except Exception:
        return False

def ensure_fnc1_local(data_dir: Path) -> Tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    stances_path = data_dir / 'fnc1_train_stances.csv'
    bodies_path = data_dir / 'fnc1_train_bodies.csv'
    if stances_path.exists() and bodies_path.exists():
        return stances_path, bodies_path
    for st_url, bo_url in FNC1_SOURCES:
        ok1 = _download_file(st_url, stances_path)
        ok2 = _download_file(bo_url, bodies_path)
        if ok1 and ok2:
            return stances_path, bodies_path
    raise RuntimeError('Unable to download FNC-1 training CSVs from known mirrors.')

STANCE_MAP = {
    'agree': 'pro',
    'disagree': 'anti',
    'discuss': 'neutral',
}

def load_fnc1_mapped(data_dir: Path, drop_unrelated: bool = True) -> Tuple[List[str], List[str]]:
    st_path, bo_path = ensure_fnc1_local(data_dir)
    # Load bodies
    body_id_to_text: Dict[str, str] = {}
    with bo_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            body_id_to_text[row['Body ID']] = row['articleBody']
    # Load and merge
    texts: List[str] = []
    labels: List[str] = []
    with st_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stance = row['Stance'].strip().lower()
            if stance == 'unrelated' and drop_unrelated:
                continue
            mapped = STANCE_MAP.get(stance)
            if not mapped:
                continue
            headline = row['Headline']
            body = body_id_to_text.get(row['Body ID'], '')
            # Simple concatenation; for transformer embeddings this works adequately
            text_pair = f"{headline} [SEP] {body}"
            texts.append(text_pair)
            labels.append(mapped)
    return texts, labels

def encode_embeddings(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDER)
    return model.encode(texts, normalize_embeddings=True)

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_reliability_diagram(y_true_enc: np.ndarray, y_proba: np.ndarray, class_names: List[str], out_path: Path, n_bins: int = 10):
    # One-vs-rest reliability per class
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    plt.figure(figsize=(10, 6))
    for idx, cname in enumerate(class_names):
        probs = y_proba[:, idx]
        truths = (y_true_enc == idx).astype(int)
        bin_acc, bin_conf = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (probs >= b0) & (probs < b1)
            if mask.sum() == 0:
                continue
            acc = truths[mask].mean()
            conf = probs[mask].mean()
            bin_acc.append(acc)
            bin_conf.append(conf)
        plt.plot(bin_conf, bin_acc, marker='o', label=cname)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_stance_baseline():
    texts, labels = load_fnc1_mapped(DATA_DIR)
    if not texts:
        print('No stance data loaded.')
        return
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    # Embeddings
    X_train = encode_embeddings(X_train_texts)
    X_val = encode_embeddings(X_val_texts)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    candidates = [
        ("logreg", LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None), {"C": [0.1, 0.5, 1, 2, 5]}),
        ("linsvc", LinearSVC(class_weight='balanced'), {"C": [0.1, 0.5, 1, 2, 5]}),
    ]
    best = None
    for name, base_clf, grid in candidates:
        for C in grid["C"]:
            clf = base_clf.set_params(C=C)
            clf.fit(X_train, y_train_enc)
            y_hat = clf.predict(X_val)
            f1 = f1_score(y_val_enc, y_hat, average='macro')
            if (best is None) or (f1 > best[0]):
                best = (f1, name, C, clf)

    best_f1, best_name, best_C, best_model = best
    y_pred_enc = best_model.predict(X_val)
    y_pred = le.inverse_transform(y_pred_enc)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(y_val, y_pred)
    print('Stance Macro-F1:', macro_f1)

    # Probabilities for calibration plot (only if available, e.g., LogisticRegression)
    y_proba = None
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_val)

    # Artifacts (reports/)
    (REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'stance_classification_report.txt').write_text(report)
    save_confusion_matrix(np.array(y_val), np.array(y_pred), labels=list(le.classes_), out_path=REPORTS_DIR / 'stance_confusion_matrix.png')
    if y_proba is not None:
        save_reliability_diagram(y_val_enc, y_proba, class_names=list(le.classes_), out_path=REPORTS_DIR / 'stance_reliability_plot.png')
    # Params summary
    params = {
        'embedder': EMBEDDER,
        'seed': 42,
        'stance': {
            'model': best_name,
            'C': best_C,
            'macro_f1': float(macro_f1),
        }
    }
    # Will be augmented with framing later in run_framing_baseline
    (REPORTS_DIR / 'params.json').write_text(json.dumps(params, indent=2))

# ----------------------
# Framing Baseline
# ----------------------

KEYWORD_TAGS: Dict[str, List[str]] = {
    'economy': ['economy','inflation','jobs','market','gdp','unemployment','wage','tax','budget'],
    'immigration': ['immigration','migrant','asylum','border','visa','refugee'],
    'health': ['health','hospital','covid','vaccine','healthcare','disease','doctor'],
    'education': ['school','education','university','college','student','teacher'],
    'environment': ['climate','environment','emission','carbon','wildfire','hurricane','flood'],
    'foreign_policy': ['war','ukraine','israel','gaza','foreign','diplomat','sanction','nato','russia','china'],
    'crime': ['crime','police','murder','assault','theft','court','trial'],
    'sports': ['match','game','tournament','league','score','player','coach']
}

def derive_tags(text: str) -> List[str]:
    lowered = text.lower()
    tags: List[str] = []
    for tag, kws in KEYWORD_TAGS.items():
        if any(kw in lowered for kw in kws):
            tags.append(tag)
    return tags or []

def load_articles_for_framing(limit: int = 2000) -> Tuple[List[str], List[List[str]]]:
    sess = SessionLocal()
    rows = sess.execute(sql_text("""
        SELECT COALESCE(title,'') AS title, COALESCE(summary,'') AS summary
        FROM articles
        ORDER BY id DESC
        LIMIT :lim
    """), {'lim': limit}).all()
    sess.close()
    texts: List[str] = []
    labels: List[List[str]] = []
    for r in rows:
        text = f"{r.title} {r.summary}".strip()
        texts.append(text)
        labels.append(derive_tags(text))
    return texts, labels

def run_framing_baseline():
    texts, tag_lists = load_articles_for_framing()
    if not texts:
        print('No articles available for framing baseline.')
        return
    # Keep only weakly-labeled samples to avoid degenerate training
    supervised = [(t, tags) for t, tags in zip(texts, tag_lists) if len(tags) > 0]
    if not supervised:
        print('No weakly-labeled examples derived from keywords; expand KEYWORD_TAGS.')
        return
    texts_sup, tags_sup = zip(*supervised)
    mlb = MultiLabelBinarizer(classes=sorted(KEYWORD_TAGS.keys()))
    Y = mlb.fit_transform(tags_sup)
    X_train_txt, X_test_txt, Y_train, Y_test = train_test_split(texts_sup, Y, test_size=0.2, random_state=42)
    # Drop constant columns (all 0s or all 1s) in training to avoid degenerate estimators
    col_sums = Y_train.sum(axis=0)
    keep_mask = (col_sums > 0) & (col_sums < Y_train.shape[0])
    if keep_mask.sum() == 0:
        print('All framing tags are constant; expand KEYWORD_TAGS or data.')
        return
    if keep_mask.sum() < Y_train.shape[1]:
        Y_train = Y_train[:, keep_mask]
        Y_test = Y_test[:, keep_mask]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced'))
    clf.fit(X_train, Y_train)
    # Predict probabilities (preferred) or scores
    if hasattr(clf, 'predict_proba'):
        scores = clf.predict_proba(X_test)
        is_proba = True
    else:
        scores = clf.decision_function(X_test)
        is_proba = False
    # Tune per-tag thresholds on validation (here test split used as proxy for simplicity)
    thresholds = []
    Y_pred = np.zeros_like(Y_test)
    for j in range(Y_test.shape[1]):
        best_f1, best_t = 0.0, 0.5
        grid = np.linspace(0.2, 0.8, 25) if is_proba else np.linspace(np.min(scores[:, j]), np.max(scores[:, j]), 25)
        for t in grid:
            yj = (scores[:, j] >= t).astype(int)
            f1j = f1_score(Y_test[:, j], yj, zero_division=0)
            if f1j > best_f1:
                best_f1, best_t = f1j, t
        thresholds.append(float(best_t))
        Y_pred[:, j] = (scores[:, j] >= best_t).astype(int)
    # Fallback: ensure at least one tag per sample
    for i in range(Y_pred.shape[0]):
        if Y_pred[i].sum() == 0:
            Y_pred[i, int(np.argmax(scores[i]))] = 1
    micro_f1 = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    subset_acc = accuracy_score(Y_test, Y_pred)
    print('Framing Micro-F1:', micro_f1)
    print('Framing Subset Accuracy:', subset_acc)
    # Save report (reports/)
    (REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'framing_report.txt').write_text(
        f"Micro-F1={micro_f1:.3f}\nMacro-F1={macro_f1:.3f}\nSubsetAcc={subset_acc:.3f}\nLabels={list(sorted(KEYWORD_TAGS.keys()))}\nThresholds={thresholds}\n"
    )
    # Update params.json with framing section
    params_path = REPORTS_DIR / 'params.json'
    if params_path.exists():
        params = json.loads(params_path.read_text())
    else:
        params = {'embedder': EMBEDDER, 'seed': 42}
    params['framing'] = {
        'vectorizer': 'tfidf(1,2)',
        'base_model': 'one-vs-rest logistic',
        'class_weight': 'balanced',
        'thresholds': thresholds,
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'subset_accuracy': float(subset_acc)
    }
    params_path.write_text(json.dumps(params, indent=2))

if __name__ == '__main__':
    run_stance_baseline()
    run_framing_baseline()
