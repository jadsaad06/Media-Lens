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
import hashlib
import random
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
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV

SEED = int(os.getenv('SEED', '42'))
random.seed(SEED)
np.random.seed(SEED)

EMBEDDER = os.getenv('EVAL_EMBEDDER', 'sentence-transformers/all-MiniLM-L6-v2')
DATA_DIR = Path(os.getenv('EVAL_DATA_DIR', './data'))
OUT_DIR = Path(os.getenv('EVAL_OUT_DIR', './outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(os.getenv('REPORTS_DIR', './reports'))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
STANCE_CALIBRATE = os.getenv('STANCE_CALIBRATE', '1') == '1'
FRAMING_TUNE_THRESHOLDS = os.getenv('FRAMING_TUNE_THRESHOLDS', '1') == '1'
GLOBAL_FRAMING_THRESHOLD = float(os.getenv('FRAMING_THRESHOLD', '0.3'))
FRAMING_EMBEDDINGS_ONLY = os.getenv('FRAMING_EMBEDDINGS_ONLY', '0') == '1'

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

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

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
    # Calibrate if needed to obtain probabilities (useful for reliability, abstention, etc.)
    if best_name == 'linsvc' and STANCE_CALIBRATE:
        svc = LinearSVC(class_weight='balanced', C=best_C)
        cal = CalibratedClassifierCV(svc, method='sigmoid', cv=3)
        cal.fit(X_train, y_train_enc)
        best_model = cal
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
    # Persist model for reuse
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump({'model': best_model, 'label_encoder': le, 'embedder': EMBEDDER}, OUT_DIR / 'stance_model.joblib')

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
    # Append dataset manifest info
    try:
        st_path, bo_path = ensure_fnc1_local(DATA_DIR)
        params['dataset'] = {
            'fnc1_train_stances.csv': {
                'path': str(st_path),
                'sha256': _sha256_file(st_path)
            },
            'fnc1_train_bodies.csv': {
                'path': str(bo_path),
                'sha256': _sha256_file(bo_path)
            }
        }
    except Exception:
        pass
    (REPORTS_DIR / 'params.json').write_text(json.dumps(params, indent=2))

# ----------------------
# Framing Baseline
# ----------------------

KEYWORD_TAGS: Dict[str, List[str]] = {
    'economy': [
        'economy','economic','inflation','deflation','jobs','employment','market','markets','stock','stocks',
        'gdp','unemployment','wage','wages','salary','salaries','payroll','tax','taxes','budget','deficit','surplus',
        'interest rate','rates','fed','federal reserve','central bank','bond','bonds','recession','growth'
    ],
    'immigration': ['immigration','migrant','migrants','asylum','border','visa','visas','refugee','refugees','deport'],
    'health': [
        'health','hospital','covid','vaccine','vaccination','healthcare','disease','doctor','doctors','nurse','medical',
        'public health','clinic','infection','outbreak','pandemic'
    ],
    'education': [
        'school','schools','education','educational','university','universities','college','student','students','teacher','teachers',
        'curriculum','classroom','tuition','loan','loans','debt','degree'
    ],
    'environment': [
        'climate','environment','emission','emissions','carbon','wildfire','wildfires','hurricane','flood','drought','heatwave',
        'greenhouse','pollution','renewable','solar','wind','sustainability'
    ],
    'foreign_policy': ['war','ukraine','israel','gaza','foreign','diplomat','diplomacy','sanction','sanctions','nato','russia','china','iran','alliance'],
    'crime': ['crime','police','murder','assault','theft','burglary','robbery','court','trial','arrest','charges','homicide','violence'],
    'sports': ['match','game','tournament','league','score','player','players','coach','team','season','final','cup']
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
    all_labels = sorted(KEYWORD_TAGS.keys())
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_all = mlb.fit_transform(tags_sup)
    # train/val/test split (avoid tuning on test)
    X_tmp, X_test_txt, Y_tmp, Y_test_all = train_test_split(list(texts_sup), Y_all, test_size=0.2, random_state=42)
    X_train_txt, X_val_txt, Y_train_all, Y_val_all = train_test_split(X_tmp, Y_tmp, test_size=0.2, random_state=42)
    # Drop constant columns (all 0s or all 1s) in training to avoid degenerate estimators
    col_sums = Y_train_all.sum(axis=0)
    keep_mask = (col_sums > 0) & (col_sums < Y_train_all.shape[0])
    if keep_mask.sum() == 0:
        print('All framing tags are constant; expand KEYWORD_TAGS or data.')
        return
    keep_idx = np.where(keep_mask)[0]
    kept_labels = [all_labels[i] for i in keep_idx]
    Y_train = Y_train_all[:, keep_mask]
    Y_val = Y_val_all[:, keep_mask]
    Y_test = Y_test_all[:, keep_mask]
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,3))
    X_train = vectorizer.fit_transform(X_train_txt)
    X_val = vectorizer.transform(X_val_txt)
    X_test = vectorizer.transform(X_test_txt)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced'))
    clf.fit(X_train, Y_train)
    # Predict probabilities (preferred) or scores on VAL for threshold tuning
    if hasattr(clf, 'predict_proba'):
        val_scores_tfidf = clf.predict_proba(X_val)
        is_proba = True
    else:
        val_scores_tfidf = clf.decision_function(X_val)
        is_proba = False
    # SBERT embeddings pathway (for embeddings-only or ensemble)
    ensemble = os.getenv('FRAMING_ENSEMBLE', '0') == '1'
    use_emb = FRAMING_EMBEDDINGS_ONLY or ensemble
    emb_model = None
    if use_emb:
        emb_model = SentenceTransformer(EMBEDDER)
        X_train_emb = emb_model.encode(X_train_txt, normalize_embeddings=True)
        X_val_emb = emb_model.encode(X_val_txt, normalize_embeddings=True)
        X_test_emb = emb_model.encode(X_test_txt, normalize_embeddings=True)
        emb_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced'))
        emb_clf.fit(X_train_emb, Y_train)
        if hasattr(emb_clf, 'predict_proba'):
            val_scores_emb = emb_clf.predict_proba(X_val_emb)
        else:
            val_scores_emb = emb_clf.decision_function(X_val_emb)
    # Choose scores for threshold tuning
    if FRAMING_EMBEDDINGS_ONLY:
        val_scores = np.asarray(val_scores_emb)
    elif ensemble and use_emb:
        val_scores = (np.asarray(val_scores_tfidf) + np.asarray(val_scores_emb)) / 2.0
    else:
        val_scores = np.asarray(val_scores_tfidf)

    thresholds = []
    if FRAMING_TUNE_THRESHOLDS:
        # Tune per-tag thresholds on VAL
        for j in range(Y_val.shape[1]):
            best_f1, best_t = 0.0, 0.5
            grid = np.linspace(0.2, 0.8, 25) if is_proba else np.linspace(np.min(val_scores[:, j]), np.max(val_scores[:, j]), 25)
            for t in grid:
                yj = (val_scores[:, j] >= t).astype(int)
                f1j = f1_score(Y_val[:, j], yj, zero_division=0)
                if f1j > best_f1:
                    best_f1, best_t = f1j, t
            thresholds.append(float(best_t))
    else:
        # Use a single global threshold (fast path, approximates earlier behavior)
        for _ in range(Y_val.shape[1]):
            thresholds.append(float(GLOBAL_FRAMING_THRESHOLD if is_proba else 0.0))

    # Evaluate on TEST with tuned thresholds
    # Compute test scores for selected variant(s)
    if hasattr(clf, 'predict_proba'):
        test_scores_tfidf = clf.predict_proba(X_test)
    else:
        test_scores_tfidf = clf.decision_function(X_test)
    test_scores = np.asarray(test_scores_tfidf)
    test_scores_emb = None
    if use_emb:
        if hasattr(emb_clf, 'predict_proba'):
            test_scores_emb = emb_clf.predict_proba(X_test_emb)
        else:
            test_scores_emb = emb_clf.decision_function(X_test_emb)
        test_scores_emb = np.asarray(test_scores_emb)
    if FRAMING_EMBEDDINGS_ONLY and test_scores_emb is not None:
        test_scores = test_scores_emb
    elif ensemble and test_scores_emb is not None:
        test_scores = (test_scores + test_scores_emb) / 2.0

    Y_pred = np.zeros_like(Y_test)
    for j, t in enumerate(thresholds):
        Y_pred[:, j] = (test_scores[:, j] >= t).astype(int)
    # Fallback: ensure at least one tag per sample
    for i in range(Y_pred.shape[0]):
        if Y_pred[i].sum() == 0:
            Y_pred[i, int(np.argmax(test_scores[i]))] = 1
    # Evaluate only on samples that still have at least one true label after label filtering
    nonempty_mask = (Y_test.sum(axis=1) > 0).A1 if hasattr(Y_test, 'A1') else (Y_test.sum(axis=1) > 0)
    if nonempty_mask.sum() == 0:
        print('No non-empty ground-truth samples in test after label filtering; cannot compute framing metrics.')
        return
    Y_test_eval = Y_test[nonempty_mask]
    Y_pred_eval = Y_pred[nonempty_mask]
    micro_f1 = f1_score(Y_test_eval, Y_pred_eval, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_test_eval, Y_pred_eval, average='macro', zero_division=0)
    subset_acc = accuracy_score(Y_test_eval, Y_pred_eval)
    print('Framing Micro-F1:', micro_f1)
    print('Framing Subset Accuracy:', subset_acc)
    # Save report (reports/)
    (REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'framing_report.txt').write_text(
        f"Micro-F1={micro_f1:.3f}\nMacro-F1={macro_f1:.3f}\nSubsetAcc={subset_acc:.3f}\nLabels={kept_labels}\nThresholds={thresholds}\nTestSamples={int(Y_test.shape[0])}\nEvaluatedSamples={int(nonempty_mask.sum())}\n"
    )
    # Write per-variant comparison if available
    comparisons: Dict[str, Dict[str, float]] = {}
    # TF-IDF only metrics
    Y_pred_tfidf = np.zeros_like(Y_test)
    for j, t in enumerate(thresholds):
        Y_pred_tfidf[:, j] = (test_scores_tfidf[:, j] >= t).astype(int)
    Y_pred_tfidf_eval = Y_pred_tfidf[nonempty_mask]
    comparisons['tfidf'] = {
        'micro_f1': float(f1_score(Y_test_eval, Y_pred_tfidf_eval, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(Y_test_eval, Y_pred_tfidf_eval, average='macro', zero_division=0)),
        'subset_accuracy': float(accuracy_score(Y_test_eval, Y_pred_tfidf_eval))
    }
    if test_scores_emb is not None:
        Y_pred_emb = np.zeros_like(Y_test)
        for j, t in enumerate(thresholds):
            Y_pred_emb[:, j] = (test_scores_emb[:, j] >= t).astype(int)
        Y_pred_emb_eval = Y_pred_emb[nonempty_mask]
        comparisons['sbert'] = {
            'micro_f1': float(f1_score(Y_test_eval, Y_pred_emb_eval, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(Y_test_eval, Y_pred_emb_eval, average='macro', zero_division=0)),
            'subset_accuracy': float(accuracy_score(Y_test_eval, Y_pred_emb_eval))
        }
        if ensemble:
            comparisons['ensemble'] = {
                'micro_f1': float(micro_f1),
                'macro_f1': float(macro_f1),
                'subset_accuracy': float(subset_acc)
            }
    (REPORTS_DIR / 'framing_metrics.json').write_text(json.dumps(comparisons, indent=2))
    # Support counts report
    support = {
        'labels': kept_labels,
        'train_counts': [int(x) for x in Y_train.sum(axis=0).A1] if hasattr(Y_train, 'A1') else [int(x) for x in Y_train.sum(axis=0)],
        'val_counts': [int(x) for x in Y_val.sum(axis=0).A1] if hasattr(Y_val, 'A1') else [int(x) for x in Y_val.sum(axis=0)],
        'test_counts': [int(x) for x in Y_test.sum(axis=0).A1] if hasattr(Y_test, 'A1') else [int(x) for x in Y_test.sum(axis=0)]
    }
    (REPORTS_DIR / 'framing_support.json').write_text(json.dumps(support, indent=2))
    # Per-tag classification report
    (REPORTS_DIR / 'framing_classification_report.txt').write_text(
        classification_report(Y_test_eval, Y_pred_eval, target_names=kept_labels, zero_division=0)
    )
    # Persist framing model and vectorizer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    framing_bundle = {
        'vectorizer': vectorizer,
        'clf': clf,
        'thresholds': thresholds,
        'labels': kept_labels,
        'ensemble': ensemble,
        'emb_model': EMBEDDER if use_emb else None,
        'emb_clf': emb_clf if use_emb else None
    }
    dump(framing_bundle, OUT_DIR / 'framing_model.joblib')
    # Update params.json with framing section
    params_path = REPORTS_DIR / 'params.json'
    if params_path.exists():
        params = json.loads(params_path.read_text())
    else:
        params = {'embedder': EMBEDDER, 'seed': 42}
    params['framing'] = {
        'vectorizer': 'tfidf(1,3)',
        'base_model': 'one-vs-rest logistic',
        'class_weight': 'balanced',
        'thresholds': thresholds,
        'labels': kept_labels,
        'ensemble': ensemble,
        'embeddings_only': FRAMING_EMBEDDINGS_ONLY,
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'subset_accuracy': float(subset_acc)
    }
    params_path.write_text(json.dumps(params, indent=2))

if __name__ == '__main__':
    run_stance_baseline()
    run_framing_baseline()
