"""
Diagnose model false positives and show feature contributions.

Usage:
  python src/diagnose_model.py --model models/initial_model.pkl --data data/train.csv --max 10

This script loads the saved pipeline, evaluates it on a test split (same split
strategy as training), prints confusion matrix and finds false positives
(true=0, pred=1). For each false positive it prints the text, predicted
probability, and the top contributing TF-IDF features (words/phrases).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/initial_model.pkl')
    p.add_argument('--data', default='data/train.csv')
    p.add_argument('--max', type=int, default=10, help='Max false positives to show')
    return p.parse_args()


def top_contributing_features(pipeline, text, top_k=10):
    # Extract TF-IDF vector and classifier coef
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    X_vec = tfidf.transform([text])  # sparse
    try:
        coef = clf.coef_[0]
    except Exception:
        # Not a linear classifier
        return []

    # Compute per-feature contribution: tfidf_value * coef
    vec_arr = X_vec.toarray()[0]
    contrib = vec_arr * coef
    if np.all(contrib == 0):
        return []

    # Get top positive contributions (most pushing towards class 1/malicious)
    idx = np.argsort(-contrib)[:top_k]
    features = tfidf.get_feature_names_out()
    result = [(features[i], float(contrib[i]), float(vec_arr[i])) for i in idx if contrib[i] > 0]
    return result


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found at {args.model}")
        return
    if not os.path.exists(args.data):
        print(f"Data file not found at {args.data}")
        return

    print(f"Loading model from {args.model}...")
    pipeline = joblib.load(args.model)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")

    X = df['text'].astype(str)
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    print("Evaluating on test split...")
    probs = pipeline.predict_proba(X_test)
    if probs.shape[1] == 2:
        pred_scores = probs[:, 1]
    else:
        pred_scores = probs.max(axis=1)

    preds = (pred_scores >= 0.5).astype(int)

    print('\nClassification report:')
    print(classification_report(y_test, preds, digits=4))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, preds))

    # Find false positives: true label 0 but predicted 1
    fp_mask = (y_test.values == 0) & (preds == 1)
    fp_texts = X_test[fp_mask]
    fp_scores = pred_scores[fp_mask]

    n_fp = fp_texts.shape[0]
    print(f"\nFound {n_fp} false positives in the test split. Showing up to {args.max}:\n")

    for i, (text, score) in enumerate(zip(fp_texts[:args.max], fp_scores[:args.max])):
        print(f"--- False Positive #{i+1} (score={score:.4f}) ---")
        print(text)
        contribs = top_contributing_features(pipeline, text, top_k=15)
        if contribs:
            print('\nTop contributing features (feature, contribution, tfidf_value):')
            for feat, c, tf in contribs:
                print(f"  {feat:30s}  {c:8.4f}  tfidf={tf:.4f}")
        else:
            print('  (no linear contributions available)')
        print('\n')

    if n_fp == 0:
        print('No false positives found. Good job!')


if __name__ == '__main__':
    main()
