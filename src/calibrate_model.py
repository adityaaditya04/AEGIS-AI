"""
Calibrate an existing trained pipeline using a validation split and
`CalibratedClassifierCV` (isotonic or sigmoid/Platt). Saves a new
pipeline at `models/calibrated_model.pkl`.

Usage:
  python src/calibrate_model.py --model models/initial_model.pkl --data data/train.csv --method isotonic

Note: isotonic requires more data and can overfit; `sigmoid` (Platt) is
more conservative.
"""

import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/initial_model.pkl')
    p.add_argument('--data', default='data/train.csv')
    p.add_argument('--method', choices=['isotonic', 'sigmoid'], default='sigmoid')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--output', default='models/calibrated_model.pkl')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print('Model not found at', args.model)
        return
    if not os.path.exists(args.data):
        print('Data not found at', args.data)
        return

    print('Loading trained pipeline...')
    pipeline = joblib.load(args.model)

    # Extract tfidf and clf
    if 'tfidf' not in pipeline.named_steps or 'clf' not in pipeline.named_steps:
        print('Expected pipeline with steps: tfidf and clf')
        return

    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    print('Loading data...')
    df = pd.read_csv(args.data)
    X = df['text'].astype(str)
    y = df['label'].astype(int)

    # Reserve a validation set for calibration
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    print('Transforming validation texts with TF-IDF...')
    X_val_vec = tfidf.transform(X_val)

    print(f'Fitting CalibratedClassifierCV(method={args.method})...')
    # sklearn renamed argument to `estimator` in some versions
    try:
        calibrated = CalibratedClassifierCV(estimator=clf, method=args.method, cv='prefit')
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=clf, method=args.method, cv='prefit')
    calibrated.fit(X_val_vec, y_val)

    print('Building calibrated pipeline and saving to', args.output)
    calibrated_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', calibrated),
    ])

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(calibrated_pipeline, args.output)
    print('Calibrated pipeline saved.')


if __name__ == '__main__':
    main()
