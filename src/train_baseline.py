import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocess import load_and_preprocess

def main():
    print("🚀 Starting training...")
    os.makedirs('models', exist_ok=True)

    # Load and preprocess data
    df = load_and_preprocess('data/train.csv')
    X = df['clean'].values
    y = df['label'].values.astype(int)

    # Split data (50% test size to avoid stratify error on small dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y if len(set(y)) > 1 else None, random_state=42
    )

    # TF-IDF vectorization
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test_tfidf)
    print('📊 Classification report:')
    print(classification_report(y_test, y_pred, zero_division=0))
    print('✅ Accuracy:', accuracy_score(y_test, y_pred))
    print('🧮 Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(vect, 'models/tfidf_vectorizer.pkl')
    joblib.dump(clf, 'models/logreg_model.pkl')
    print('💾 Saved models to models/')

if __name__ == '__main__':
    main()
