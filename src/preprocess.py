import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

def load_and_preprocess(path='data/train.csv'):
    df = pd.read_csv(path)
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].fillna('') + ' . ' + df['text'].fillna('')
    elif 'content' in df.columns:
        df['content'] = df['content'].fillna('')
    else:
        raise ValueError("CSV must have 'title' and 'text' or 'content' column.")
    df['clean'] = df['content'].apply(clean_text)
    df = df[['clean', 'label']].dropna().reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = load_and_preprocess()
    print(df.head())
