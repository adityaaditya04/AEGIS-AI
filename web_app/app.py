import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request
import joblib
from src.preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer from the models folder
vect = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl'))
clf = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'logreg_model.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['news']
        cleaned = clean_text(input_text)
        x = vect.transform([cleaned])
        prediction = clf.predict(x)[0]
        try:
            probability = clf.predict_proba(x)[0][1]
        except:
            probability = None

    return render_template('index.html', prediction=prediction, probability=probability, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
