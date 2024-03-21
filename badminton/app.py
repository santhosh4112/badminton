from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
import sklearn

app = Flask(__name__)

# Load CountVectorizer and RandomForestClassifier model
vectorizer = load('count_vectorizer_vocab.pkl')
rf_model = load('best_rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict():
    text = request.form['text']
    text_transformed = vectorizer.transform([text])
    prediction = rf_model.predict(text_transformed)
    # sentiment = 'negative' if prediction < 3 else 'positive'
    return render_template('results.html', text=text, sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')
