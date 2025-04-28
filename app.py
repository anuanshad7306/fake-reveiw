from flask import Flask, request, render_template
import joblib
from utils.text_cleaning import preprocess_text

app = Flask(__name__)

model = joblib.load(rb'C:\\Users\\Pc\\Downloads\\ReviewSleuth-Fake-Product-Review-Detector-main\\nextspet_model.pkl')
vectorizer = joblib.load(rb'C:\\Users\\Pc\\Downloads\\ReviewSleuth-Fake-Product-Review-Detector-main\\model\\tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = preprocess_text(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    result = 'Fake Review' if prediction == 1 else 'Original Review'
    return render_template('index.html', review=review, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)