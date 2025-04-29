# ReviewSleuth-Fake-Product-Review-Detector
ReviewSleuth is a machine learning-powered web app designed to detect whether a product review is genuine or fake. Built with a Logistic Regression model and TF-IDF vectorizer, the app provides a simple and clean UI for users to paste a review and instantly check its authenticity.

# Review Sentiment Classifier – Fake vs Real Review Detection

This project leverages **Machine Learning** techniques to build an intelligent review classifier. It uses a combination of **TF-IDF Vectorization** and **Logistic Regression** to distinguish between fake and real reviews. The system has been trained on real-world data to understand product reviews' sentiment and authenticity. The goal is to provide a robust and reliable model to identify fake reviews based on text analysis.

## Features

- **Text Classification**: Detects fake vs real reviews.
- **TF-IDF Vectorization**: Transforms text data into numerical features for model input.
- **Logistic Regression Model**: A reliable machine learning algorithm to classify reviews.
- **Model Serialization**: Uses joblib to save the model for easy reuse.
- **Interactive User Interface**: Simple frontend to input new review text.
- **Prediction Output**: Predicts if a review is fake (0) or real (1).

---

## Tech Stack

| Layer       | Tool / Library                           |
|-------------|------------------------------------------|
| Backend     | Python + Flask                           |
| ML/NLP      | Scikit-learn, TF-IDF, Logistic Regression|
| Serialization | joblib                                 |
| Frontend    | HTML, CSS                                |
| Deployment  | Localhost / Web Hosting Ready            |

---

## Libraries Used

- flask
- scikit-learn
- joblib
- numpy
- pandas
- html
- css

## Folder Structure

Review-Classifier/

│
├── static/             

│   └── style.css         

│

├── templates/         

│   └── index.html     

│

├── model/          

│   └── logistic_model.pkl

│   └── tfidf_vectorizer.pkl

│

├── app.py               

├── preprocess.py       

├── model.py             

├── prediction.py          

└── README.md    


---

## Why This Stack?

- **TF-IDF**: Efficient for transforming text data into numerical format, accounting for word frequency and importance.
- **Logistic Regression**: A straightforward and effective classification model for binary outcomes (fake vs real).
- **Flask**: Lightweight web framework to expose the model as a REST API.
- **joblib**: Saves and loads the model efficiently to avoid retraining every time.

---

## Project Walkthrough

1. **Preprocessing**:
    - The text data is first cleaned and transformed using **TF-IDF**.
    - **TF-IDF** accounts for the importance of each word across the entire dataset, converting the reviews into vectors.
  
2. **Model**:
    - The model used is **Logistic Regression**, which is trained to differentiate between fake and real reviews based on the features extracted from the TF-IDF vectorizer.
  
3. **Prediction**:
    - Once the model is trained and saved using **joblib**, it can be used for real-time prediction. The app receives a review as input, transforms it with the vectorizer, and passes it to the model to predict whether the review is fake or real.
  
4. **UI**:
    - A simple HTML/CSS interface is provided for the user to input review text and get the prediction.

---

## Future Plans

- Integrate additional models for comparison (e.g., **SVM**, **Naive Bayes**).
- Add **Web Scraping** capabilities to collect real-time reviews for predictions.
- **Deploy the model** to cloud platforms like **Heroku** or **AWS** for production.
- Enhance the UI with better styling and features.
- Experiment with **Deep Learning** models for higher accuracy.

---
