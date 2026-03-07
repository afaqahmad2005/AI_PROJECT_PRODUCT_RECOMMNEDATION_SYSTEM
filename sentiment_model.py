import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_sentiment_model(df_sentiments):
    """
    Train sentiment model using Logistic Regression and TF-IDF
    """
    # Assume df_sentiments has 'review_text' and 'sentiment' columns
    X = df_sentiments['review_text']
    y = df_sentiments['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, y_pred)
    
    return vectorizer, model, test_acc

def predict_sentiment(vectorizer, model, text):
    """
    Predict sentiment for a given text
    """
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction