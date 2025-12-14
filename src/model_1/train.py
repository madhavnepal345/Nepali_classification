from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import joblib
import os



def train_model(X_train,y_train):
    model=Pipeline([
        ('tfidf',TfidfVectorizer(
            ngram_range=(1,2),
            max_features=100000,
            min_df=5,
            sublinear_tf=True
        )),
        ('clf',LogisticRegression(
            
            max_iter=1000,
            n_jobs=-1
           
        ))
    
    ])
    model.fit(X_train,y_train)
    os.makedirs("models/model_1",exist_ok=True)
    joblib.dump(model,"models/model_1/model1.pkl")

    return model

def predict(model,texts):
    return model.predict(texts)