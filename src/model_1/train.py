from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, y_train, model_dir="models/model_1", model_name="model1.pkl", encode_labels=True):
    
    label_encoder = None
    if encode_labels:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)

    # Define pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

   
    model.fit(X_train, y_train)
    print("Model training completed.")


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    return model, label_encoder


def predict(model, texts, label_encoder=None):

   
    if isinstance(model, tuple):
        model = model[0]

    preds = model.predict(texts)

    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)

    return preds