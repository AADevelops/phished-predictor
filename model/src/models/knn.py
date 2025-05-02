

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            neighbors = np.argsort(distances)[:self.k]
            values, counts = np.unique(self.y_train[neighbors], return_counts=True)
            predicted = values[np.argmax(counts)]
            predictions.append(predicted)
        return np.array(predictions)

if __name__ == "__main__":
    df = pd.read_csv("model/data/5preprocessed/preprocessed.csv")
    X = df.drop(columns=["URL", "label"]).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = KNNClassifier(k=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))