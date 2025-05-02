import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import sys

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        chunk_size = 1000
        preds = []
        for start in tqdm(range(0, n_samples, chunk_size)):
            end = min(start + chunk_size, n_samples)
            X_chunk = X[start:end]
            xt2 = np.sum(X_chunk**2, axis=1)
            tr2 = np.sum(self.X_train**2, axis=1)
            cross = X_chunk.dot(self.X_train.T)
            distances_sq = xt2[:, None] + tr2[None, :] - 2 * cross
            distances_sq = np.maximum(distances_sq, 0)
            dists = np.sqrt(distances_sq)
            nbrs = np.argsort(dists, axis=1)[:, :self.k]
            for inds in nbrs:
                values, counts = np.unique(self.y_train[inds], return_counts=True)
                preds.append(values[np.argmax(counts)])
        return np.array(preds, dtype=int)

if __name__ == "__main__":
    df = pd.read_csv("model/data/5preprocessed/preprocessed.csv")
    labels = df["label"]
    labels = labels.astype(int)
    X = df.drop(columns=["URL", "label"]).values
    y = labels.values
    print("Overall label distribution:", np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train label distribution:", np.unique(y_train, return_counts=True))
    print("Test label distribution:", np.unique(y_test, return_counts=True))
    model = KNNClassifier(k=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
