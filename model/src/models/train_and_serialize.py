import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from knn import KNNClassifier   # make sure your PYTHONPATH lets you import this

# — where to read data from & write artifacts to
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "5preprocessed", "preprocessed.csv"))
ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# — load & split
df = pd.read_csv(CSV_PATH)
feature_names = [c for c in df.columns if c not in ("URL","label")]
X = df[feature_names].values
y = df["label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# — fit all of our preprocessing steps
scaler     = StandardScaler().fit(X_train)
X_scaled   = scaler.transform(X_train)
normalizer = Normalizer().fit(X_scaled)
X_norm     = normalizer.transform(X_scaled)
minmax     = MinMaxScaler().fit(X_norm)

# — train the KNN (on the fully‐transformed train set)
X_prepped  = minmax.transform(X_norm)
knn        = KNNClassifier(k=5)
knn.fit(X_prepped, y_train)

# — compute per–feature train means & stds for z–scores
means = dict(zip(feature_names, X_train.mean(axis=0)))
stds  = dict(zip(feature_names, X_train.std(axis=0)))

# — dump everything
joblib.dump(scaler,       os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
joblib.dump(normalizer,   os.path.join(ARTIFACTS_DIR, "normalizer.pkl"))
joblib.dump(minmax,       os.path.join(ARTIFACTS_DIR, "minmax.pkl"))
joblib.dump(knn,          os.path.join(ARTIFACTS_DIR, "knn_model.pkl"))
joblib.dump(feature_names,os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
joblib.dump(means,        os.path.join(ARTIFACTS_DIR, "train_means.pkl"))
joblib.dump(stds,         os.path.join(ARTIFACTS_DIR, "train_stds.pkl"))

print("Artifacts written to", ARTIFACTS_DIR)