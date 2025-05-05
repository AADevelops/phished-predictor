import os
import joblib
import pandas as pd
import numpy as np

# base directory for loading artifacts
ARTIFACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
)

_artifacts = None

def _load_artifacts():
    global _artifacts
    if _artifacts is None:
        try:
            scaler       = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
            normalizer   = joblib.load(os.path.join(ARTIFACTS_DIR, "normalizer.pkl"))
            minmax       = joblib.load(os.path.join(ARTIFACTS_DIR, "minmax.pkl"))
            model        = joblib.load(os.path.join(ARTIFACTS_DIR, "knn_model.pkl"))
            feature_names= joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
            means        = joblib.load(os.path.join(ARTIFACTS_DIR, "train_means.pkl"))
            stds         = joblib.load(os.path.join(ARTIFACTS_DIR, "train_stds.pkl"))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Missing artifact file: {e.filename}. Please run train_and_serialize.py first."
            ) from e
        _artifacts = (
            scaler, normalizer, minmax,
            model, feature_names, means, stds
        )
    return _artifacts

def evaluate(input_dict):
    scaler, normalizer, minmax, model, feature_names, means, stds = _load_artifacts()

    # validate keys
    missing = set(feature_names) - set(input_dict)
    extra   = set(input_dict) - set(feature_names) - {"URL"}
    if missing or extra:
        raise ValueError(
            f"Expected features {feature_names}, got extras {extra}, missing {missing}"
        )

    # build DataFrame and extract values
    df = pd.DataFrame([input_dict])
    X = df[feature_names].values

    # apply preprocessing steps
    X = scaler.transform(X)
    X = normalizer.transform(X)
    X = minmax.transform(X)

    # predict probabilities (or fallback to predict)
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(X)
        arr = np.asarray(raw_probs)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            phish_prob = float(arr[0, 1])
        elif arr.ndim == 1:
            phish_prob = float(arr[0])
        else:
            raise RuntimeError(f"Unexpected output shape from predict_proba: {arr.shape}")
        pred = int(phish_prob > 0.5)
    else:
        preds = model.predict(X)
        phish_prob = float(preds[0])
        pred = int(preds[0])

    # compute top reasons via z-scores
    z_scores = {
        f: abs((input_dict[f] - means[f]) / stds[f])
        for f in feature_names
    }
    reasons = sorted(z_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_reasons = [{"feature": f, "z": round(z, 3)} for f, z in reasons]

    return {
        "prediction": pred,
        "phishingProb": phish_prob,
        "topReasons": top_reasons,
        "featureValues": {f: input_dict[f] for f in feature_names}
    }

if __name__ == "__main__":
    # load artifacts to retrieve feature names
    _, _, _, _, feature_names, _, _ = _load_artifacts()
    example = {f: 0.0 for f in feature_names}
    print(evaluate(example))
