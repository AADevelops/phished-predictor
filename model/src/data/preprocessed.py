import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

df = pd.read_csv("model/data/4feature_generation/generated_features.csv")

url   = df.pop("URL")
label = df.pop("label")
features = df  # now just your numeric columns

X = StandardScaler().fit_transform(features)
X = Normalizer().fit_transform(X)
X = MinMaxScaler().fit_transform(X)

df_out = pd.DataFrame(X, columns=features.columns)
df_out["label"] = label.values
df_out["URL"]   = url.values

df_out.to_csv("model/data/5preprocessed/preprocessed.csv", index=False)
