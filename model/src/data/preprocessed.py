import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

in_path  = 'model/data/4feature_generation/generated_features.csv'
out_path = 'model/data/5preprocessed/preprocessed.csv'
df = pd.read_csv(in_path)

url = df[['URL']]
X   = df.drop(columns=['URL'])

# Standardize → Scale → Normalize
X_std    = StandardScaler().fit_transform(X)
X_scaled = MinMaxScaler().fit_transform(X_std)
X_norm   = Normalizer().fit_transform(X_scaled)

df_out = pd.DataFrame(X_norm, columns=X.columns)
df_out = pd.concat([url.reset_index(drop=True), df_out], axis=1)
df_out.to_csv(out_path, index=False)

print(f"Preprocessed data saved to {out_path}")