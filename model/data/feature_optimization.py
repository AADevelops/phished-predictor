import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo

# Fetch dataset
df = pd.read_csv('https://archive.ics.uci.edu/static/public/967/data.csv')

# Remove rows with URLLength > 1200
df = df[df['URLLength'] < 1200] 

# Remove features with low variance
vt = VarianceThreshold(threshold=0.01)
vt.fit(df.select_dtypes(include=[np.number]))
keep_vars = df.select_dtypes(include=[np.number]).columns[vt.get_support()]

# Drop features with correlation > 0.9
corr = df[keep_vars].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
filtered_vars = [v for v in keep_vars if v not in to_drop]

# Saving file after removing features with low variance and hugh correlation
df = df[filtered_vars + ['label']]
df.to_csv('corr&var_filtered_data.csv', index=False)

# Scoring features by importance using randmom forest, selecting top 20

X = df[filtered_vars]
Y = df['label']

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X,Y)

importances = pd.Series(rf.feature_importances_, index=filtered_vars)
top25 = importances.sort_values(ascending = False).head(25)
print(top25)

