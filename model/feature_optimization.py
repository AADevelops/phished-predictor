import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier


# Fetch dataset
df = pd.read_csv('https://archive.ics.uci.edu/static/public/967/data.csv')
df.to_csv('model/data/csv-versions/1raw/raw_data.csv', index=False)

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
df = df[['URL'] + filtered_vars]
df.to_csv('model/data/csv-versions/2corr&var/corr&var_filtered_data.csv', index=False)

# Scoring features by importance using randmom forest, selecting top 25

X = df[filtered_vars]
Y = df['label']

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X,Y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top25 = importances.sort_values(ascending = False).head(25)
print(top25)

# Saving file after using random forest to select top 25 features

features = list(top25.index)
df_selected = df[['URL'] + features]
df_selected.to_csv('model/data/csv-versions/3rf/rf_filtered_data.csv', index=False)
