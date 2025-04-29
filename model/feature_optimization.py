import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

# 1) Load raw
df = pd.read_csv('model/data/csv-versions/1raw/raw_data.csv')
y = df['label']

# 2) Variance threshold
num_df = df.select_dtypes(include=[np.number]).drop(columns=['label'])
vt = VarianceThreshold(threshold=0.01)
vt.fit(num_df)
keep = num_df.columns[vt.get_support()]

# 3) Corr filter
corr = df[keep].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [c for c in upper.columns if any(upper[c] > 0.90)]
filtered = [c for c in keep if c not in drop]

# 4) Save corr&var–filtered with URL
df1 = df[['URL'] + filtered]
df1.to_csv('model/data/csv-versions/2corr&var/corr&var_filtered_data.csv', index=False)

# 5) RF importances
X = df1[filtered]
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
imps = pd.Series(rf.feature_importances_, index=X.columns)
top25 = imps.nlargest(25).index.tolist()

# 6) Save RF–filtered with URL
df2 = df1[['URL'] + top25]
df2.to_csv('model/data/csv-versions/3rf/rf_filtered_data.csv', index=False)

print("Feature optimization finished, top 25 saved.")