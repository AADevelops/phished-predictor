import pandas as pd

in_path  = 'model/data/csv-versions/3rf/rf_filtered_data.csv'
out_path = 'model/data/csv-versions/4feature_generation/generated_features.csv'
df = pd.read_csv(in_path)

# “?” count
df['NumQuestionMarks'] = df['URL'].str.count(r'\?')

# Digit-to-letter ratio
digits = df['URL'].str.count(r'\d')
letters = df['URL'].str.count(r'[A-Za-z]').replace(0, 1)
df['DigitLetterRatio'] = digits / letters

# Suspicious extension flag
exts = ['exe','zip','scr','js','php']
df['HasSuspiciousExt'] = (df['URL'].str.lower()
                            .str.endswith(tuple(f'.{e}' for e in exts))
                            .astype(int))

# Upper-to-lowercase ratio
upper = df['URL'].str.count(r'[A-Z]')
lower = df['URL'].str.count(r'[a-z]').replace(0, 1)
df['UpperLowerRatio'] = upper / lower

# Reorder so new features sit up front
new_cols = [
    'URL',
    'NumQuestionMarks',
    'DigitLetterRatio',
    'HasSuspiciousExt',
    'UpperLowerRatio'
]
others = [c for c in df.columns if c not in new_cols]
df = df[new_cols + others]

df.to_csv(out_path, index=False)
print(f"Generated features saved to {out_path}")