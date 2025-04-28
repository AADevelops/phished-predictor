import pandas as pd

# Load the most updated CSV & create new features
in_path  = 'model/data/csv-versions/3rf/rf_filtered_data.csv'
out_path = 'model/data/csv-versions/FINAL_DATA/final.csv'
df = pd.read_csv(in_path)

# Count “?” in each URL
df['NumQuestionMarks'] = df['URL'].str.count(r'\?')

# Digit-to-letter ratio
num_digits = df['URL'].str.count(r'\d')
num_letters = df['URL'].str.count(r'[A-Za-z]').replace(0, 1)
df['DigitLetterRatio'] = num_digits / num_letters

# File-extension flag
exts = ['exe', 'zip', 'scr', 'js', 'php']
df['HasSuspiciousExt'] = (
    df['URL']
      .str.lower()
      .str.endswith(tuple(f'.{e}' for e in exts))
      .astype(int)
)

# Reorder so URL & new features come first
cols = [
    'URL',
    'NumQuestionMarks',
    'DigitLetterRatio',
    'HasSuspiciousExt'
] + [c for c in df.columns if c not in (
    'URL',
    'NumQuestionMarks',
    'DigitLetterRatio',
    'HasSuspiciousExt'
)]
df = df[cols]

# Save out
df.to_csv(out_path, index=False)
print(f"Saved with new features to {out_path}")