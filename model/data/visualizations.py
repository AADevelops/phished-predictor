import pandas as pd 
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
from itertools import combinations


# Preview the dataset
df = pd.read_csv('https://archive.ics.uci.edu/static/public/967/data.csv')
df = df[df['URLLength'] < 1200]  # Filter out URLs longer than 21200 characters
print(df.head())

# Label distrubution of phishing or not phishing
# 1 = not phishing, 0 = phishing
plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind='bar')
#plt.show()

#Correlation matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 12))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar() 
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

#Scatter plot of URLLength vs NoOfLettersInURL
x = df['URLLength']
y = df['NoOfLettersInURL']

df['URLBin'] = (df['URLLength'] // 250) * 250
grouped = df.groupby('URLBin')['NoOfLettersInURL'].agg(['mean', 'std']).sort_index()
bin_centers = grouped.index + 125

coef = np.polyfit(x, y, 1)
poly_fn = np.poly1d(coef)
x_sorted = np.sort(x)
y_pred = poly_fn(x_sorted)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, label='Data points')
plt.plot(x_sorted, y_pred, '--', linewidth=2, label='Best-fit line')
plt.errorbar(bin_centers, grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, label='Binned mean ± 1 std')
plt.xlabel('URL Length')
plt.ylabel('Number of Letters in URL')
plt.title('URL Length vs. #Letters with Binned Std Dev')
plt.legend()
plt.grid(True)
plt.show()

#Scatter plot of URLLength vs NoOfDegitsInURL
x = df['URLLength']
y = df['NoOfDegitsInURL']

df['URLBin'] = (df['URLLength'] // 250) * 250
grouped = df.groupby('URLBin')['NoOfDegitsInURL'].agg(['mean', 'std']).sort_index()
bin_centers = grouped.index + 125

coef = np.polyfit(x, y, 1)
poly_fn = np.poly1d(coef)
x_sorted = np.sort(x)
y_pred = poly_fn(x_sorted)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, label='Data points')
plt.plot(x_sorted, y_pred, '--', linewidth=2, label='Best-fit line')
plt.errorbar(bin_centers, grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, label='Binned mean ± 1 std')
plt.xlabel('URL Length')
plt.ylabel('Number of Digits in URL')
plt.title('URL Length vs. #Digits with Binned Std Dev')
plt.legend()
plt.grid(True)
plt.show()

# Scatterplots for every pair of numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col_x, col_y in combinations(numeric_cols, 2):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col_x], df[col_y], alpha=0.5)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'{col_x} vs {col_y}')
    plt.grid(True)
    plt.show()
