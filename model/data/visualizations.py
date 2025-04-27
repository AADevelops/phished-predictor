import pandas as pd 
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


# Preview the dataset
df = pd.read_csv('https://archive.ics.uci.edu/static/public/967/data.csv')

print(df.head())