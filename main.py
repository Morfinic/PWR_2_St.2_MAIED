import pandas as pd

df = pd.read_csv("./Data/nashville.csv", low_memory=False)
# Usun kolumny ktore mozna zignorowac do analizy
df.drop(inplace=True, columns=[

])
# Drop rows with wmpty values
df.dropna(axis=0, how='any', inplace=True)
