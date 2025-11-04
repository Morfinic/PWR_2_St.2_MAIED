import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = None

try:
    df = pd.read_csv("./Data/nashville.csv", low_memory=False)
    # Usun kolumny ktore mozna zignorowac do analizy
    df.drop(inplace=True, columns=[
        "raw_row_number", "location", "lat", "lng", "reporting_area",
        "zone", "officer_id_hash", "type", "notes"
    ])
    # Usun wiersze, które posiadaja NaN w kolumnie
    print(len(df))
    df.dropna(axis=0, how='any', inplace=True)
    print(len(df))

    df.to_csv("./Data/nashville_drop.csv")
except FileNotFoundError:
    df = pd.read_csv("./Data/nashville_drop.csv", low_memory=False)

# print(df.columns.values)

sns.set(style="whitegrid", rc={'figure.figsize':(10, 6)})
anal_col = "subject_age"

# Analiza wieku
print(f"\nAnaliza rozkładu kolumny: {anal_col}")
print("Rozkład:")
print(df[anal_col].describe())

print("\nSkośność i kurtoza:")
skosnosc = df[anal_col].skew()
kurtoza = df[anal_col].kurtosis()
print(f"Skośność: {skosnosc}")
print(f"Kurtoza: {kurtoza}")

if skosnosc > 0.5:
        print("Rozkład jest prawostronnie skośny (więcej młodych osób).")
elif skosnosc < -0.5:
    print("Rozkład jest lewostronnie skośny (więcej starszych osób).")
else:
    print("Rozkład jest w przybliżeniu symetryczny.")

# Wykres rozkładu
plt.figure(figsize=(10, 6))
sns.histplot(df[anal_col], kde=True, bins=40)

# Średnia
plt.axvline(
    df[anal_col].mean(), color='red', linestyle='--', linewidth=2,
    label=f'Średnia ({df[anal_col].mean():.1f})'
)
# Mediana
plt.axvline(
    df[anal_col].median(), color='green', linestyle='-', linewidth=2,
    label=f'Mediana ({df[anal_col].median():.1f})'
)

plt.title(f'Rozkład wieku zatrzymanych (kolumna: {anal_col})')
plt.xlabel('Wiek')
plt.ylabel('Liczba obserwacji (częstość)')
plt.legend()
plt.show()

# Analiza rasy
anal_col = "subject_race"
print("\nZatrzymania według rasy")
print(f"Udział procentowy: {(df[anal_col].value_counts(normalize=True) * 100).round(2)}%")

# Wykres
plt.figure(figsize=(10, 5)) # Definiujemy rozmiar dla tego konkretnego wykresu
sns.countplot(x='subject_race', data=df, order=df['subject_race'].value_counts().index)
plt.title('Rozkład zatrzymań ze względu na rasę (subject_race)')
plt.xlabel('Rasa')
plt.ylabel('Liczba zatrzymań')
plt.show()