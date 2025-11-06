import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

df = None

try:
    df = pd.read_csv("./Data/nashville.csv", low_memory=False)
    # Usun kolumny ktore mozna zignorowac do analizy
    df.drop(inplace=True, columns=[
        "raw_row_number", "location", "lat", "lng", "reporting_area",
        "zone", "officer_id_hash", "type", "notes", "search_person", "search_vehicle",
        "date", "vehicle_registration_state", "raw_suspect_ethnicity", "raw_driver_searched",
        "raw_passenger_searched", "raw_search_consent", "raw_search_inventory", "raw_search_plain_view",
        'raw_verbal_warning_issued', 'raw_written_warning_issued',
        'raw_traffic_citation_issued', 'raw_misd_state_citation_issued',
        'raw_search_arrest', 'raw_search_warrant'
    ])
    # Usun wiersze, które posiadaja NaN w kolumnie
    print(len(df))
    # df.dropna(axis=0, how='any', inplace=True)
    # print(len(df))
    n_samples = 50_000
    stratify_columns_list = ["subject_race", "subject_sex"]

    age_bins = [0, 17, 25, 35, 45, 55, 65, np.inf]
    age_labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']

    df["age_bin"] = pd.cut(
        df["subject_age"],
        bins=age_bins,
        labels=age_labels,
        right=True
    )

    if df['age_bin'].isnull().any():
        df['age_bin'] = df['age_bin'].cat.add_categories('Unknown_Age').fillna('Unknown_Age')

    helper_column = "combined_stratify_key"
    final_stratify_list = stratify_columns_list + ["age_bin"]

    df[helper_column] = (
        df[final_stratify_list]
        .astype(str)
        .agg('_'.join, axis=1)
    )

    counts = df[helper_column].value_counts()
    min_count_threshold = 10

    rare_combinations = counts[counts < min_count_threshold].index

    if len(rare_combinations) > 0:
        df[helper_column] = df[helper_column].replace(rare_combinations, "other_combinations")

    sample_frac = n_samples / len(df)

    _, df_sample = train_test_split(
        df,
        test_size=sample_frac,
        stratify=df[helper_column],
        random_state=42
    )

    df_sample = df_sample.drop(columns=[helper_column, "age_bin"])

    print(len(df_sample))

    df_sample.to_csv("./Data/nashville_sample.csv", index=False)

    # df.to_csv("./Data/nashville_drop.csv")
except FileNotFoundError:
    df = pd.read_csv("./Data/nashville_sample.csv", low_memory=False)

# print(df.columns.values)

# sns.set(style="whitegrid", rc={'figure.figsize':(10, 6)})
# anal_col = "subject_age"
#
# # Analiza wieku
# print(f"\nAnaliza rozkładu kolumny: {anal_col}")
# print("Rozkład:")
# print(df[anal_col].describe())
#
# print("\nSkośność i kurtoza:")
# skosnosc = df[anal_col].skew()
# kurtoza = df[anal_col].kurtosis()
# print(f"Skośność: {skosnosc}")
# print(f"Kurtoza: {kurtoza}")
#
# if skosnosc > 0.5:
#         print("Rozkład jest prawostronnie skośny (więcej młodych osób).")
# elif skosnosc < -0.5:
#     print("Rozkład jest lewostronnie skośny (więcej starszych osób).")
# else:
#     print("Rozkład jest w przybliżeniu symetryczny.")
#
# # Wykres rozkładu
# plt.figure(figsize=(10, 6))
# sns.histplot(df[anal_col], kde=True, bins=40)
#
# # Średnia
# plt.axvline(
#     df[anal_col].mean(), color='red', linestyle='--', linewidth=2,
#     label=f'Średnia ({df[anal_col].mean():.1f})'
# )
# # Mediana
# plt.axvline(
#     df[anal_col].median(), color='green', linestyle='-', linewidth=2,
#     label=f'Mediana ({df[anal_col].median():.1f})'
# )
#
# plt.title(f'Rozkład wieku zatrzymanych (kolumna: {anal_col})')
# plt.xlabel('Wiek')
# plt.ylabel('Liczba obserwacji (częstość)')
# plt.legend()
# plt.show()
#
# # Analiza rasy
# anal_col = "subject_race"
# print("\nZatrzymania według rasy")
# print(f"Udział procentowy: {(df[anal_col].value_counts(normalize=True) * 100).round(2)}%")
#
# # Wykres
# plt.figure(figsize=(10, 5)) # Definiujemy rozmiar dla tego konkretnego wykresu
# sns.countplot(x='subject_race', data=df, order=df['subject_race'].value_counts().index)
# plt.title('Rozkład zatrzymań ze względu na rasę (subject_race)')
# plt.xlabel('Rasa')
# plt.ylabel('Liczba zatrzymań')
# plt.show()
#
# print("Zatrzymania według wykroczenia")
# print(f"Liczba unikalnych typów wykroczeń: {df['violation'].nunique()}")
#
# # Ilość wystąpień
# print("\nTop 10 najczęstszych wykroczeń:")
# top_10_violations = df['violation'].value_counts()
# print(top_10_violations)
#
# # Procenty wystąpień
# print("\nUdział procentowy:")
# print((df['violation'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
#
# plt.figure(figsize=(10, 7))
# sns.countplot(
#     y='violation',
#     data=df,
#     order=top_10_violations.index
# )
# plt.title('Rozkład zatrzymań ze względu na wykroczenie')
# plt.xlabel('Liczba zatrzymań')
# plt.ylabel('Typ wykroczenia')
# plt.show()