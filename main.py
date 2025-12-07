import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from dython.nominal import associations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Wczytywanie danych
df = pd.read_csv("Data/nashville_sample.csv", low_memory=False)
sns.set_theme(style="whitegrid", rc={'figure.figsize':(10, 6)})

print(f"Kolumny danych:\n{df.columns.values}\n")
#
# # Rozkład zatrzymań w ciągu doby
# time_df = pd.DataFrame()
# time_df["time_obj"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
# time_df["hour_of_day"] = time_df["time_obj"].dt.hour
#
# hourly_distribution = time_df["hour_of_day"].value_counts().sort_index()
# hourly_order = sorted(time_df['hour_of_day'].unique())
#
# plt.figure(figsize=(12, 6))
# sns.countplot(
#     x='hour_of_day',
#     data=time_df,
#     order=hourly_order,
# )
# plt.title('Rozkład liczby zatrzymań w ciągu doby', fontsize=16)
# plt.xlabel('Godzina (0-23)')
# plt.ylabel('Liczba zatrzymań')
# plt.show()
#
# bins = [-1, 6, 12, 18, 24]
# labels = ["Noc", "Rano", "Popołudnie", "Wieczór"]
#
# time_df["day_timeframe"] = pd.cut(
#     time_df["hour_of_day"],
#     bins=bins,
#     labels=labels,
#     right=True
# )
# plt.figure(figsize=(8, 5))
# sns.countplot(x='day_timeframe', data=time_df, order=labels)
# plt.title('Rozkład zatrzymań wg pory dnia')
# plt.xlabel('Pora dnia')
# plt.ylabel('Liczba zatrzymań')
# plt.show()
#
# # Rozkład wieku zatrzymanych
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
# plt.ylabel('Liczba zatrzymań')
# plt.legend()
# plt.show()
#
# # Rozkład zatrzymań ze względu na rasę
# # Analiza rasy
# anal_col = "subject_race"
# print("\nZatrzymania według rasy")
# print(f"Udział procentowy: {(df[anal_col].value_counts(normalize=True) * 100).round(2).astype(str) + '%'}")
#
# # Wykres
# plt.figure(figsize=(10, 5)) # Definiujemy rozmiar dla tego konkretnego wykresu
# sns.countplot(x='subject_race', data=df, order=df['subject_race'].value_counts().index)
# plt.title('Rozkład zatrzymań ze względu na rasę (subject_race)')
# plt.xlabel('Rasa')
# plt.ylabel('Liczba zatrzymań')
# plt.show()
#
# # Rozkład zatrzymań ze względu na płeć
# counts = df["subject_sex"].value_counts()
# print(counts)
#
# print("\nProcent")
# percentages = (df["subject_sex"].value_counts(normalize=True) * 100).round(2)
# print(percentages.astype(str) + '%')
#
# counts = df["subject_sex"].value_counts()
# plt.figure(figsize=(8, 8))
# counts.plot(
#         kind='pie',
#         autopct='%1.1f%%',
#         startangle=90,
#         labels=None,
#         pctdistance=0.85,
#         explode=[0.01] * len(counts)
#     )
#
# plt.title(f'Rozkład płci (kolumna: {"subject_sex"})', fontsize=16)
# plt.ylabel('')
# plt.legend(counts.index, title="Płeć", loc="best")
# plt.axis('equal')
#
# plt.show()
#
# # Rozkład zatrzymań ze względu na wykorczenie
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
#
# anal_col = "search_conducted"
#
# # Analiza wieku
# print(f"\nAnaliza rozkładu kolumny: {anal_col}")
# print("Rozkład:")
# print(df[anal_col].describe())
#
# print("\nUdział procentowy:")
# print((df[anal_col].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

# Analiza relacji pomiędzy zmiennymi
columns_to_analyze = [
    'time',
    'subject_age',
    'subject_race',
    'subject_sex',
    'violation',
    'arrest_made',
    'citation_issued',
    'warning_issued',
    'contraband_found',
    'contraband_drugs',
    'contraband_weapons',
    'frisk_performed',
    'search_conducted',
    'search_basis',
]

cols_present = [c for c in columns_to_analyze if c in df.columns]
df_subset = df[cols_present]

assoc_results = associations(
    df_subset,
    nominal_columns='auto',
    numerical_columns='auto',
    mark_columns=False,
    nom_nom_assoc='cramer',
    num_num_assoc='pearson',
    figsize=(12, 12),
    annot=True,
    fmt='.2f',
    cmap=colormaps["coolwarm"],
    title='Macierz asoscjacji'
)
pd_tmp = assoc_results["corr"]
pd_tmp.to_csv("assoc.csv", encoding='utf-8', index=False, header=True)

# # Ekstrakcja wiedzy
# df_prep = df.copy()
# df_prep["time_obj"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
# df_prep["hour_of_day"] = df_prep["time_obj"].dt.hour
# df_prep = df_prep.dropna(subset=["hour_of_day"])
# df_prep["hour_of_day"] = df_prep["hour_of_day"].astype(int)
#
# target_var = "outcome"
#
# t10_violations = df_prep['violation'].value_counts().head(10)
# df_prep["violations_simplified"] = df_prep["violation"].apply(lambda x: x if x in t10_violations.index else "other")
#
# feature_list = ["subject_age", "subject_race", "subject_sex", "violations_simplified", "hour_of_day"]
# df_model = df_prep[feature_list + [target_var]].dropna(subset=[target_var])
#
# imputer = SimpleImputer(strategy='median')
# df_model["subject_age"] = imputer.fit_transform(df_model[["subject_age"]])
#
# X = pd.get_dummies(df_model[feature_list], drop_first=True)
# y = df_model[target_var]
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.3,
#     random_state=42,
#     stratify=y
# )
#
# scaler = StandardScaler()
# X_train[["subject_age", "hour_of_day"]] = scaler.fit_transform(X_train[["subject_age", "hour_of_day"]])
# X_test[["subject_age", "hour_of_day"]] = scaler.transform(X_test[["subject_age", "hour_of_day"]])
#
# model = RandomForestClassifier(
#     random_state=42,
#     class_weight="balanced",
#     n_estimators=100
# )
#
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# cm = confusion_matrix(y_test, y_pred)
# class_labels = model.classes_
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=class_labels,
#             yticklabels=class_labels)
# plt.title('Macierz Pomyłek')
# plt.xlabel('Przewidziane')
# plt.ylabel('Rzeczywiste')
# plt.show()
#
# importances = model.feature_importances_
# feature_names = X_train.columns
#
# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
#
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
# plt.title('Top 15 Zmiennych mających wpływ na przewidywanie aresztowania')
# plt.show()