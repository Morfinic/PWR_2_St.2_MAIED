import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

df = None

try:
    df = pd.read_csv("./Data/nashssville.csv", low_memory=False)
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
    stratify_columns_list = ["subject_race"]

    # age_bins = [0, 17, 25, 35, 45, 55, 65, np.inf]
    # age_labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']

    # df["age_bin"] = pd.cut(
    #     df["subject_age"],
    #     bins=age_bins,
    #     labels=age_labels,
    #     right=True
    # )

    # if df['age_bin'].isnull().any():
    #     df['age_bin'] = df['age_bin'].cat.add_categories('Unknown_Age').fillna('Unknown_Age')

    helper_column = "combined_stratify_key"
    final_stratify_list = stratify_columns_list
    # + ["age_bin"]

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
        stratify=df[helper_column]
    )

    df_sample = df_sample.drop(columns=[helper_column])

    print(len(df_sample))

    df_sample.to_csv("./Data/nashville_sample.csv", index=False)

    # df.to_csv("./Data/nashville_drop.csv")
except FileNotFoundError:
    df = pd.read_csv("./Data/nashville.csv", low_memory=False)
