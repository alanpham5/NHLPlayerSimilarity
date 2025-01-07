import re
import numpy as np
from sklearn.impute import KNNImputer
from config import icetime_relative_features
import pandas as pd
from datetime import datetime

storage_options = {'User-Agent': 'Mozilla/5.0'}

# Util Functions
# Only used to clean/display data. Nothing too substantial

def get_data(year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv"
    data = pd.read_csv(url, storage_options=storage_options)
    return data

def convert_height_to_inches(height_str):
    if pd.isna(height_str):
      return np.nan
    match = re.match(r"(\d+)[\'\-]\s*(\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    else:
        return np.nan

def add_bmi(df):
    df['bmi'] = df['weight'] / (df['height'] ** 2) * 703
    return df

def merge_player_data(df_stats, df_info):
    df_info['birthDate'] = pd.to_datetime(df_info['birthDate'])

    merged_df = pd.merge(df_stats, df_info, on='playerId', how='inner')

    def calculate_age(row):
        season_year = int(row['season'])
        # Assume season starts in October
        season_date = datetime(season_year, 10, 1)
        age = season_date.year - row['birthDate'].year - (
            (season_date.month, season_date.day) < (row['birthDate'].month, row['birthDate'].day)
        )
        return age

    merged_df['age'] = merged_df.apply(calculate_age, axis=1)

    columns = ['playerId', 'height', 'weight', 'age'] + [col for col in df_stats.columns if col not in ['playerId']]
    return merged_df[columns]

def scale_stats_per_60_min(df):
    df['icetime_hours'] = df['icetime'] / 3600

    stats_columns = [col for col in df.columns if col not in ["playerId", "name", "season", 'height', 'weight',"age", "bmi", "position", "icetime"]]

    df_scaled = df.copy()
    df_scaled[stats_columns] = df[stats_columns].div(df['icetime_hours'], axis=0)

    df_scaled = df_scaled.drop(columns=['icetime_hours', 'icetime'])

    return df_scaled

def process_data(df, player_bio):
    mergedData = merge_player_data(df, player_bio)

    alldata = mergedData[mergedData['situation'] == 'all']

    alldata = alldata[alldata['icetime'] >= alldata['icetime'].quantile(0.20)]

    alldata = alldata.loc[:, icetime_relative_features]

    alldata["height"] = alldata["height"].apply(convert_height_to_inches)
    return alldata


def impute_data(alldata, include_icetime=True):
    nonnum_columns = ["playerId", "name", "position", "season"]
    numeric_columns  = list(set(icetime_relative_features) - set(nonnum_columns))
    if not include_icetime:
      numeric_columns.remove("icetime")

    alldata_nonnum = alldata[nonnum_columns]
    alldata_numeric = alldata[numeric_columns]

    imputer = KNNImputer(n_neighbors=7)
    alldata_numeric = pd.DataFrame(imputer.fit_transform(alldata_numeric), columns=alldata_numeric.columns)

    alldata = pd.concat([alldata_nonnum.reset_index(drop=True),alldata_numeric.reset_index(drop=True)], axis=1)
    alldata = add_bmi(alldata)
    return alldata


