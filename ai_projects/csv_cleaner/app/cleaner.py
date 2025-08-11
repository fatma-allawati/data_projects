#Core data cleaning logic
import pandas as pd
import os 

"""Cleans a CSV DataFrame by:
    1. Removing empty rows and columns
    2. Normalizing column names (lowercase, underscores, no spaces)
    3. Stripping extra spaces from string values
    4. Removing duplicate rows
    5. Attempting to convert string columns to datetime (if applicable) """

def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    #Drop empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    #Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    #Strip strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    #Remove duplicates
    df.drop_duplicates(inplace=True)

    #Convert data to columns
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass
    return df