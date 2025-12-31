import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'dataset.parquet') 

if os.path.exists(data_path):
    df = pd.read_parquet(data_path)

    print(df.shape)

    print("\nTYPY DANYCH:")
    print(df.info())
    
    print("\nSTATYSTYKI OPISOWE")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df[['price', 'squareMeters', 'rooms', 'buildYear']].describe())

    print("\nBRAKI DANYCH")
    print(df.isnull().sum())

    print("\nLICZBA OFERT WG MIAST:")
    print(df['city'].value_counts())

    print("\nPRZYKŁADOWE WIERSZE:")
    print(df.head())

else:
    print(f"Nie znaleziono pliku {data_path}")
