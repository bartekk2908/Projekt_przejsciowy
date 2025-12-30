import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'processed', 'dataset2.parquet')

if os.path.exists(data_path):
    df = pd.read_parquet(data_path)

    print("\nTYPY DANYCH:")
    print(df.info())

    print("\nSTATYSTYKI OPISOWE")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df[['Cena', 'Metraz', 'Pokoje', 'Pietro']].describe())

    print("\nBRAKI DANYCH")
    print(df.isnull().sum())

    print("\nLICZBA OFERT WG MIAST:")
    print(df['Miasto'].value_counts())

    print("\nPRZYKŁADOWE WIERSZE")
    print(df.head())

else:
    print(f"Nie znaleziono pliku {data_path}")
