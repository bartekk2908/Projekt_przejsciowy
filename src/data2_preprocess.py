import os
import pandas as pd
import numpy as np

from data1_preprocess import save_preprocessed_data


def _clean_currency(val):
    """ Zamienia np '1 800 000 zł' (z twardymi spacjami \xa0) na float 1800000.0 """
    if pd.isna(val): return np.nan
    val = str(val).replace('zł', '').replace('\xa0', '').replace(' ', '')
    try:
        return float(val)
    except ValueError:
        return np.nan
    

def _clean_area(val):
    """ Zamienia np '97.0 m²' na float 97.0. """
    if pd.isna(val): return np.nan
    val = str(val).replace('m²', '').strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def _parse_fraction(val):
    """ Zamienia ułamek np '1/2' na float 0.5 """
    if pd.isna(val): return np.nan
    val = str(val).strip()
    try:
        if '/' in val:
            parts = val.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(val)
    except (ValueError, ZeroDivisionError):
        return np.nan


def load_and_preprocess_scrapped_data(input_filename):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, 'data', 'scrapped', input_filename)
    if not os.path.exists(input_path):
        print(f"Nie znaleziono pliku: {input_path}")

    print(f"Wczytywanie danych z: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)

    # Czyszczenie i konwersja danych
    df['Cena'] = df['Cena'].apply(_clean_currency)
    df['Metraz'] = df['Metraz'].apply(_clean_area)
    df['Udzial'] = df['Udzial'].apply(_parse_fraction)
    df['Pietro'] = pd.to_numeric(df['Pietro'], errors='coerce')
    df['Pokoje'] = pd.to_numeric(df['Pokoje'], errors='coerce')
    # Zamiana 0 pokoi na 1 (Zakładamy, że 0 to kawalerka)
    df['Pokoje'] = df['Pokoje'].replace(0, 1)
    # Format daty w pliku to D.MM.YYYY (np. 1.09.2025)
    df['Data_transakcji'] = pd.to_datetime(df['Data_transakcji'], format='%d.%m.%Y', errors='coerce')
    
    # Wyciągamy rok i miesiąc jako cechy numeryczne dla modelu
    df['Rok'] = df['Data_transakcji'].dt.year
    df['Miesiac'] = df['Data_transakcji'].dt.month

    # Usuwamy wiersze, gdzie cena lub metraż są puste
    df = df.dropna(subset=['Cena', 'Metraz']).copy()
    
    # Filtrowanie tylko pełnych własności (udział 1/1)
    df = df[df['Udzial'] == 1.0]

    # Konwersja Miasta na typ kategoryczny
    df['Miasto'] = df['Miasto'].astype('category')

    print(f"Wczytano: {initial_count} wierszy.")
    print(f"Po przetworzeniu: {len(df)} wierszy.")
    
    return df


if __name__ == "__main__":

    input_file = "deweloperuch_1.csv" 
    
    df = load_and_preprocess_scrapped_data(input_file)

    save_preprocessed_data(df, output_filename="dataset2.parquet")
