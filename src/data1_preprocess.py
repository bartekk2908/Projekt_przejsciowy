import os
import glob
import pandas as pd

from data1_download import data_download


def load_and_preprocess_data(data_path):
    """ Funkcja wczytuje dane ze ścieżki, scala je i wstępnie obrabia oraz zwraca obiekt data frame. """
    
    # Ignorujemy 'apartments_rent_pl...', interesują nas pliki 'apartments_pl_...'
    search_pattern = os.path.join(data_path, "apartments_pl_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        print(f"Nie znaleziono plików sprzedaży w: {data_path}")
    print(f"Znaleziono {len(files)} plików sprzedaży.")
    
    # Wczytywanie plików csv i zapisywanie rok i miesiąc z nazwy
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        filename = os.path.basename(f)
        parts = filename.split('_')
        try:
            year = int(parts[-2])
            month = int(parts[-1].replace('.csv', ''))
            temp_df['month'] = month
            temp_df['year'] = year
        except (IndexError, ValueError):
            print(f"Nie udało się wyciągnąć daty z pliku: {filename}")
        
        df_list.append(temp_df)

    # Scalanie wczytanych plików csv do jednego df
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Usuwanie duplikatów (ta sama oferta w różnych miesiącach)
    initial_count = len(full_df)
    full_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
    print(f"Usunięto {initial_count - len(full_df)} duplikatów.\nPozostało {len(full_df)} unikalnych ofert.")

    # Obsłużenie kolumn binarnych (zamiana tekstu 'yes'/'no' na liczby 1/0)
    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    for col in binary_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].map({'yes': 1, 'no': 0})
            full_df[col] = full_df[col].astype(float)

    # Wybieramy kluczowe kolumny kategoryczne
    categorical_cols = ['city', 'type', 'ownership', 'buildingMaterial', 'condition']
    
    for col in categorical_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    # Usunięcie ofert bez ceny
    full_df.dropna(subset=['price'], inplace=True)

    return full_df


def save_preprocessed_data(df, output_filename="dataset.parquet"):
    """ Funkcja zapisuje data frame do pliku parquet """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Tworzymy folder data jeśli nie istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_filename)
    df.to_parquet(output_path, index=False)

    print(f"Zapisano przetworzony dataset w: {output_path}")
    print(f"Rozmiar: {df.shape}")


if __name__ == "__main__":

    # https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland/code
    path = data_download("krzysztofjamroz/apartment-prices-in-poland")

    df = load_and_preprocess_data(path)

    save_preprocessed_data(df)
