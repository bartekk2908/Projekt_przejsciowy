import os
import glob
import pandas as pd

from data_download import data_download



def load_and_preprocess_data(data_path):
    
    # Ignorujemy 'apartments_rent_pl...', interesują nas pliki 'apartments_pl_...'
    search_pattern = os.path.join(data_path, "apartments_pl_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików sprzedaży w: {data_path}")
    print(f"Znaleziono {len(files)} plików sprzedaży.")
    
    # Scalanie plików csv do jednego df
    df_list = [pd.read_csv(f) for f in files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Usuwanie duplikatów (ta sama oferta w różnych miesiącach)
    initial_count = len(full_df)
    full_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
    print(f"Usunięto {initial_count - len(full_df)} duplikatów.\nPozostało {len(full_df)} unikalnych ofert.")

    # Formatowanie typów danych pod XGBoost / LightGBM
    # Wybieramy kluczowe kolumny kategoryczne
    categorical_cols = ['city', 'type', 'ownership', 'condition', 'buildingMaterial']
    
    for col in categorical_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    # Usunięcie ofert bez ceny
    full_df.dropna(subset=['price'], inplace=True)

    return full_df


def save_preprocessed_data(df):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'processed')

    # Tworzymy folder data jeśli nie istnieje
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_file = os.path.join(data_dir, "dataset.parquet")
    df.to_parquet(output_file, index=False)
    
    print(f"Zapisano dataset w: {output_file}")
    print(f"Rozmiar: {df.shape}")


if __name__ == "__main__":

    path = data_download("krzysztofjamroz/apartment-prices-in-poland")

    df = load_and_preprocess_data(path)

    save_preprocessed_data(df)
