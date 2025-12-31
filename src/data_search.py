import pandas as pd
import os


def find_nan_rows(file_path, column_name):
    """ Szybka funkcja do filtrowania i wyświetlenia wierszy """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, 'data', 'scrapped', file_path)
    if not os.path.exists(input_path):
        print(f"Nie znaleziono pliku: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    if column_name not in df.columns:
        print(f"Błąd: Kolumna '{column_name}' nie istnieje w pliku.")
        print(f"Dostępne kolumny: {list(df.columns)}")
        return

    nan_rows = df[df[column_name].isna()]

    if not nan_rows.empty:
        print(f"\n--- Znaleziono {len(nan_rows)} wierszy z wartością NaN w kolumnie '{column_name}' ---")
        # Wyświetlamy cały wiersz (to_string() pozwala zobaczyć całość bez kropek '...')
        print(nan_rows.to_string())
    else:
        print(f"\nW kolumnie '{column_name}' nie ma żadnych brakujących wartości.")


if __name__ == "__main__":
    FILE = "deweloperuch_1.csv"
    COLUMN = "Pietro"

    find_nan_rows(FILE, COLUMN)
