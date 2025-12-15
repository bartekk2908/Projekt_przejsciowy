import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np


def load_data():
    """ Funkcja wczytuje dane z 'data/processed/dataset.parquet' i zwraca data frame. """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'processed', 'dataset.parquet')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nie znaleziono pliku {data_path}")
        
    return pd.read_parquet(data_path)


def prepare_features(df):
    """ Funkcja rozdziela dane w obiekcie data frame na cechy (X) i etykiety (y) oraz usuwa zbędne kolumny. """
    
    data = df.copy()
    
    y = data['price']
    
    drop_cols = ['price', 'id']
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    
    return X, y


def train_xgboost(X_train, y_train, X_test, y_test):
    """ Funckja definiuje instancję modelu XGBoost, przeprowadza uczenie danych, zwraca wyuczony model. """

    print("\nTrenowanie XGBoost")
    start = time.time()
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True, # trzeba włączyć obsługę kategorii
        tree_method='hist', # szybsza dla dużych zbiorów danych
        # device='cuda' if xgb.is_cuda_available() else 'cpu', # jeżeli jest GPU to oblicza na GPU
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    print(f"Czas treningu XGBoost: {time.time() - start:.2f} s")
    return model


def train_lightgbm(X_train, y_train, X_test, y_test):
    """ Funckja definiuje instancję modelu XGBoost, przeprowadza uczenie danych, zwraca wyuczony model. """
    
    print("\nTrenowanie LightGBM")
    start = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1, # liczba rdzeni procesora (wszystkie dostępne)
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse'
    )
    
    print(f"Czas treningu LightGBM: {time.time() - start:.2f} s")
    return model


def evaluate_model(model, X_test, y_test, name="Model"):
    """ Funckja wyświetla metryki danego modelu dla danych testowych i zwraca predykowane wartości. """

    preds = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    within_10 = np.mean(np.abs(preds - y_test) / y_test <= 0.10) * 100
    
    print(f"\nWyniki dla {name}:")
    print(f"RMSE (błąd średniokwadratowy): {rmse:.2f} [PLN]")
    print(f"MAE (średni błąd bezwzględny): {mae:.2f} [PLN]")
    print(f"MAPE (średni błąd względny): {mape:.2f} [%]")
    print(f"R2 Score (dopasowanie): {r2:.4f}")
    print(f"{within_10:.1f}% ofert wyceniono z dokładnością ±10%")

    return preds


if __name__ == "__main__":

    df = load_data()
    
    X, y = prepare_features(df)
    
    # Podział na zbiór treningowy, walidacyjny i testowy (70% / 15% / 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"\nZbiór treningowy: {X_train.shape}\nZbiór walidacyjny: {X_val.shape}\nZbiór testowy: {X_test.shape}")
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # LightGBM
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    evaluate_model(lgb_model, X_test, y_test, "LightGBM")
