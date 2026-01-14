import os
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb


# ============================================================
# 1) Wspólny schemat cech (mapping)
# ============================================================
COMMON_MAP_1 = {  # Kaggle -> wspólne nazwy
    "city": "city",
    "squareMeters": "area",
    "rooms": "rooms",
    "floor": "floor",
    "year": "year",
    "month": "month",
}

COMMON_MAP_2 = {  # Deweloperuch -> wspólne nazwy
    "Miasto": "city",
    "Metraz": "area",
    "Pokoje": "rooms",
    "Pietro": "floor",
    "Rok": "year",
    "Miesiac": "month",
}


def to_common_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    source: "kaggle" albo "deweloperuch"
    Zwraca DF z kolumnami: [city, area, rooms, floor, year, month]
    """
    if source == "kaggle":
        mapping = COMMON_MAP_1
    elif source == "deweloperuch":
        mapping = COMMON_MAP_2
    else:
        raise ValueError("source must be 'kaggle' or 'deweloperuch'")

    # wybierz i przemianuj
    X = df[list(mapping.keys())].rename(columns=mapping).copy()

    # typy
    X["city"] = X["city"].astype("category")

    for c in ["area", "rooms", "floor"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X["year"] = pd.to_numeric(X["year"], errors="coerce").astype("Int64")
    X["month"] = pd.to_numeric(X["month"], errors="coerce").astype("Int64")

    return X


# ============================================================
# 2) Wczytanie danych
# ============================================================
def project_parquet_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "data", "processed", filename)


# ============================================================
# 3) Ewaluacja
# ============================================================
def evaluate(model, X, y, name: str):
    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    mask = y_true != 0
    within_10 = (np.abs(y_pred[mask] - y_true[mask]) / y_true[mask] <= 0.10).mean() * 100

    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")
    print(f"±10%: {within_10:.1f}%")


# ============================================================
# 4) Main
# ============================================================
def main():
    df1 = pd.read_parquet(project_parquet_path("dataset.parquet"))
    df2 = pd.read_parquet(project_parquet_path("dataset2.parquet"))

    # wspólne cechy
    X1 = to_common_schema(df1, "kaggle")
    y1 = df1["price"]

    X2 = to_common_schema(df2, "deweloperuch")
    y2 = df2["Cena"]

    # split tylko na dataset1
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=42
    )

    # dopasowanie kategorii city (ważne dla XGBoost)
    X1_test["city"] = X1_test["city"].cat.set_categories(X1_train["city"].cat.categories)
    X2["city"] = X2["city"].cat.set_categories(X1_train["city"].cat.categories)

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X1_train, y1_train, eval_set=[(X1_test, y1_test)], verbose=200)

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X1_train, y1_train,
        eval_set=[(X1_test, y1_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
    )

    # testy
    evaluate(xgb_model, X1_test, y1_test, "TEST na dataset1 (wspólne cechy) XGB")
    evaluate(xgb_model, X2, y2, "TEST na dataset2 (wspólne cechy) XGB")

    evaluate(lgb_model, X1_test, y1_test, "TEST na dataset1 (wspólne cechy) LGB")
    evaluate(lgb_model, X2, y2, "TEST na dataset2 (wspólne cechy) LGB")

    # ważność cech (metryka gain)

    xgb_gain = xgb_model.get_booster().get_score(importance_type="gain")

    xgb_fi = (
        pd.DataFrame({
            "feature": xgb_gain.keys(),
            "gain": xgb_gain.values()
        })
        .sort_values("gain", ascending=False)
    )

    xgb_fi["gain_norm"] = xgb_fi["gain"] / xgb_fi["gain"].sum()

    print("\nXGBoost - ważność cech (metryka gain):")
    print(xgb_fi)


    lgb_fi = pd.DataFrame({
        "feature": lgb_model.feature_name_,
        "gain": lgb_model.booster_.feature_importance(importance_type="gain")
    }).sort_values("gain", ascending=False)

    lgb_fi["gain_norm"] = lgb_fi["gain"] / lgb_fi["gain"].sum()

    print("\nLightGBM - ważność cech (metryka gain):")
    print(lgb_fi)

    # permutacja cech – XGBoost

    perm_xgb = permutation_importance(
        xgb_model,
        X2,
        y2,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    perm_xgb_df = (
        pd.DataFrame({
            "feature": X2.columns,
            "importance": perm_xgb.importances_mean,
            "std": perm_xgb.importances_std
        })
        .sort_values("importance", ascending=False)
    )

    print("\nPermutacja cech - XGBoost:")
    print(perm_xgb_df)


    # permutacja cech – LightGBM

    perm_lgb = permutation_importance(
        lgb_model,
        X2,
        y2,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    perm_lgb_df = (
        pd.DataFrame({
            "feature": X2.columns,
            "importance": perm_lgb.importances_mean,
            "std": perm_lgb.importances_std
        })
        .sort_values("importance", ascending=False)
    )

    print("\nPermutacja cech - LightGBM:")
    print(perm_lgb_df)


if __name__ == "__main__":
    main()
