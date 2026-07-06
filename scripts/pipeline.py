from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "generated"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or DATA_DIR
    recipes = pd.read_csv(data_dir / "recipes.csv")
    supplier_prices = pd.read_csv(data_dir / "supplier_prices.csv")
    sales = pd.read_csv(data_dir / "sales.csv")

    recipes["product"] = recipes["product"].astype(str)
    supplier_prices["date"] = pd.to_datetime(supplier_prices["date"])
    sales["date"] = pd.to_datetime(sales["date"])
    supplier_prices["ingredient"] = supplier_prices["ingredient"].astype(str)
    sales["product"] = sales["product"].astype(str)
    return recipes, supplier_prices, sales


def prepare_cost_dataset(recipes: pd.DataFrame, supplier_prices: pd.DataFrame) -> pd.DataFrame:
    recipes_long = recipes.melt(id_vars="product", var_name="ingredient_col", value_name="pct")
    recipes_long["ingredient"] = recipes_long["ingredient_col"].str.replace("_pct", "")
    recipes_long = recipes_long[["product", "ingredient", "pct"]]

    supplier_prices = supplier_prices.sort_values(["date", "ingredient", "price"]).copy()
    price_by_day = supplier_prices.groupby(["date", "ingredient"], as_index=False)["price"].min()
    price_by_day = price_by_day.rename(columns={"price": "price_per_kg"})

    dates = pd.DataFrame({"date": price_by_day["date"].unique()})
    recipes_long = recipes_long.assign(key=1)
    recipe_dates = recipes_long.merge(dates.assign(key=1), on="key")
    del recipe_dates["key"]

    merged = recipe_dates.merge(price_by_day, on=["date", "ingredient"], how="left")
    merged["ingredient_cost"] = merged["pct"] / 100 * merged["price_per_kg"]
    cost_by_day = merged.groupby(["date", "product"], as_index=False)["ingredient_cost"].sum()
    cost_by_day = cost_by_day.rename(columns={"ingredient_cost": "cost_per_kg"})
    return cost_by_day


def build_profitability_dataset(recipes: pd.DataFrame, supplier_prices: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    cost_by_day = prepare_cost_dataset(recipes, supplier_prices)
    combined = sales.merge(cost_by_day, on=["date", "product"], how="left")
    combined["selling_price_per_kg"] = combined["cost_per_kg"] * 1.35
    combined["gross_margin_per_kg"] = combined["selling_price_per_kg"] - combined["cost_per_kg"]
    combined["profit"] = combined["kg"] * combined["gross_margin_per_kg"]
    combined["month"] = combined["date"].dt.month
    return combined.sort_values(["product", "date"]).reset_index(drop=True)


def optimize_supplier_selection(recipes: pd.DataFrame, supplier_prices: pd.DataFrame, target_date: str | pd.Timestamp, product: str) -> dict:
    target_dt = pd.Timestamp(target_date)
    recipe = recipes.loc[recipes["product"] == product].iloc[0]
    ingredient_cols = ["beef_pct", "pork_pct", "fat_pct", "spices_pct", "casing_pct"]
    ingredient_names = [col.replace("_pct", "") for col in ingredient_cols]

    day_prices = supplier_prices[supplier_prices["date"] == target_dt].copy()
    selected = []
    total_cost = 0.0
    for ingredient_name, ingredient_col in zip(ingredient_names, ingredient_cols):
        candidate = day_prices[day_prices["ingredient"] == ingredient_name]
        if candidate.empty:
            raise ValueError(f"No supplier prices found for {ingredient_name} on {target_dt.date()}")
        best = candidate.sort_values("price").iloc[0]
        selected.append({"ingredient": ingredient_name, "supplier": best["supplier"], "price": float(best["price"])})
        total_cost += float(recipe[ingredient_col]) / 100 * float(best["price"])
    return {"product": product, "date": target_dt.date().isoformat(), "selected": selected, "estimated_cost_per_kg": round(total_cost, 3)}


def build_demand_features(sales: pd.DataFrame) -> pd.DataFrame:
    sales = sales.sort_values(["product", "date"]).copy()
    sales["day_of_week"] = sales["date"].dt.dayofweek
    sales["month"] = sales["date"].dt.month
    sales["trend"] = sales.groupby("product").cumcount()
    sales["rolling_3d"] = sales.groupby("product")["kg"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    sales["target"] = sales.groupby("product")["kg"].shift(-7)
    return sales.dropna(subset=["target"]).reset_index(drop=True)


def train_forecast_model(sales: pd.DataFrame, model_path: Path | None = None) -> tuple[RandomForestRegressor, list[str]]:
    prepared = build_demand_features(sales)
    feature_columns = ["product", "day_of_week", "month", "trend", "rolling_3d"]
    X = pd.get_dummies(prepared[feature_columns], columns=["product"], dtype=float)
    y = prepared["target"]
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, model_path)

    return model, X.columns.tolist()


def predict_next_week(product: str, forecast_date: str | pd.Timestamp, sales: pd.DataFrame, model_path: Path | None = None) -> float:
    model_path = model_path or MODELS_DIR / "forecast_model.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    sales = sales.sort_values(["product", "date"]).copy()
    history = sales[sales["product"] == product]
    if history.empty:
        raise ValueError(f"No historical data for product {product}")

    forecast_dt = pd.Timestamp(forecast_date)
    row = pd.DataFrame(
        [{
            "product": product,
            "day_of_week": forecast_dt.dayofweek,
            "month": forecast_dt.month,
            "trend": len(history) - 1,
            "rolling_3d": float(history.tail(3)["kg"].mean()),
        }]
    )
    encoded = pd.get_dummies(row, columns=["product"], dtype=float)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)
    prediction = model.predict(encoded)[0]
    return round(float(prediction), 1)


def evaluate_model(sales: pd.DataFrame) -> dict:
    prepared = build_demand_features(sales)
    features = ["day_of_week", "month", "trend", "rolling_3d"]
    X = pd.get_dummies(prepared[["product", *features]], columns=["product"], dtype=float)
    y = prepared["target"]
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)
    predicted = model.predict(X)
    return {
        "mae": round(float(mean_absolute_error(y, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y, predicted))), 2),
    }
