from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_data(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-30")
    dates = pd.date_range(start_date, end_date, freq="D")

    recipes = pd.DataFrame(
        [
            {"product": "servelat", "beef_pct": 45, "pork_pct": 30, "fat_pct": 10, "spices_pct": 8, "casing_pct": 7},
            {"product": "bologna", "beef_pct": 35, "pork_pct": 40, "fat_pct": 12, "spices_pct": 7, "casing_pct": 6},
            {"product": "smoked_sausage", "beef_pct": 25, "pork_pct": 50, "fat_pct": 12, "spices_pct": 8, "casing_pct": 5},
        ]
    )
    recipes.to_csv(output_dir / "recipes.csv", index=False)

    ingredients = ["beef", "pork", "fat", "spices", "casing"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C"]
    prices_rows = []
    for date in dates:
        for ingredient in ingredients:
            base_price = {"beef": 7.2, "pork": 5.4, "fat": 3.8, "spices": 12.0, "casing": 4.1}[ingredient]
            for supplier in suppliers:
                volatility = 0.04 if supplier == "Supplier B" else 0.02
                price = base_price * (1 + volatility * np.sin((date.day + len(ingredient)) / 4)) + (0.02 if supplier == "Supplier C" else 0.0)
                prices_rows.append({"date": date.date(), "ingredient": ingredient, "supplier": supplier, "price": round(price, 3)})
    supplier_prices = pd.DataFrame(prices_rows)
    supplier_prices.to_csv(output_dir / "supplier_prices.csv", index=False)

    sales_rows = []
    for date in dates:
        for product in recipes["product"].tolist():
            base = {"servelat": 220, "bologna": 180, "smoked_sausage": 160}[product]
            seasonality = 1 + 0.12 * np.sin((date.dayofyear / 30) + {"servelat": 0.3, "bologna": 1.1, "smoked_sausage": 1.7}[product])
            weekday_bias = 1.05 if date.dayofweek < 5 else 0.9
            demand = base * seasonality * weekday_bias + np.random.normal(0, 10)
            sales_rows.append({"date": date.date(), "product": product, "kg": max(30, round(demand, 1))})
    sales = pd.DataFrame(sales_rows)
    sales.to_csv(output_dir / "sales.csv", index=False)

    print(f"Synthetic data saved to {output_dir}")


if __name__ == "__main__":
    generate_synthetic_data()
