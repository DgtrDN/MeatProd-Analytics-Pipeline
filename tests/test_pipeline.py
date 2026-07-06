from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.generate_data import generate_synthetic_data
from scripts.pipeline import build_demand_features, build_profitability_dataset, load_data, optimize_supplier_selection, prepare_cost_dataset


def test_pipeline_generates_expected_outputs(tmp_path):
    generate_synthetic_data(tmp_path)
    recipes, supplier_prices, sales = load_data(tmp_path)

    assert not recipes.empty
    assert not supplier_prices.empty
    assert not sales.empty

    cost_df = prepare_cost_dataset(recipes, supplier_prices)
    assert {"date", "product", "cost_per_kg"}.issubset(cost_df.columns)
    assert (cost_df["cost_per_kg"] >= 0).all()

    profitability = build_profitability_dataset(recipes, supplier_prices, sales)
    assert {"profit", "gross_margin_per_kg"}.issubset(profitability.columns)
    assert profitability["profit"].ge(0).all()

    features = build_demand_features(sales)
    assert "target" in features.columns
    assert features["target"].notna().all()

    result = optimize_supplier_selection(recipes, supplier_prices, "2024-01-10", "servelat")
    assert result["estimated_cost_per_kg"] > 0
    assert result["product"] == "servelat"
