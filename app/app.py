from pathlib import Path
import sys
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.generate_data import generate_synthetic_data
from scripts.pipeline import (
    build_profitability_dataset,
    evaluate_model,
    load_data,
    optimize_supplier_selection,
    predict_next_week,
    prepare_cost_dataset,
)

st.set_page_config(page_title="MeatProd Analytics", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "recipes.csv").exists():
    generate_synthetic_data(DATA_DIR)

recipes, supplier_prices, sales = load_data(DATA_DIR)
profitability = build_profitability_dataset(recipes, supplier_prices, sales)

st.title("MeatProd Analytics Pipeline")
st.caption("Сквозная аналитика для мясоперерабатывающего цеха")

with st.sidebar:
    st.header("Настройки")
    selected_product = st.selectbox("Продукт", sorted(sales["product"].unique()))
    selected_date = st.date_input("Дата для анализа", value=pd.Timestamp("2024-03-10").date())

st.tabs(["Аналитика цен", "Прогноз спроса", "Рентабельность"])  # type: ignore[arg-type]

with st.container():
    st.subheader("Аналитика себестоимости")
    cost_df = prepare_cost_dataset(recipes, supplier_prices)
    cost_df = cost_df[cost_df["product"] == selected_product]
    st.line_chart(cost_df.set_index("date")["cost_per_kg"])

with st.container():
    st.subheader("Прогноз спроса")
    model_path = Path(__file__).resolve().parents[1] / "models" / "forecast_model.joblib"
    if not model_path.exists():
        from scripts.pipeline import train_forecast_model
        train_forecast_model(sales, model_path)
    predicted = predict_next_week(selected_product, selected_date, sales, model_path)
    st.metric("Прогноз на следующую неделю, кг", f"{predicted:.1f}")

with st.container():
    st.subheader("Рентабельность")
    metrics = evaluate_model(sales)
    st.metric("MAE", metrics["mae"])
    st.metric("RMSE", metrics["rmse"])
    st.dataframe(profitability.groupby("product")["profit"].sum().reset_index().sort_values("profit", ascending=False))

    opt_result = optimize_supplier_selection(recipes, supplier_prices, selected_date, selected_product)
    st.json(opt_result)
