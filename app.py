"""Nassau Candy Distributor - Product Line Profitability & Margin Performance Dashboard.

Entry point only: page setup, sidebar filters and wiring. Logic lives in logic.py / customers.py / findings.py,
drawing in views.py / views_customers.py, settings in config.py, and the product map in product_map.json.
Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st

import config
from customers import customer_validity
from findings import generate_findings
from logic import clean_data, kpis, product_table, slice_dates, yoy_windows
from views import (render_findings, render_kpis, tab_concentration, tab_cost, tab_divisions, tab_grow, tab_notes,
                   tab_products)
from views_customers import tab_customers


@st.cache_data(show_spinner=False)
def load_and_clean(source, dedupe: bool = True, file_stamp: float = 0.0):
    """file_stamp is part of the cache key so an edited CSV is re-read instead of served stale."""
    if hasattr(source, "seek"):
        source.seek(0)  # an uploaded file must be rewound before every re-read
    return clean_data(pd.read_csv(source), dedupe)


def main():
    st.set_page_config(page_title="Nassau Candy | Profitability", page_icon="🍬", layout="wide")
    st.sidebar.title("🍬 Filters")
    dedupe = st.sidebar.checkbox("Remove exact duplicate rows", value=True,
                                 help="Rows identical in every column except Row ID. Untick to keep them.")

    path = config.data_file()
    try:
        if path.exists():
            df, log = load_and_clean(path, dedupe, path.stat().st_mtime)
        else:
            up = st.sidebar.file_uploader("Upload Nassau Candy CSV", type="csv")
            if up is None:
                st.info("Place `Nassau_Candy_Distributor.csv` next to app.py or upload it in the sidebar.")
                st.stop()
            df, log = load_and_clean(up, dedupe, 0.0)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    dmin, dmax = df["Order Date"].min().date(), df["Order Date"].max().date()
    dr = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    divs = st.sidebar.multiselect("Division", sorted(df["Division"].unique()), default=sorted(df["Division"].unique()))
    threshold = st.sidebar.slider("Margin threshold (%)", 0, 90, 50,
                                  help="Products below this gross margin are flagged as margin risks.")
    min_lines = int(st.sidebar.number_input("Min. order lines for a reliable call", 1, 1000, config.MIN_LINES_DEFAULT,
                                            help="Products with fewer order lines are shown as 'Insufficient data' "
                                                 "and never get a reprice/discontinue recommendation."))
    options = sorted(df.loc[df["Division"].isin(divs), "Product Name"].dropna().unique().tolist())
    selected_product = st.sidebar.selectbox("Product", ["All Products"] + options)

    if not (isinstance(dr, (tuple, list)) and len(dr) == 2):
        st.info("Pick an end date to apply the date range.")
        st.stop()
    start, end = dr

    base = df[df["Division"].isin(divs)]
    if selected_product != "All Products":
        base = base[base["Product Name"] == selected_product]
    f = slice_dates(base, start, end)
    if f.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # Year-over-year: last 12 months of the selection vs the same dates a year earlier (same filters)
    windows = yoy_windows(start, end, df["Order Date"].min())
    cur = slice_dates(base, *windows["cur"]) if windows else None
    prev = slice_dates(base, *windows["prev"]) if windows else None
    if windows and (cur.empty or prev.empty):
        windows = cur = prev = None

    prod = product_table(f, threshold, min_lines)
    f_cross = slice_dates(df, start, end)  # cross-division views compare across the Division/Product filters
    validity = customer_validity(df)

    st.title("Product Line Profitability & Margin Performance")
    st.caption("Nassau Candy Distributor | which products really drive profit, and where margin is at risk")
    unmapped = sorted(df.loc[df["Factory"] == "Unmapped", "Product Name"].dropna().unique())
    if unmapped:
        st.warning(f"Products missing from product_map.json (factory shown as 'Unmapped'): {', '.join(unmapped)}")
    if selected_product != "All Products":
        st.info("A single product is selected: portfolio views (segments, Pareto, share-of-total) are limited or hidden.")

    render_findings(generate_findings(f, prod, min_lines=min_lines, windows=windows, cur=cur, prev=prev,
                                      f_cross=f_cross, validity=validity))
    render_kpis(kpis(f), kpis(cur) if windows else None, kpis(prev) if windows else None, windows,
                int((prod["Margin Flag"] != "Healthy").sum()), threshold)

    tabs = st.tabs(["Product Profitability", "Division Performance", "Cost vs Margin", "Where to Grow",
                    "Profit Concentration", "Customers", "Data & Notes"])
    with tabs[0]:
        tab_products(f, prod, threshold, min_lines)
    with tabs[1]:
        tab_divisions(f, cur, prev, windows, f_cross)
    with tabs[2]:
        tab_cost(prod, threshold, min_lines, f["Sales"].sum(), f["Gross Profit"].sum())
    with tabs[3]:
        tab_grow(prod, min_lines, f_cross)
    with tabs[4]:
        tab_concentration(f, prod, cur, prev, windows)
    with tabs[5]:
        tab_customers(df, start, end)
    with tabs[6]:
        tab_notes(df, log, dedupe, min_lines)


if __name__ == "__main__":
    main()
