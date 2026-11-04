"""Unit tests for the dashboard's pure logic (no Streamlit). Run: pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

import config
from config import INSUFFICIENT
from customers import (build_rfm, cohort_retention, customer_validity, new_vs_returning, products_per_visit,
                       validity_messages, visits_table)
from findings import generate_findings
from logic import (ENGINE, GROW, REVIEW, clean_data, cross_sell, factory_concentration, factory_table,
                   growth_table, kpis, margin_is_fixed, mix_rate_bridge, price_cost_changes, product_table,
                   seasonality, where_to_grow, yoy_windows)


def raw_frame(rows):
    cols = ["Row ID", "Order ID", "Order Date", "Customer ID", "Division", "Product Name",
            "Sales", "Units", "Gross Profit", "Cost"]
    return pd.DataFrame(rows, columns=cols)


def row(i, date, cust, product="Nerds", sales=10.0, units=1, cost=5.0, division="Chocolate", oid=None):
    return [i, oid or f"O{i}", date, cust, division, product, sales, units, sales - cost, cost]


def make(rows, **kw):
    return clean_data(raw_frame(rows), **kw)[0]


# --------------------------------------------------------------------------- cleaning / config
def test_clean_standardises_names_dedupes_and_flags_unmapped():
    raw = raw_frame([
        [1, "A-1", "03-01-2024", 1, "Chocolate", "Wonka Bar -Milk Chocolate", 10, 2, 6, 4],
        [2, "A-1", "03-01-2024", 1, "Chocolate", "Wonka Bar -Milk Chocolate", 10, 2, 6, 4],   # exact dup (Row ID differs)
        [3, "A-2", "04-01-2024", 2, "Sugar", "Mystery Candy", 5, 1, 3, 2],                   # not in the product map
    ])
    df, log = clean_data(raw, dedupe=True)
    assert len(df) == 2 and log["Duplicate rows removed"] == 1
    assert log["Sales value in those duplicate rows ($)"] == 10.0
    assert "Wonka Bar - Milk Chocolate" in set(df["Product Name"])
    assert (df.loc[df["Product Name"] == "Mystery Candy", "Factory"] == "Unmapped").all()
    kept, _ = clean_data(raw, dedupe=False)
    assert len(kept) == 3


def test_clean_uses_an_injected_product_map_and_rejects_missing_columns():
    df = make([row(1, "01-06-2025", 1, product="Mystery Candy")],
              product_map={"Mystery Candy": ("Sugar", "Test Factory")})
    assert df.loc[0, "Factory"] == "Test Factory"
    with pytest.raises(ValueError, match="missing required column"):
        clean_data(raw_frame([row(1, "01-06-2025", 1)]).drop(columns=["Cost"]))


def test_product_map_loads_from_json_and_validates(tmp_path):
    assert config.PRODUCT_MAP["Lickable Wallpaper"] == ("Other", "Secret Factory")
    assert len(config.PRODUCT_MAP) == 15
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"products": {"X": {"division": "Sugar"}}}))
    with pytest.raises(ValueError, match="needs both"):
        config.load_product_map(bad)


# --------------------------------------------------------------------------- customer definitions
def test_visit_is_customer_plus_date_not_order_id():
    df = make([row(1, "01-06-2025", 7, "Nerds"), row(2, "01-06-2025", 7, "Fun Dip"),   # same day, 2 products, 2 Order IDs
               row(3, "01-09-2025", 7, "Nerds"), row(4, "01-06-2025", 8, "Nerds")])
    v = visits_table(df)
    c7 = v[v["Customer ID"] == 7]
    assert len(c7) == 2 and list(c7["Visit No"]) == [1, 2] and list(c7["Products"]) == [2, 1]
    assert len(v) == 3


def test_rfm_frequency_counts_distinct_purchase_dates():
    rows = []
    for cust, dates in [(1, ["01-01-2025"]), (2, ["01-01-2025", "01-03-2025"]),
                        (3, [f"01-0{m}-2025" for m in range(1, 6)])]:
        rows += [row(len(rows) + j, d, cust) for j, d in enumerate(dates)]
    # customer 4: five products on ONE date = one visit, not five
    rows += [row(100 + j, "01-02-2025", 4, product=p) for j, p in enumerate(["Nerds", "Fun Dip", "Hair Toffee", "Kazookles", "Wonka Gum"])]
    df = make(rows)
    r = build_rfm(df, df["Order Date"].max() + pd.Timedelta(days=1)).set_index("Customer ID")
    assert [r.loc[c, "Frequency"] for c in (1, 2, 3, 4)] == [1, 2, 5, 1]
    assert r.loc[3, "F_Score"] == 5 and r.loc[4, "F_Score"] == 1


def test_rfm_ties_share_scores_and_order_does_not_matter():
    df = make([row(i, "01-06-2025", 1000 + i) for i in range(50)])
    ref = df["Order Date"].max() + pd.Timedelta(days=1)
    a = build_rfm(df, ref).set_index("Customer ID").sort_index()
    b = build_rfm(df.sample(frac=1, random_state=7), ref).set_index("Customer ID").sort_index()
    assert a["F_Score"].nunique() == 1 and a["R_Score"].nunique() == 1
    assert (a[["R_Score", "F_Score", "M_Score"]] == b[["R_Score", "F_Score", "M_Score"]]).all().all()


def _with_product_id(df):
    df = df.copy()
    df["Product ID"] = df["Product Name"].str[:3].str.upper()
    df["Order ID"] = "US-" + df["Customer ID"].astype(str) + "-" + df["Product ID"]
    return df


def test_validity_flags_basket_splitting_and_hides_unreliable_repeat_metrics():
    # 200 customers, each buys 2 products on one date: looks like "repeat" by Order ID but is one visit
    rows = []
    for c in range(200):
        rows += [row(len(rows), "01-06-2024", c, "Nerds"), row(len(rows) + 1, "01-06-2024", c, "Fun Dip")]
    rows += [row(9000, "01-06-2025", 0, "Nerds")]            # one genuine later-date return, a year later
    v = customer_validity(_with_product_id(make(rows)))
    assert v["multi_order_id_share"] > 90 and v["multi_date"] == 1
    assert v["same_date_share_of_multi_order"] > 99
    assert v["order_id_has_product_share"] == 100
    assert v["year_retention"] == [{"year": 2024, "next_year": 2025, "customers": 200, "returned": 1}]
    assert not v["reliable"]
    text = " ".join(validity_messages(v))
    assert "product lines" in text and "basket-splitting" in text and "Customer IDs" in text


def test_validity_passes_when_customers_really_return():
    rows = []
    for c in range(150):
        rows += [row(len(rows), "01-02-2024", c), row(len(rows) + 1, "01-08-2024", c)]
    v = customer_validity(make(rows))
    assert v["multi_date"] == 150 and v["reliable"]
    assert v["order_id_has_product_share"] is None          # no Product ID column: check is skipped, not crashed


def test_new_vs_returning_uses_first_visit_and_ignores_same_day_second_product():
    df = make([row(1, "05-01-2024", 1, "Nerds"), row(2, "05-01-2024", 1, "Fun Dip"),     # first visit, 2 products
               row(3, "10-03-2024", 1), row(4, "10-03-2024", 2)])                         # c1 returns, c2 is new
    nv = new_vs_returning(df)
    jan, mar = nv.loc["2024-01-01"], nv.loc["2024-03-01"]
    assert jan["New customers"] == 1 and jan["Returning customers"] == 0 and jan["Returning visits"] == 0
    assert mar["New customers"] == 1 and mar["Returning customers"] == 1 and mar["Active customers"] == 2
    assert mar["Returning sales share %"] == 50.0


def test_cohort_retention_blank_when_unobservable_zero_when_observed():
    df = make([row(1, "10-01-2024", "A"), row(2, "10-04-2024", "A"),     # cohort 2024-Q1 returns in Q2
               row(3, "15-01-2024", "B"),
               row(4, "01-07-2024", "C")])                              # cohort 2024-Q3 (also sets end of data)
    size, counts, pct = cohort_retention(df, "Q")
    assert size["2024-Q1"] == 2 and size["2024-Q3"] == 1
    assert pct.loc["2024-Q1", 1] == 50.0 and pct.loc["2024-Q1", 2] == 0.0      # observed, nobody came back
    assert np.isnan(pct.loc["2024-Q3", 1])                                      # not observable yet: not 0%
    assert list(pct.columns) == [0, 1, 2]                                       # no skipped ages


def test_products_per_visit_distribution():
    df = make([row(1, "01-06-2025", 1, "Nerds"), row(2, "01-06-2025", 1, "Fun Dip"), row(3, "01-06-2025", 2, "Nerds")])
    dist, yearly = products_per_visit(df)
    assert dict(zip(dist["Products per visit"], dist["Visits"])) == {"1": 1, "2": 1}
    assert yearly.loc[0, "Products per visit"] == 1.5


# --------------------------------------------------------------------------- products
def test_thin_products_are_not_judged():
    rows = [row(i, "01-06-2025", i, "Kazookles", sales=3.25, cost=3.0, division="Other") for i in range(40)]
    rows += [row(100 + i, "01-06-2025", i, "Fun Dip", sales=1.5, cost=0.9, division="Sugar") for i in range(3)]
    p = product_table(make(rows), threshold=50, min_lines=30).set_index("Product Name")
    assert p.loc["Fun Dip", "Segment"] == INSUFFICIENT
    assert p.loc["Fun Dip", "Margin Flag"] == "Low margin - too few orders to judge"
    assert "Reprice" in p.loc["Kazookles", "Margin Flag"] or "discontinue" in p.loc["Kazookles", "Margin Flag"]


def test_mix_rate_bridge_adds_up_exactly():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        prod = ["Nerds", "Kazookles", "Wonka Bar - Milk Chocolate"][i % 3]
        sales = float(rng.integers(5, 50))
        cost = round(sales * rng.uniform(0.3, 0.9), 2)   # varying margin per line, so rate effects exist
        rows.append(row(i, f"{1 + i % 28:02d}-{1 + i % 6:02d}-2025", i, prod, sales=sales, cost=cost, division="Other"))
    df = make(rows)
    bridge, base = mix_rate_bridge(df)
    assert np.allclose(bridge["Margin %"] - base, bridge["Mix effect (pp)"] + bridge["Rate effect (pp)"])
    assert not margin_is_fixed(df)


def test_price_cost_changes_detects_a_repriced_product():
    rows = [row(i, "01-06-2025", i, "Nerds", sales=10, cost=5) for i in range(5)]
    rows += [row(10 + i, "01-06-2025", 10 + i, "Fun Dip", sales=10 + (i > 2), cost=5) for i in range(5)]
    assert price_cost_changes(make(rows)) == ["Fun Dip"]


def _portfolio():
    rows = [row(i, "01-06-2025", i, "Wonka Bar - Milk Chocolate", sales=7, units=2, cost=2) for i in range(400)]   # volume engine
    rows += [row(1000 + i, "01-06-2025", 1000 + i, "Lickable Wallpaper", sales=20, units=1, cost=10, division="Other") for i in range(40)]
    rows += [row(2000 + i, "01-06-2025", 2000 + i, "Kazookles", sales=1.5, units=3, cost=1.2, division="Other") for i in range(40)]
    rows += [row(3000 + i, "01-06-2025", 3000 + i, "Fun Dip", sales=5, units=1, cost=1, division="Sugar") for i in range(3)]
    return make(rows)


def test_where_to_grow_finds_high_yield_low_volume_and_ranks_disagree():
    p = product_table(_portfolio(), 50, 30)
    w = where_to_grow(p, 30).set_index("Product Name")
    assert w.loc["Lickable Wallpaper", "Where to Grow"] == GROW
    assert w.loc["Wonka Bar - Milk Chocolate", "Where to Grow"] == ENGINE
    assert w.loc["Kazookles", "Where to Grow"] == REVIEW
    assert w.loc["Fun Dip", "Where to Grow"] == INSUFFICIENT and np.isnan(w.loc["Fun Dip", "Rank: Profit/Unit"])
    lw = w.loc["Lickable Wallpaper"]
    assert lw["Rank: Profit/Unit"] == 1 and lw["Rank: Margin %"] == 3 - 1     # #1 on profit/unit, not on margin %
    assert lw["If Volume Doubles: +% of Total GP"] == pytest.approx(lw["GrossProfit"] / p["GrossProfit"].sum() * 100)
    assert where_to_grow(p, 30).iloc[0]["Product Name"] == "Lickable Wallpaper"   # growth lever listed first


def test_cross_sell_is_lopsided_in_the_right_direction():
    rows = []
    for c in range(100):                      # 100 chocolate buyers
        rows.append(row(len(rows), "01-06-2025", c, "Wonka Bar - Milk Chocolate", sales=7, units=2, cost=2))
    for c in range(40):                       # 40 buyers of a non-chocolate product, 30 of them also chocolate
        rows.append(row(len(rows), "01-06-2025", c if c < 30 else 500 + c, "Lickable Wallpaper", division="Other"))
    cs = cross_sell(make(rows), min_lines=30)
    assert cs["anchor"] == "Chocolate" and cs["products"] == ["Lickable Wallpaper"]
    assert cs["pct_group_also_anchor"] == pytest.approx(75.0)
    assert cs["anchor_buyers"] == 100 and cs["pct_anchor_also_group"] == pytest.approx(30.0)   # 30 of the 100 chocolate buyers


def test_factory_concentration_top_two():
    fac = factory_table(product_table(_portfolio(), 50, 30))
    c = factory_concentration(fac)
    assert len(c["names"]) == 2 and c["share"] <= 100 and c["n_factories"] == len(fac)


# --------------------------------------------------------------------------- growth
def test_yoy_windows_cases():
    first = pd.Timestamp("2024-01-02")
    w = yoy_windows("2024-01-02", "2025-12-31", first)               # full range -> last 12 months vs the year before
    assert w["cur"] == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    assert w["prev"] == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    w = yoy_windows("2025-06-01", "2025-12-31", first)               # partial year -> same months a year earlier
    assert w["prev"] == (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-12-31"))
    assert yoy_windows("2024-01-02", "2024-12-31", first) is None    # no prior year in the data


def test_growth_table_and_kpis():
    df = make([row(1, "05-01-2024", 1, sales=10), row(2, "05-01-2024", 1, sales=10, product="Fun Dip"),
               row(3, "05-01-2025", 2, sales=30), row(4, "06-01-2025", 3, sales=30)])
    df["Region"] = ["N", "N", "N", "S"]
    cur, prev = df[df["Order Date"].dt.year == 2025], df[df["Order Date"].dt.year == 2024]
    g = growth_table(cur, prev, "Region").set_index("Region")
    assert g.loc["N", "Growth %"] == pytest.approx(50.0) and np.isnan(g.loc["S", "Growth %"])
    assert g["Share of Growth %"].sum() == pytest.approx(100.0)
    k = kpis(prev)
    assert k["visits"] == 1 and k["aov"] == 20.0 and k["ppv"] == 2.0   # AOV is per visit, not per line (10)


def test_seasonality_ignores_partial_years():
    rows = [row(m, f"15-{m:02d}-2024", m, sales=10) for m in range(1, 13)] + [row(50, "15-01-2025", 99, sales=10)]
    s = seasonality(make(rows))
    assert list(s.index) == [2024] and s.loc[2024].sum() == pytest.approx(100.0)


# --------------------------------------------------------------------------- findings
def test_findings_are_generated_from_data_and_do_not_crash_on_tiny_input():
    df = _portfolio()
    p = product_table(df, 50, 30)
    topics = {x["topic"] for x in generate_findings(df, p, min_lines=30)}
    assert {"Concentration", "Pricing", "Profit per unit", "Cross-sell", "Margin ≠ profit"} & topics
    tiny = df[df["Product Name"] == "Fun Dip"]
    out = generate_findings(tiny, product_table(tiny, 50, 30), min_lines=30)
    assert all(set(x) == {"topic", "level", "text"} for x in out)
