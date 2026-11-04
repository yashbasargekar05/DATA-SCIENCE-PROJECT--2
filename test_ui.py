"""End-to-end UI tests: run the real app.py with Streamlit's AppTest and interact with the sidebar.

They use a small synthetic CSV (via NASSAU_DATA_FILE) so they do not depend on the real data; one extra test
runs against the real CSV when it is present. Run: pytest -q
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent / "app.py")
REAL_CSV = Path(__file__).parent / "Nassau_Candy_Distributor.csv"

PRODUCTS = [  # name as it appears in the CSV (note the sloppy spacing), id, division, unit price, unit cost, weight
    ("Wonka Bar -Milk Chocolate", "CHO-MIL", "Chocolate", 3.25, 1.14, 30),
    ("Wonka Bar - Fudge Mallows", "CHO-FUD", "Chocolate", 3.60, 1.20, 30),
    ("Wonka Bar - Scrumdiddlyumptious", "CHO-SCR", "Chocolate", 3.60, 1.10, 30),
    ("Lickable Wallpaper", "OTH-LIC", "Other", 20.0, 10.0, 3),
    ("Kazookles", "OTH-KAZ", "Other", 3.25, 3.0, 3),
    ("Nerds", "SUG-NER", "Sugar", 1.5, 0.8, 1),
]
COLUMNS = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Country/Region", "City",
           "State/Province", "Postal Code", "Division", "Region", "Product ID", "Product Name", "Sales", "Units",
           "Gross Profit", "Cost"]


def synthetic_csv(path, n=900, seed=3, drop_column=None, customers=5000):
    rng = np.random.default_rng(seed)
    weights = np.array([p[5] for p in PRODUCTS], float)
    weights /= weights.sum()
    start = pd.Timestamp("2024-01-02")
    rows = []
    for i in range(n):
        name, pid, div, price, cost, _ = PRODUCTS[rng.choice(len(PRODUCTS), p=weights)]
        # 2025 gets ~1.5x the rows of 2024. A big customer pool means few repeat dates, like the real data
        day = int(rng.integers(0, 365)) + (365 if rng.random() < 0.6 else 0)
        date = start + pd.Timedelta(days=day)
        cust = 100000 + int(rng.integers(0, customers))
        units = int(rng.integers(1, 4))
        sales, tot_cost = round(price * units, 2), round(cost * units, 2)
        rows.append([i + 1, f"US-2021-{cust}-{pid}", date.strftime("%d-%m-%Y"), "30-06-2026", "Standard Class", cust,
                     "United States", "Houston", f"State{cust % 6}", "77095", div, ["Atlantic", "Gulf", "Interior", "Pacific"][cust % 4],
                     pid, name, sales, units, round(sales - tot_cost, 2), tot_cost])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = pd.concat([df, df.iloc[:5].assign(**{"Row ID": range(n + 1, n + 6)})])   # 5 exact duplicates
    if drop_column:
        df = df.drop(columns=[drop_column])
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NASSAU_DATA_FILE", str(synthetic_csv(tmp_path / "data.csv")))
    return AppTest.from_file(APP, default_timeout=120).run()


def texts(elements):
    return " ".join(str(getattr(e, "value", e)) for e in elements)


def test_app_renders_every_tab_without_errors(app):
    assert not app.exception and not app.error
    assert [t.label for t in app.tabs] == ["Product Profitability", "Division Performance", "Cost vs Margin",
                                           "Where to Grow", "Profit Concentration", "Customers", "Data & Notes"]
    assert "Key findings" in texts(app.subheader)


def test_kpi_cards_show_year_over_year_change(app):
    cards = {m.label: m for m in app.metric}
    for label in ("Total Sales", "Gross Profit", "Gross Margin", "Profit per Unit", "Customers",
                  "Avg order value (per visit)", "Products per visit"):
        assert cards[label].delta and "YoY" in cards[label].delta, label
    assert cards["Total Sales"].delta.startswith("+")            # synthetic 2025 has more rows than 2024
    assert cards["Gross Margin"].delta.endswith("pp YoY")


def test_yoy_delta_disappears_when_the_prior_year_is_not_in_the_data(app):
    app.sidebar.date_input[0].set_value((dt.date(2024, 1, 2), dt.date(2024, 12, 31))).run()
    assert not app.exception
    assert all(not m.delta for m in app.metric if m.label in ("Total Sales", "Gross Profit"))
    assert "No YoY change shown" in texts(app.caption)


def test_key_findings_panel_and_data_validity_warning(app):
    findings = texts(app.markdown)
    assert "Profit per unit" in findings and "Pricing" in findings and "Margin ≠ profit" in findings
    warn = texts(app.warning)
    assert "Data validity" in warn and "product lines" in warn          # the Customers-tab warning
    assert "RFM segmentation is hidden" in texts(app.info)             # no At Risk / Champions on invalid data


def test_filtering_to_one_division_keeps_cross_division_views_working(app):
    app.sidebar.multiselect[0].set_value(["Sugar"]).run()
    assert not app.exception and not app.error
    assert any(m.label == "Total Sales" for m in app.metric)


def test_selecting_a_single_product(app):
    app.sidebar.selectbox[0].set_value("Lickable Wallpaper").run()
    assert not app.exception
    assert "A single product is selected" in texts(app.info)
    assert "Growth views need at least 3 products" in texts(app.info)


def test_emptying_the_division_filter_shows_a_warning_not_a_crash(app):
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert "No data for the selected filters" in texts(app.warning)


def test_duplicate_toggle_and_thresholds_rerun_cleanly(app):
    before = {m.label: m.value for m in app.metric}["Total Sales"]
    app.sidebar.checkbox[0].set_value(False).run()                    # keep the 5 duplicate rows
    assert not app.exception
    after = {m.label: m.value for m in app.metric}["Total Sales"]
    assert before != after
    app.sidebar.slider[0].set_value(80).run()
    app.sidebar.number_input[0].set_value(900).run()                  # nothing is "reliable" any more
    assert not app.exception and not app.error


def test_a_csv_missing_a_required_column_shows_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NASSAU_DATA_FILE", str(synthetic_csv(tmp_path / "bad.csv", drop_column="Cost")))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "missing required column" in texts(at.error) and "Cost" in texts(at.error)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real CSV not present")
def test_real_data_headline_numbers(monkeypatch):
    monkeypatch.setenv("NASSAU_DATA_FILE", str(REAL_CSV))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception and not at.error
    cards = {m.label: m for m in at.metric}
    assert cards["Total Sales"].delta == "+44.4% YoY" and cards["Customers"].delta == "+49.3% YoY"
    assert cards["Gross Margin"].delta == "+0.1 pp YoY"
    text = texts(at.markdown) + texts(at.warning)
    assert "Lickable Wallpaper earns $10.00" in text.replace(r"\$", "$")
    assert "Only 21 of 5,044 customers" in text and "Only 20 of the 2,031" in text


def test_rfm_returns_when_customers_genuinely_repeat(tmp_path, monkeypatch):
    # a small customer pool with many rows => plenty of customers buy on several different dates
    monkeypatch.setenv("NASSAU_DATA_FILE", str(synthetic_csv(tmp_path / "repeat.csv", n=1500, customers=150)))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and not at.error
    assert "Data-validity check passed" in texts(at.success)
    assert "Data validity" not in texts(at.warning)
    assert "RFM segmentation is hidden" not in texts(at.info)
