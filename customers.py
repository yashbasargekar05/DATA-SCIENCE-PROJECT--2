"""Customer analysis. No Streamlit imports.

Definitions (important): in this file every Order ID embeds a product code, so an "Order ID" is really one
product line. A *visit* is therefore Customer ID + Order Date, and *frequency* is the number of distinct
purchase dates. Everything below is built on visits, never on Order IDs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HIGH_F, MIN_REPEAT_CUSTOMERS, MIN_REPEAT_SHARE, REPEAT_F


def visits_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per customer per purchase date, with a running visit number (1 = first ever visit)."""
    g = d.groupby(["Customer ID", "Order Date"], as_index=False).agg(
        Sales=("Sales", "sum"), GrossProfit=("Gross Profit", "sum"), Units=("Units", "sum"),
        Lines=("Product Name", "size"), Products=("Product Name", "nunique"))
    g = g.sort_values(["Customer ID", "Order Date"]).reset_index(drop=True)
    g["Visit No"] = g.groupby("Customer ID").cumcount() + 1
    return g


def customer_validity(d: pd.DataFrame) -> dict:
    """Facts that decide whether repeat-purchase analysis (RFM, retention) can be trusted on this data."""
    oid = d["Order ID"].astype(str)
    out: dict = {"rows": len(d), "order_ids": int(oid.nunique())}
    out["single_line_order_share"] = float((oid.map(oid.value_counts()) == 1).mean() * 100)
    if "Product ID" in d.columns:
        out["order_id_has_product_share"] = float(np.mean([p in o for p, o in zip(d["Product ID"].astype(str), oid)]) * 100)
    else:
        out["order_id_has_product_share"] = None

    per = d.groupby("Customer ID").agg(order_ids=("Order ID", "nunique"), dates=("Order Date", "nunique"))
    multi_oid, multi_date = per["order_ids"] > 1, per["dates"] > 1
    n = len(per)
    out.update(customers=n, multi_order_id=int(multi_oid.sum()), multi_date=int(multi_date.sum()))
    out["multi_order_id_share"] = out["multi_order_id"] / n * 100 if n else 0.0
    out["multi_date_share"] = out["multi_date"] / n * 100 if n else 0.0
    out["same_date_share_of_multi_order"] = (
        float((per.loc[multi_oid, "dates"] == 1).mean() * 100) if multi_oid.any() else None)

    gaps = visits_table(d).groupby("Customer ID")["Order Date"].diff().dt.days.dropna()
    out["gap_min_days"] = int(gaps.min()) if len(gaps) else None
    out["gap_max_days"] = int(gaps.max()) if len(gaps) else None

    by_year = d.groupby(d["Order Date"].dt.year)["Customer ID"].agg(lambda s: set(s.unique()))
    out["year_retention"] = [
        {"year": int(y), "next_year": int(y) + 1, "customers": len(by_year[y]),
         "returned": len(by_year[y] & by_year[y + 1])}
        for y in by_year.index if (y + 1) in by_year.index]
    out["reliable"] = out["multi_date"] >= MIN_REPEAT_CUSTOMERS and out["multi_date_share"] >= MIN_REPEAT_SHARE
    return out


def validity_messages(v: dict) -> list[str]:
    """Plain-English data-validity warnings, generated from the numbers so they stay true if the data changes."""
    msgs = []
    embeds = v.get("order_id_has_product_share")
    if (embeds or 0) >= 99 or v["single_line_order_share"] >= 99:
        basis = f"{embeds:.0f}% of rows embed the product code in the Order ID" if embeds else \
            f"{v['single_line_order_share']:.0f}% of Order IDs have a single line"
        msgs.append(f"**Order IDs are product lines, not baskets.** {basis}, so counting Order IDs counts products "
                    "per visit, not repeat visits. This dashboard defines a visit as Customer + Order Date.")
    if v["same_date_share_of_multi_order"] is not None and v["multi_order_id"]:
        gap = (f" Their gaps are {v['gap_min_days']} to {v['gap_max_days']} days." if v["gap_min_days"] is not None else "")
        msgs.append(f"**\"Repeat\" is mostly basket-splitting.** {v['multi_order_id_share']:.1f}% of customers have 2+ "
                    f"Order IDs, but {v['same_date_share_of_multi_order']:.1f}% of those placed them all on one date. Only "
                    f"{v['multi_date']:,} of {v['customers']:,} customers ({v['multi_date_share']:.1f}%) ever bought on two "
                    f"different dates.{gap}")
    for r in v["year_retention"]:
        if r["customers"] and r["returned"] / r["customers"] * 100 >= MIN_REPEAT_SHARE:
            continue  # retention is healthy: no caveat needed
        msgs.append(f"**Near-zero cross-year retention.** Only {r['returned']:,} of the {r['customers']:,} customers who "
                    f"bought in {r['year']} bought again in {r['next_year']}. Either customers genuinely do not return, or "
                    "Customer IDs are not persistent across years. The data cannot say which: ask the data owner.")
    return msgs


# --------------------------------------------------------------------------- new vs returning / cohorts
def new_vs_returning(d: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Per period: customers/visits/sales from first-ever visits (New) vs later-date visits (Returning).

    Pass the customer's FULL history so 'first visit' is not distorted by a date filter. A customer who makes a
    first visit and a later-date visit in the same period counts once in each column; 'Active customers' counts
    them once. History starts when the data starts, so everyone in the first period is New by construction.
    """
    v = visits_table(d)
    v["Period"] = v["Order Date"].dt.to_period(freq).dt.to_timestamp()
    v["Type"] = np.where(v["Visit No"] == 1, "New", "Returning")
    t = v.groupby(["Period", "Type"]).agg(Customers=("Customer ID", "nunique"), Visits=("Customer ID", "size"),
                                         Sales=("Sales", "sum")).unstack("Type", fill_value=0)
    t.columns = [f"{typ} {metric.lower()}" for metric, typ in t.columns]
    for c in [f"{t_} {m}" for m in ("customers", "visits", "sales") for t_ in ("New", "Returning")]:
        if c not in t.columns:
            t[c] = 0
    t["Active customers"] = v.groupby("Period")["Customer ID"].nunique()
    t["Returning sales share %"] = t["Returning sales"] / (t["New sales"] + t["Returning sales"]) * 100
    order = ["New customers", "Returning customers", "Active customers", "New visits", "Returning visits",
             "New sales", "Returning sales", "Returning sales share %"]
    return t[order]


def cohort_retention(d: pd.DataFrame, freq: str = "Q"):
    """Cohort = period of a customer's first visit. Returns (cohort sizes, retained counts, retention %).

    Columns are periods since the first visit (0 = cohort period itself). Cells that fall after the end of the
    data are NaN, so a young cohort is never shown as 0% retention.
    """
    ppy = {"M": 12, "Q": 4}[freq]
    step = 12 // ppy
    v = visits_table(d)
    v["p"] = v["Order Date"].dt.year * ppy + (v["Order Date"].dt.month - 1) // step
    v["cohort"] = v.groupby("Customer ID")["p"].transform("min")
    v["age"] = v["p"] - v["cohort"]
    size = v[v["Visit No"] == 1].groupby("cohort")["Customer ID"].nunique()
    active = v.groupby(["cohort", "age"])["Customer ID"].nunique().unstack(fill_value=0).astype(float)
    last = v["p"].max()
    active = active.reindex(columns=range(int(last - v["cohort"].min()) + 1), fill_value=0.0)
    pct = active.div(size, axis=0) * 100
    for c in pct.index:
        pct.loc[c, pct.columns > last - c] = np.nan
        active.loc[c, active.columns > last - c] = np.nan

    def label(i):
        y, k = divmod(int(i), ppy)
        return f"{y}-Q{k + 1}" if freq == "Q" else f"{y}-{k + 1:02d}"
    for frame in (size, active, pct):
        frame.index = [label(i) for i in frame.index]
    return size.rename("Cohort size"), active, pct


def products_per_visit(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(distribution of distinct products per visit, yearly averages of products, sales and units per visit)."""
    v = visits_table(d)
    dist = v["Products"].clip(upper=5).value_counts().sort_index().rename_axis("Products per visit").reset_index(name="Visits")
    dist["Products per visit"] = dist["Products per visit"].map(lambda p: f"{p}+" if p == 5 else str(p))
    dist["Share %"] = dist["Visits"] / dist["Visits"].sum() * 100
    yearly = v.groupby(v["Order Date"].dt.year).agg(
        Visits=("Customer ID", "size"), **{"Products per visit": ("Products", "mean"),
                                           "Sales per visit": ("Sales", "mean"),
                                           "Units per visit": ("Units", "mean")})
    yearly.index.name = "Year"
    return dist, yearly.reset_index()


# --------------------------------------------------------------------------- RFM (only meaningful if repeat buying is real)
def build_rfm(data: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Customer-level RFM. Frequency = distinct purchase dates (visits), NOT distinct Order IDs.

    R and M use percentile ranks where ties share a score (no arbitrary tie-breaking); F is scored on the
    actual visit count (1, 2, 3, 4, 5+).
    """
    rfm = data.groupby("Customer ID", as_index=False).agg(
        Last_Order_Date=("Order Date", "max"),
        Frequency=("Order Date", "nunique"),
        Monetary=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
    )
    rfm["Recency"] = (reference_date - rfm["Last_Order_Date"]).dt.days.clip(lower=0)
    rfm["Customer Margin %"] = rfm["Gross_Profit"] / rfm["Monetary"] * 100

    if len(rfm) < 2:
        rfm["R_Score"] = rfm["F_Score"] = rfm["M_Score"] = 3
    else:
        rfm["R_Score"] = 6 - np.ceil(rfm["Recency"].rank(method="average", pct=True) * 5).clip(1, 5).astype(int)
        rfm["M_Score"] = np.ceil(rfm["Monetary"].rank(method="average", pct=True) * 5).clip(1, 5).astype(int)
        rfm["F_Score"] = pd.cut(rfm["Frequency"], [0, 1, 2, 3, 4, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= HIGH_F and m >= 4:
            return "Champions"
        if r >= 3 and f >= HIGH_F:
            return "Loyal Customers"
        if r >= 4 and f >= REPEAT_F:
            return "Potential Loyalists"
        if r >= 4:
            return "Recent Customers"
        if r <= 2 and f >= HIGH_F:
            return "Cannot Lose Them"
        if r <= 2 and f >= REPEAT_F:
            return "At Risk"
        if r <= 2:
            return "Hibernating"
        return "Needs Attention"

    rfm["RFM Segment"] = rfm.apply(segment, axis=1)
    rfm["RFM Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return rfm.sort_values(["RFM Score", "Monetary"], ascending=False).reset_index(drop=True)
