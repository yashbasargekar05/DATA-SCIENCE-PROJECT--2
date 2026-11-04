"""Pure data logic for the Nassau Candy dashboard. No Streamlit imports, so it is easy to test."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from config import INSUFFICIENT, MIN_LINES_DEFAULT, MIN_VOL_MONTHS, YIELD_MULT


def short(name: str) -> str:
    return name.replace("Wonka Bar - ", "WB: ")


# --------------------------------------------------------------------------- cleaning
def clean_data(raw: pd.DataFrame, dedupe: bool = True,
               product_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean the raw CSV frame and return (clean frame, step-by-step log)."""
    product_map = config.PRODUCT_MAP if product_map is None else product_map
    missing = [c for c in config.CORE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"The CSV is missing required column(s): {', '.join(missing)}")
    log = {"Raw rows": len(raw)}
    df = raw.copy()

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    names = df["Product Name"]
    df["Product Name"] = names.str.replace(r"\s*-\s*", " - ", regex=True)
    log["Product names standardised (spacing around '-')"] = int((names != df["Product Name"]).fillna(False).sum())
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")

    n = len(df)
    df = df.dropna(subset=["Order Date", "Sales", "Cost"])
    log["Dropped: missing/unparseable Order Date, Sales or Cost"] = n - len(df)
    log["Units imputed with the median (were missing)"] = int(df["Units"].isna().sum())
    df["Units"] = df["Units"].fillna(df["Units"].median())
    n = len(df)
    df = df[(df["Sales"] > 0) & (df["Cost"] >= 0) & (df["Units"] > 0)]
    log["Removed zero/invalid sales, cost or units"] = n - len(df)

    recomputed = (df["Sales"] - df["Cost"]).round(2)
    log["Gross Profit mismatches fixed (Sales - Cost)"] = int((recomputed - df["Gross Profit"]).abs().gt(0.01).sum())
    df["Gross Profit"] = recomputed
    df = df[df["Gross Profit"].notna()]

    key = [c for c in df.columns if c != "Row ID"]
    dup = df.duplicated(subset=key, keep="first")
    log["Exact duplicate rows found (ignoring Row ID)"] = int(dup.sum())
    log["Sales value in those duplicate rows ($)"] = round(float(df.loc[dup, "Sales"].sum()), 2)
    if dedupe:
        df = df[~dup]
    log["Duplicate rows removed"] = int(dup.sum()) if dedupe else 0

    df["Division (source)"] = df["Division"]
    df["Division"] = df["Product Name"].map(lambda p: product_map.get(p, (None, None))[0]).fillna(df["Division"])
    df["Factory"] = df["Product Name"].map(lambda p: product_map.get(p, (None, "Unmapped"))[1])
    log["Rows whose Division label was re-mapped to the brief"] = int((df["Division"] != df["Division (source)"]).sum())
    log["Rows with a product missing from the brief's mapping"] = int((df["Factory"] == "Unmapped").sum())

    df["Margin %"] = df["Gross Profit"] / df["Sales"] * 100
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    log["Clean rows"] = len(df)
    return df.reset_index(drop=True), log


def slice_dates(d: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= Order Date <= end (both ends inclusive, whole days)."""
    return d[(d["Order Date"] >= pd.Timestamp(start)) & (d["Order Date"] < pd.Timestamp(end) + pd.Timedelta(days=1))]


def months_spanned(d: pd.DataFrame) -> int:
    lo, hi = d["Order Date"].min(), d["Order Date"].max()
    return max(1, (hi.year - lo.year) * 12 + hi.month - lo.month + 1)


# --------------------------------------------------------------------------- products
def margin_is_fixed(d: pd.DataFrame, tol: float = 0.05) -> bool:
    """True when every product has the same margin on every order line (fixed unit price and cost)."""
    return bool(d.groupby("Product Name")["Margin %"].std().fillna(0).max() < tol)


def price_cost_changes(d: pd.DataFrame, tol: float = 0.02) -> list[str]:
    """Products whose unit price or unit cost varies by more than `tol` dollars across order lines."""
    u = d.assign(up=d["Sales"] / d["Units"], uc=d["Cost"] / d["Units"]).groupby("Product Name")
    spread = pd.DataFrame({"up": u["up"].max() - u["up"].min(), "uc": u["uc"].max() - u["uc"].min()})
    return sorted(spread.index[(spread["up"] > tol) | (spread["uc"] > tol)])


def sales_cut(prod: pd.DataFrame, min_lines: int):
    """'High sales' = at/above an equal revenue share among products with enough order lines."""
    rel = prod[prod["Lines"] >= min_lines]
    return float(rel["Sales"].mean()) if not rel.empty else None


def product_table(d: pd.DataFrame, threshold: float, min_lines: int = MIN_LINES_DEFAULT,
                  months: int | None = None) -> pd.DataFrame:
    months = months or months_spanned(d)
    g = d.groupby(["Product Name", "Division", "Factory"], as_index=False).agg(
        Sales=("Sales", "sum"), Units=("Units", "sum"), Cost=("Cost", "sum"),
        GrossProfit=("Gross Profit", "sum"), Customers=("Customer ID", "nunique"), Lines=("Order ID", "size"),
    )
    g["Gross Margin %"] = g["GrossProfit"] / g["Sales"] * 100
    g["Profit per Unit"] = g["GrossProfit"] / g["Units"]
    g["Lines per Month"] = g["Lines"] / months
    g["Revenue Contribution %"] = g["Sales"] / g["Sales"].sum() * 100
    g["Profit Contribution %"] = g["GrossProfit"] / g["GrossProfit"].sum() * 100
    g["Cost / Sales %"] = g["Cost"] / g["Sales"] * 100

    # Margin volatility = std dev of monthly gross margin (percentage points)
    m = d.groupby(["Product Name", "Month"]).agg(s=("Sales", "sum"), p=("Gross Profit", "sum"))
    m["mg"] = m["p"] / m["s"] * 100
    grp = m.groupby("Product Name")["mg"]
    vol = grp.std().where(grp.count() >= MIN_VOL_MONTHS).rename("Margin Volatility (pp)")
    g = g.merge(vol, on="Product Name", how="left")
    g["Short"] = g["Product Name"].map(short)

    cut = sales_cut(g, min_lines)

    def seg(r):
        if r["Lines"] < min_lines:
            return INSUFFICIENT
        hs, hm = r["Sales"] >= cut, r["Gross Margin %"] >= threshold
        return ("Star: high sales, high margin" if hs and hm else
                "Volume trap: high sales, low margin" if hs else
                "Niche earner: low sales, high margin" if hm else
                "Underperformer: low sales, low margin")
    g["Segment"] = g.apply(seg, axis=1)

    def action(r):
        if r["Gross Margin %"] >= threshold:
            return "Healthy"
        if r["Lines"] < min_lines:
            return "Low margin - too few orders to judge"
        return "Reprice / renegotiate cost" if r["Sales"] >= cut else "Reprice or discontinue review"
    g["Margin Flag"] = g.apply(action, axis=1)
    return g.sort_values("GrossProfit", ascending=False).reset_index(drop=True)


GROW = "Grow: high profit/unit, low volume"
DEFEND = "Defend: high profit/unit, high volume"
ENGINE = "Volume engine: average profit/unit, high volume"
REVIEW = "Review: low profit/unit, low volume"
TIER_ORDER = {GROW: 0, DEFEND: 1, ENGINE: 2, REVIEW: 3, INSUFFICIENT: 4}


def where_to_grow(prod: pd.DataFrame, min_lines: int = MIN_LINES_DEFAULT) -> pd.DataFrame:
    """Rank products by growth headroom. Margin % and profit per unit often disagree, so both ranks are shown.

    'High profit/unit' = at least YIELD_MULT x the portfolio's gross profit per unit. 'Low volume' = order
    lines per month below an equal share among products with enough data (the same idea as 'high sales'). 'If volume doubles' = the product's
    share of total gross profit, i.e. what one extra copy of its current profit would add.
    """
    p = prod.copy()
    portfolio_ppu = p["GrossProfit"].sum() / p["Units"].sum()
    reliable = p["Lines"] >= min_lines
    equal_share_vol = p.loc[reliable, "Lines per Month"].mean() if reliable.any() else np.nan
    p["Profit/Unit vs Portfolio (x)"] = p["Profit per Unit"] / portfolio_ppu
    p["If Volume Doubles: +% of Total GP"] = p["GrossProfit"] / p["GrossProfit"].sum() * 100
    p["Rank: Margin %"] = p["Gross Margin %"].where(reliable).rank(ascending=False, method="min")
    p["Rank: Profit/Unit"] = p["Profit per Unit"].where(reliable).rank(ascending=False, method="min")

    def tier(r):
        if r["Lines"] < min_lines:
            return INSUFFICIENT
        high_yield = r["Profit/Unit vs Portfolio (x)"] >= YIELD_MULT
        low_volume = r["Lines per Month"] < equal_share_vol
        if high_yield:
            return GROW if low_volume else DEFEND
        return REVIEW if low_volume else ENGINE
    p["Where to Grow"] = p.apply(tier, axis=1)
    p["_o"] = p["Where to Grow"].map(TIER_ORDER)
    return (p.sort_values(["_o", "If Volume Doubles: +% of Total GP"], ascending=[True, False])
             .drop(columns="_o").reset_index(drop=True))


# --------------------------------------------------------------------------- margin bridge / concentration
def mix_rate_bridge(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Split each month's gross margin vs the period margin into product-mix and price/cost-rate effects.

    margin_t - base = sum((w_t - W) * M)  [mix]  +  sum(w_t * (m_t - M))  [rate]
    where w = sales weight, m = product margin, W/M = period-wide weight and margin.
    """
    s = d.pivot_table(index="Month", columns="Product Name", values="Sales", aggfunc="sum", fill_value=0)
    p = d.pivot_table(index="Month", columns="Product Name", values="Gross Profit", aggfunc="sum", fill_value=0)
    W = s.sum() / s.values.sum()
    M = p.sum() / s.sum() * 100
    w = s.div(s.sum(axis=1), axis=0)
    m = (p / s.where(s > 0) * 100).fillna(M)  # absent product-months carry zero weight anyway
    out = pd.DataFrame({
        "Margin %": (w * m).sum(axis=1),
        "Mix effect (pp)": ((w - W) * M).sum(axis=1),
        "Rate effect (pp)": (w * (m - M)).sum(axis=1),
    })
    return out, float((W * M).sum())


def pareto(series: pd.Series) -> pd.DataFrame:
    s = series.clip(lower=0).sort_values(ascending=False)
    out = pd.DataFrame({"value": s})
    out["cum_pct"] = out["value"].cumsum() / out["value"].sum() * 100
    out["rank"] = range(1, len(out) + 1)
    return out


def n_for_80(p: pd.DataFrame) -> int:
    return int((p["cum_pct"] < 80).sum() + 1)


def factory_table(prod: pd.DataFrame) -> pd.DataFrame:
    fac = prod.groupby("Factory").agg(Products=("Product Name", "nunique"), Sales=("Sales", "sum"),
                                      GrossProfit=("GrossProfit", "sum"))
    fac["Margin %"] = fac["GrossProfit"] / fac["Sales"] * 100
    fac["Sales Share %"] = fac["Sales"] / fac["Sales"].sum() * 100
    fac["Profit Share %"] = fac["GrossProfit"] / fac["GrossProfit"].sum() * 100
    return fac.sort_values("GrossProfit", ascending=False).reset_index()


def factory_concentration(fac: pd.DataFrame, k: int = 2) -> dict:
    """Profit held by the top-k factories (fac as returned by factory_table)."""
    top = fac.head(k)
    return {"names": top["Factory"].tolist(), "shares": top["Profit Share %"].tolist(),
            "share": float(top["Profit Share %"].sum()), "n_factories": len(fac),
            "hhi": float((fac["Profit Share %"] ** 2).sum())}


# --------------------------------------------------------------------------- growth / year over year
def yoy_windows(start, end, data_min):
    """Pick the window pair for year-over-year comparison, or None if the prior year is not in the data.

    current = the selection, capped at its last 12 months; prior = the same dates one year earlier.
    A 7-day tolerance on the data start covers data that begins a day or two into January.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    cur_start = max(start, end - pd.DateOffset(years=1) + pd.Timedelta(days=1))
    prev_start, prev_end = cur_start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    if prev_start < pd.Timestamp(data_min).normalize() - pd.Timedelta(days=7):
        return None
    return {"cur": (cur_start, end), "prev": (prev_start, prev_end)}


def window_label(start, end) -> str:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.year == end.year and start.month == 1 and start.day <= 7 and end.month == 12 and end.day >= 24:
        return str(start.year)
    return f"{start:%b %Y}–{end:%b %Y}"


def kpis(d: pd.DataFrame) -> dict:
    """Headline numbers. A 'visit' is one customer on one date (the real unit of an order in this data)."""
    if d.empty:
        return {k: np.nan for k in ["sales", "gp", "margin", "ppu", "units", "customers", "visits", "aov", "ppv"]}
    visits = d.groupby(["Customer ID", "Order Date"]).agg(s=("Sales", "sum"), n=("Product Name", "nunique"))
    sales, gp, units = float(d["Sales"].sum()), float(d["Gross Profit"].sum()), float(d["Units"].sum())
    return {"sales": sales, "gp": gp, "margin": gp / sales * 100, "ppu": gp / units, "units": units,
            "customers": int(d["Customer ID"].nunique()), "visits": len(visits),
            "aov": float(visits["s"].mean()), "ppv": float(visits["n"].mean())}


def pct_change(cur: float, prev: float) -> float:
    return (cur / prev - 1) * 100 if prev and not np.isnan(prev) and prev > 0 else float("nan")


def growth_table(cur: pd.DataFrame, prev: pd.DataFrame, by: str, value: str = "Sales") -> pd.DataFrame:
    """Current vs prior-window totals per group, with growth % and each group's share of total change."""
    a, b = cur.groupby(by)[value].sum(), prev.groupby(by)[value].sum()
    out = pd.concat({"Current": a, "Prior": b}, axis=1).fillna(0.0)
    out["Change"] = out["Current"] - out["Prior"]
    out["Growth %"] = np.where(out["Prior"] > 0, out["Current"] / out["Prior"].where(out["Prior"] > 0) * 100 - 100, np.nan)
    total = out["Change"].sum()
    out["Share of Growth %"] = out["Change"] / total * 100 if total else np.nan
    return out.sort_values("Current", ascending=False).reset_index()


def seasonality(d: pd.DataFrame) -> pd.DataFrame:
    """Each complete calendar year's sales split by quarter (% of that year). Partial years are skipped."""
    yr = d["Order Date"].dt.year
    complete = [y for y, m in d.groupby(yr)["Month"].nunique().items() if m == 12]
    if not complete:
        return pd.DataFrame()
    s = d[yr.isin(complete)]
    q = s.groupby([s["Order Date"].dt.year, s["Order Date"].dt.quarter])["Sales"].sum().unstack(fill_value=0)
    return q.div(q.sum(axis=1), axis=0) * 100


# --------------------------------------------------------------------------- cross-sell
def division_affinity(d: pd.DataFrame) -> pd.DataFrame:
    """Row = customers who bought division A (ever); columns = % of them who also bought division B."""
    m = pd.crosstab(d["Customer ID"], d["Division"]) > 0
    rows = {}
    for a in m.columns:
        buyers = m[m[a]]
        rows[a] = {"Buyers": len(buyers), **{b: buyers[b].mean() * 100 for b in m.columns}}
    out = pd.DataFrame(rows).T
    out.index.name = "Buyers of"
    return out


def cross_sell(d: pd.DataFrame, min_lines: int = MIN_LINES_DEFAULT) -> dict | None:
    """Do customers of the smaller lines also buy from the biggest division, and vice versa?

    Anchor = largest division by sales. Group = anchor's non-division products with >= min_lines order lines.
    """
    if d.empty:
        return None
    anchor = d.groupby("Division")["Sales"].sum().idxmax()
    lines = d.groupby("Product Name").size()
    div_of = d.drop_duplicates("Product Name").set_index("Product Name")["Division"]
    group = sorted(p for p in lines.index if div_of[p] != anchor and lines[p] >= min_lines)
    if not group:
        return None
    ct = pd.crosstab(d["Customer ID"], d["Product Name"]) > 0
    anchor_mask = ct.index.isin(d.loc[d["Division"] == anchor, "Customer ID"].unique())
    group_mask = ct[group].any(axis=1).to_numpy()
    if not group_mask.any() or not anchor_mask.any():
        return None
    per_product = pd.DataFrame({
        "Product": group,
        "Buyers": [int(ct[p].sum()) for p in group],
        f"% also bought {anchor}": [float(anchor_mask[ct[p].to_numpy()].mean() * 100) for p in group]})
    return {"anchor": anchor, "products": group, "group_buyers": int(group_mask.sum()),
            "pct_group_also_anchor": float(anchor_mask[group_mask].mean() * 100),
            "anchor_buyers": int(anchor_mask.sum()),
            "pct_anchor_also_group": float(group_mask[anchor_mask].mean() * 100),
            "per_product": per_product}


def ship_mode_table(d: pd.DataFrame) -> pd.DataFrame:
    """Share of sales and gross margin by ship mode. Without freight cost this says nothing about profitability."""
    g = d.groupby("Ship Mode").agg(Lines=("Sales", "size"), Sales=("Sales", "sum"), GrossProfit=("Gross Profit", "sum"))
    g["Share of Sales %"] = g["Sales"] / g["Sales"].sum() * 100
    g["Gross Margin %"] = g["GrossProfit"] / g["Sales"] * 100
    return g.sort_values("Sales", ascending=False).reset_index()
