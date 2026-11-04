"""Auto-generated "Key findings". Every sentence is built from the data in view, so it stays true when
filters or the data change. Returns plain dicts; no Streamlit imports.

Each finding: {"topic": str, "level": "good" | "info" | "warn", "text": str}
"""
from __future__ import annotations

import math

import pandas as pd

from config import FLAT_AOV_PCT, FLAT_MARGIN_PP
from logic import (GROW, cross_sell, factory_concentration, factory_table, growth_table, kpis, months_spanned,
                   pct_change, price_cost_changes, seasonality, where_to_grow, window_label)


def _signed(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}f}%"


def _f(topic: str, level: str, text: str) -> dict:
    return {"topic": topic, "level": level, "text": text}


def growth_findings(cur: pd.DataFrame, prev: pd.DataFrame, windows: dict) -> list[dict]:
    if cur.empty or prev.empty:
        return []
    a, b = kpis(cur), kpis(prev)
    lc, lp = window_label(*windows["cur"]), window_label(*windows["prev"])
    g_sales, g_gp = pct_change(a["sales"], b["sales"]), pct_change(a["gp"], b["gp"])
    out = [_f("Growth", "good" if g_sales >= 0 else "warn",
              f"Sales {_signed(g_sales)} ({lp} ${b['sales']:,.0f} → {lc} ${a['sales']:,.0f}); "
              f"gross profit {_signed(g_gp)}.")]

    g_cust, g_units, g_aov = (pct_change(a[k], b[k]) for k in ("customers", "units", "aov"))
    d_margin = a["margin"] - b["margin"]
    detail = (f"customers {_signed(g_cust)} ({b['customers']:,} → {a['customers']:,}), units {_signed(g_units)}, "
              f"average order value {_signed(g_aov)} (${b['aov']:.2f} → ${a['aov']:.2f}), "
              f"gross margin {a['margin']:.1f}% vs {b['margin']:.1f}% ({d_margin:+.1f} pp)")
    flat = abs(g_aov) < FLAT_AOV_PCT and abs(d_margin) < FLAT_MARGIN_PP
    if flat and g_sales > 0 and g_cust > 0:
        out.append(_f("What drove it", "info", f"Growth came from more customers and volume, not from bigger orders or "
                                               f"better margin: {detail}."))
    else:
        out.append(_f("What drove it", "info", f"{detail[0].upper()}{detail[1:]}."))

    reg = growth_table(cur, prev, "Region") if "Region" in cur.columns else pd.DataFrame()
    reg = reg.dropna(subset=["Growth %"])
    if len(reg) >= 2:
        lo, hi = reg.loc[reg["Growth %"].idxmin()], reg.loc[reg["Growth %"].idxmax()]
        if (reg["Growth %"] > 0).all():
            out.append(_f("Regions", "good", f"Every region grew, from {lo['Region']} {_signed(lo['Growth %'], 0)} "
                                             f"to {hi['Region']} {_signed(hi['Growth %'], 0)}."))
        else:
            shrinking = ", ".join(f"{r['Region']} {_signed(r['Growth %'], 0)}" for _, r in reg[reg["Growth %"] <= 0].iterrows())
            out.append(_f("Regions", "warn", f"Not every region grew: {shrinking}. Best: {hi['Region']} {_signed(hi['Growth %'], 0)}."))
    return out


def concentration_findings(f: pd.DataFrame, prod: pd.DataFrame) -> list[dict]:
    parts = []
    by_div = f.groupby("Division")["Sales"].sum().sort_values(ascending=False)
    if len(by_div) >= 2:
        share = by_div.iloc[0] / by_div.sum() * 100
        if share >= 50:
            parts.append(f"{by_div.index[0]} is {share:.1f}% of sales.")
    fac = factory_table(prod)
    if len(fac) >= 3:
        c = factory_concentration(fac)
        if c["share"] >= 80:
            names = " and ".join(f"{n} ({s:.1f}%)" for n, s in zip(c["names"], c["shares"]))
            parts.append(f"{names} together produce {c['share']:.1f}% of profit.")
    return [_f("Concentration", "warn", " ".join(parts))] if parts else []


def pricing_finding(f: pd.DataFrame) -> list[dict]:
    changed = price_cost_changes(f)
    if not changed:
        return [_f("Pricing", "info", "No product changed price or unit cost in the period, so any margin improvement "
                                      "can only come from deliberate action (repricing, cost renegotiation, mix).")]
    return [_f("Pricing", "info", f"{len(changed)} product(s) show varying unit price or cost: {', '.join(changed[:3])}"
                                  f"{'…' if len(changed) > 3 else ''}.")]


def opportunity_finding(f: pd.DataFrame, prod: pd.DataFrame, min_lines: int) -> list[dict]:
    if len(prod) < 3:
        return []
    w = where_to_grow(prod, min_lines)
    grow = w[w["Where to Grow"] == GROW]
    if grow.empty:
        return []
    r = grow.iloc[0]
    months = months_spanned(f)
    anchor = f.groupby("Division")["Sales"].sum().idxmax()
    a = f[f["Division"] == anchor]
    a_ppu, a_lines = a["Gross Profit"].sum() / a["Units"].sum(), len(a) / months
    a_margin = a["Gross Profit"].sum() / a["Sales"].sum() * 100
    return [_f("Profit per unit", "info",
               f"{r['Product Name']} earns ${r['Profit per Unit']:.2f} profit per unit vs ${a_ppu:.2f} for {anchor}, "
               f"despite a lower margin ({r['Gross Margin %']:.0f}% vs {a_margin:.0f}%). It sells only "
               f"{r['Lines per Month']:.1f} order lines a month vs {a_lines:,.0f} for {anchor}. Doubling it would add "
               f"about {r['If Volume Doubles: +% of Total GP']:.1f}% to total gross profit. "
               "Margin % alone ranks it low; profit per unit ranks it first.")]


def cross_sell_finding(f_cross: pd.DataFrame, min_lines: int) -> list[dict]:
    cs = cross_sell(f_cross, min_lines)
    if not cs or cs["pct_anchor_also_group"] >= cs["pct_group_also_anchor"]:
        return []
    names = ", ".join(cs["products"])
    return [_f("Cross-sell", "info",
               f"{cs['pct_group_also_anchor']:.0f}% of buyers of {names} also bought {cs['anchor']}, but only "
               f"{cs['pct_anchor_also_group']:.1f}% of {cs['anchor']} buyers bought any of them. "
               f"That leaves {cs['anchor_buyers'] - round(cs['anchor_buyers'] * cs['pct_anchor_also_group'] / 100):,} "
               f"{cs['anchor']} customers who have bought none of them.")]


def seasonality_finding(f: pd.DataFrame) -> list[dict]:
    s = seasonality(f)
    if s.empty:
        return []
    peak = s.idxmax(axis=1)
    if peak.nunique() != 1:
        return []
    q = int(peak.iloc[0])
    vals = ", ".join(f"{s.loc[y, q]:.1f}% ({y})" for y in s.index)
    return [_f("Seasonality", "info", f"Q{q} is the peak quarter in every full year: {vals} of annual sales.")]


def customer_finding(validity: dict | None) -> list[dict]:
    if not validity or validity["reliable"]:
        return []
    ret = validity["year_retention"][0] if validity["year_retention"] else None
    tail = (f" Only {ret['returned']:,} of the {ret['customers']:,} customers from {ret['year']} bought again in "
            f"{ret['next_year']}; the data cannot say if that is real or a Customer ID problem.") if ret else ""
    return [_f("Customers", "warn", f"Repeat buying cannot be measured reliably: only {validity['multi_date']:,} of "
                                    f"{validity['customers']:,} customers ({validity['multi_date_share']:.1f}%) ever bought on "
                                    f"two different dates; most 'repeat orders' are products in one basket.{tail}")]


def margin_caveat(f: pd.DataFrame) -> list[dict]:
    if any(c in f.columns for c in ("Freight", "Shipping Cost", "Discount", "Returns")):
        return []
    return [_f("Margin ≠ profit", "warn", "Gross margin overstates profit: there is no freight, discount or returns data "
                                          "(Ship Mode is in the file, shipping cost is not).")]


def generate_findings(f: pd.DataFrame, prod: pd.DataFrame, *, min_lines: int, windows: dict | None = None,
                      cur: pd.DataFrame | None = None, prev: pd.DataFrame | None = None,
                      f_cross: pd.DataFrame | None = None, validity: dict | None = None) -> list[dict]:
    """f = filtered frame, prod = product_table(f); cur/prev/windows = year-over-year slices (or None)."""
    out: list[dict] = []
    if windows and cur is not None and prev is not None:
        out += growth_findings(cur, prev, windows)
    out += concentration_findings(f, prod)
    out += pricing_finding(f)
    out += opportunity_finding(f, prod, min_lines)
    out += cross_sell_finding(f if f_cross is None else f_cross, min_lines)
    out += seasonality_finding(f)
    out += customer_finding(validity)
    out += margin_caveat(f)
    return out
