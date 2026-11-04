"""Streamlit views for the Nassau Candy dashboard: each function draws one block or tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DIV_COLORS, INSUFFICIENT, MIN_VOL_MONTHS
from logic import (factory_concentration, factory_table, growth_table, margin_is_fixed, mix_rate_bridge, n_for_80,
                   pareto, sales_cut, short, ship_mode_table, where_to_grow, division_affinity, cross_sell,
                   window_label, pct_change)

def show_observation(message: str) -> None:
    """Display a concise, data-driven observation directly below a chart."""
    st.caption(f"**Observation Report:** {message}".replace("$", r"\$"))  # escape $ so pairs are not read as LaTeX


# --------------------------------------------------------------------------- tab 1
def tab_products(f, prod, threshold, min_lines):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Product margin leaderboard")
        lb = prod.sort_values("Gross Margin %").copy()
        lb["Status"] = np.where(lb["Gross Margin %"] >= threshold, "At/above threshold", "Below threshold")
        lb["Label"] = [f"{m:.1f}%{' *' if n < min_lines else ''}" for m, n in zip(lb["Gross Margin %"], lb["Lines"])]
        fig = px.bar(lb, x="Gross Margin %", y="Product Name", orientation="h", color="Status",
                     color_discrete_map={"At/above threshold": "#2A9D8F", "Below threshold": "#D62828"},
                     text="Label", hover_data={"GrossProfit": ":,.0f", "Division": True, "Lines": True, "Status": False})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.add_vline(x=threshold, line_dash="dash", line_color="grey",
                      annotation_text=f"threshold {threshold}%", annotation_position="bottom right")
        fig.update_layout(height=480, yaxis_title=None, xaxis_range=[0, 100], legend_title=None,
                          legend=dict(orientation="h", y=1.06), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        if (lb["Lines"] < min_lines).any():
            st.caption(f"* fewer than {min_lines} order lines: this margin estimate is thin.")
        low, high = lb.iloc[0], lb.iloc[-1]
        below = int((lb["Gross Margin %"] < threshold).sum())
        show_observation(
            f"{high['Product Name']} has the highest gross margin ({high['Gross Margin %']:.1f}%), "
            f"while {low['Product Name']} is lowest ({low['Gross Margin %']:.1f}%). "
            f"{below} of {len(lb)} products are below the {threshold}% threshold.")
    with c2:
        st.subheader("Profit contribution")
        pos = prod[prod["GrossProfit"] > 0]
        fig = px.treemap(pos, path=["Division", "Short"], values="GrossProfit", color="Gross Margin %",
                         color_continuous_scale="RdYlGn", range_color=[0, 80])
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        if len(pos) < len(prod):
            st.caption(f"{len(prod) - len(pos)} product(s) with zero or negative gross profit are not drawn.")
        leader = prod.loc[prod["GrossProfit"].idxmax()]
        show_observation(
            f"{leader['Product Name']} is the largest product-level profit contributor, "
            f"generating ${leader['GrossProfit']:,.0f} ({leader['Profit Contribution %']:.1f}% of gross profit).")

    st.subheader("Revenue vs profit contribution by product")
    long = prod.melt(id_vars="Product Name", value_vars=["Revenue Contribution %", "Profit Contribution %"],
                     var_name="Metric", value_name="Share %")
    long["Product"] = long["Product Name"].map(short)
    fig = px.bar(long, x="Product", y="Share %", color="Metric", barmode="group")
    fig.update_layout(height=380, xaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    gap = prod.assign(_gap=prod["Profit Contribution %"] - prod["Revenue Contribution %"])
    gap_row = gap.loc[gap["_gap"].abs().idxmax()]
    show_observation(
        f"{gap_row['Product Name']} has the largest absolute gap between profit and revenue contribution "
        f"({gap_row['_gap']:+.1f} percentage points); a positive gap means it contributes more profit than sales.")

    st.subheader("Margin volatility (monthly)")
    c1, c2 = st.columns(2)
    with c1:
        valid_vol = prod.dropna(subset=["Margin Volatility (pp)"]).sort_values("Margin Volatility (pp)", ascending=False)
        if valid_vol.empty:
            st.info(f"No product has at least {MIN_VOL_MONTHS} active months for a reliable volatility estimate.")
        elif valid_vol["Margin Volatility (pp)"].max() < 0.05:
            st.info("Every product's monthly margin is flat (largest std dev "
                    f"{valid_vol['Margin Volatility (pp)'].max():.3f} pp), so there is no volatility to rank. "
                    "Each product sells at a fixed unit price and cost; see the mix-vs-rate chart below for what "
                    "actually moves the overall margin.")
        else:
            fig = px.bar(valid_vol, x="Short", y="Margin Volatility (pp)", color="Division",
                         color_discrete_map=DIV_COLORS)
            fig.update_layout(height=340, xaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            mv = valid_vol.iloc[0]
            show_observation(f"{mv['Product Name']} has the highest observed monthly margin volatility "
                             f"({mv['Margin Volatility (pp)']:.1f} percentage points).")
    with c2:
        mo = f.groupby("Month").agg(s=("Sales", "sum"), p=("Gross Profit", "sum"))
        mo["Margin %"] = mo["p"] / mo["s"] * 100
        fig = px.line(mo.reset_index(), x="Month", y="Margin %", markers=True, title="Overall monthly gross margin")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        best_month, worst_month = mo["Margin %"].idxmax(), mo["Margin %"].idxmin()
        show_observation(f"Gross margin peaked in {best_month:%b %Y} ({mo.loc[best_month, 'Margin %']:.1f}%) "
                         f"and was lowest in {worst_month:%b %Y} ({mo.loc[worst_month, 'Margin %']:.1f}%).")
    st.caption("Volatility = standard deviation of a product's monthly gross margin, in percentage points. "
               f"Products active in fewer than {MIN_VOL_MONTHS} months are left out because the figure would be unreliable.")

    st.subheader("What moves the monthly margin? Mix vs rate")
    bridge, base = mix_rate_bridge(f)
    b = bridge.reset_index().melt(id_vars="Month", value_vars=["Mix effect (pp)", "Rate effect (pp)"],
                                  var_name="Effect", value_name="pp vs period margin")
    fig = px.bar(b, x="Month", y="pp vs period margin", color="Effect", barmode="relative",
                 color_discrete_map={"Mix effect (pp)": "#E85D9E", "Rate effect (pp)": "#2A9D8F"})
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    st.plotly_chart(fig, width="stretch")
    mix_max, rate_max = bridge["Mix effect (pp)"].abs().max(), bridge["Rate effect (pp)"].abs().max()
    driver = "product mix" if mix_max >= rate_max else "price/cost (rate) changes"
    show_observation(
        f"Against the period-average margin of {base:.1f}%, monthly swings are driven mostly by {driver} "
        f"(mix up to ±{mix_max:.2f} pp, rate up to ±{rate_max:.2f} pp). "
        "Mix = which products sold; rate = a product's own margin changing.")

    st.subheader("Product scorecard")
    sc = prod.drop(columns=["Short"])
    st.dataframe(sc.style.format({
        "Sales": "${:,.0f}", "Cost": "${:,.0f}", "GrossProfit": "${:,.0f}", "Units": "{:,.0f}", "Lines": "{:,.0f}",
        "Customers": "{:,.0f}", "Lines per Month": "{:.1f}",
        "Gross Margin %": "{:.1f}", "Profit per Unit": "${:.2f}", "Revenue Contribution %": "{:.1f}",
        "Profit Contribution %": "{:.1f}", "Cost / Sales %": "{:.1f}", "Margin Volatility (pp)": "{:.1f}"}),
        width="stretch", hide_index=True)
    st.download_button("Download product scorecard (CSV)", sc.to_csv(index=False).encode("utf-8"),
                       file_name="nassau_candy_product_scorecard.csv", mime="text/csv")


# --------------------------------------------------------------------------- tab 3
def tab_cost(prod, threshold, min_lines, tot_sales, tot_gp):
    st.subheader("Cost vs sales")
    fig = px.scatter(prod, x="Sales", y="Cost", size="Units", color="Gross Margin %", text="Short",
                     color_continuous_scale="RdYlGn", range_color=[0, 80], size_max=40,
                     hover_name="Product Name", hover_data={"Short": False, "Margin Flag": True})
    lo, hi = min(prod["Sales"].min(), prod["Cost"].min()) * 0.7, max(prod["Sales"].max(), prod["Cost"].max()) * 1.4
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Cost = Sales (break-even)",
                             line=dict(dash="dot", color="grey")))
    fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], title="Sales ($, log scale)")
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], title="Cost ($, log scale)")
    fig.update_layout(height=540, legend=dict(orientation="h", y=1.08), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    closest = prod.loc[prod["Gross Margin %"].idxmin()]
    show_observation(
        f"{closest['Product Name']} is closest to the cost-equals-sales break-even line, "
        f"with a {closest['Gross Margin %']:.1f}% gross margin and costs equal to "
        f"{closest['Cost / Sales %']:.1f}% of sales.")
    st.caption("Both axes are log-scaled because product sizes differ by orders of magnitude. "
               "Points near the dotted line keep almost nothing as profit. Bubble size = units sold.")

    st.subheader("Margin risk flags")
    risk = prod[prod["Margin Flag"] != "Healthy"][["Product Name", "Division", "Lines", "Sales", "Cost / Sales %",
                                                   "Gross Margin %", "Margin Flag"]]
    if risk.empty:
        st.success(f"No product is below the {threshold}% margin threshold.")
    else:
        st.dataframe(risk.style.format({"Sales": "${:,.0f}", "Lines": "{:,.0f}", "Cost / Sales %": "{:.1f}",
                                        "Gross Margin %": "{:.1f}"}), width="stretch", hide_index=True)
        st.caption(f"Rule: below threshold with ≥{min_lines} order lines and above-average sales → reprice or renegotiate "
                   f"cost; below threshold with lower sales → reprice or discontinue review; below threshold with "
                   f"<{min_lines} order lines → flagged but not judged (too few orders).")

    st.subheader("What-if: change price or cost")
    sel = st.selectbox("Product", prod["Product Name"].tolist(),
                       index=int(prod["Gross Margin %"].idxmin()), key="whatif_product")
    row = prod.loc[prod["Product Name"] == sel].iloc[0]
    w1, w2 = st.columns(2)
    dp = w1.slider("Price change (%)", -30, 100, 0, key="whatif_price")
    dc = w2.slider("Unit cost change (%)", -60, 30, 0, key="whatif_cost")
    new_sales, new_cost = row["Sales"] * (1 + dp / 100), row["Cost"] * (1 + dc / 100)
    new_gp = new_sales - new_cost
    new_margin = new_gp / new_sales * 100 if new_sales > 0 else float("nan")
    x1, x2, x3 = st.columns(3)
    x1.metric("Gross margin", f"{new_margin:.1f}%", f"{new_margin - row['Gross Margin %']:+.1f} pp")
    x2.metric("Gross profit", f"${new_gp:,.0f}", f"${new_gp - row['GrossProfit']:+,.0f}")
    x3.metric("Share of total gross profit", f"{new_gp / (tot_gp - row['GrossProfit'] + new_gp) * 100:.1f}%",
              f"{new_gp / (tot_gp - row['GrossProfit'] + new_gp) * 100 - row['Profit Contribution %']:+.1f} pp")
    st.caption("Holds units constant (no demand response) and uses the filtered period's totals; a first-pass "
               "sensitivity, not a forecast.")

    st.subheader("Product segments")
    if len(prod) < 3:
        st.info("Segments need at least 3 products in view. Choose 'All Products' to see them.")
        return
    cut = sales_cut(prod, min_lines)
    fig = px.scatter(prod, x="Sales", y="Gross Margin %", color="Segment", text="Short",
                     size=prod["GrossProfit"].clip(lower=1), size_max=40, log_x=True, hover_name="Product Name")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"margin threshold {threshold}%")
    if cut:
        fig.add_vline(x=cut, line_dash="dash", annotation_text="equal revenue share")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, legend=dict(orientation="h", y=-0.2), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    counts = prod["Segment"].value_counts()
    show_observation(f"The largest product segment is **{counts.index[0]}** ({int(counts.iloc[0])} products out of {len(prod)}).")
    st.caption(f"High sales = at/above an equal revenue share among products with ≥{min_lines} order lines; "
               f"high margin = at/above the margin threshold. Thinner products show as '{INSUFFICIENT}'.")




# --------------------------------------------------------------------------- shared blocks
ICON = {"good": "✅", "info": "💡", "warn": "⚠️"}


def md(text: str) -> str:
    """Escape $ so Streamlit does not read two dollar amounts as a LaTeX formula."""
    return text.replace("$", r"\$")


def render_findings(findings: list[dict]) -> None:
    """One-screen auto-generated summary shown above the tabs."""
    with st.container(border=True):
        st.subheader("Key findings")
        st.caption("Generated from the data in view; it follows the sidebar filters.")
        if not findings:
            st.write("Nothing notable for this selection.")
            return
        half = (len(findings) + 1) // 2
        for col, chunk in zip(st.columns(2), (findings[:half], findings[half:])):
            for fd in chunk:
                col.markdown(md(f"{ICON[fd['level']]} **{fd['topic']}.** {fd['text']}"))


def render_kpis(k: dict, kc: dict | None, kp: dict | None, windows: dict | None, n_risk: int, threshold: float) -> None:
    """KPI cards. k = selected period; kc/kp = current/prior year windows used only for the YoY delta."""
    def pct(key):
        if not kc or not kp:
            return None
        v = pct_change(kc[key], kp[key])
        return None if np.isnan(v) else f"{v:+.1f}% YoY"

    yoy_help = (f"Year-over-year: {window_label(*windows['cur'])} vs {window_label(*windows['prev'])}, same filters."
                if windows else "No prior-year data for this selection.")
    r1 = st.columns(4)
    r1[0].metric("Total Sales", f"${k['sales']:,.0f}", pct("sales"), help=yoy_help)
    r1[1].metric("Gross Profit", f"${k['gp']:,.0f}", pct("gp"), help=yoy_help)
    r1[2].metric("Gross Margin", f"{k['margin']:.1f}%",
                 f"{kc['margin'] - kp['margin']:+.1f} pp YoY" if kc and kp else None, help=yoy_help)
    r1[3].metric("Profit per Unit", f"${k['ppu']:.2f}", pct("ppu"), help=yoy_help)
    r2 = st.columns(4)
    r2[0].metric("Customers", f"{k['customers']:,}", pct("customers"), help=yoy_help)
    r2[1].metric("Avg order value (per visit)", f"${k['aov']:.2f}", pct("aov"),
                 help="Sales per customer per purchase date. Order IDs are single product lines in this data, "
                      "so a visit (customer + date) is the real order. " + yoy_help)
    r2[2].metric("Products per visit", f"{k['ppv']:.2f}", pct("ppv"), help=yoy_help)
    r2[3].metric("Margin-risk products", n_risk,
                 help=f"Gross margin below {threshold}% (includes thin-data products that are flagged but not judged)")
    if windows:
        st.caption(f"Cards show the full selected period. YoY change compares {window_label(*windows['cur'])} "
                   f"with {window_label(*windows['prev'])} (same filters).")
    else:
        st.caption("No YoY change shown: the year before this selection is not in the data.")


# --------------------------------------------------------------------------- tab 2
def _affinity_gap(aff: pd.DataFrame):
    """The division pair with the most lopsided cross-buying: (a, b, % of a-buyers who bought b, reverse %)."""
    divs = [c for c in aff.columns if c != "Buyers"]
    best = None
    for a in divs:
        for b in divs:
            if a != b and aff.loc[a, "Buyers"] > 0 and aff.loc[b, "Buyers"] > 0:
                gap = aff.loc[a, b] - aff.loc[b, a]
                if best is None or gap > best[0]:
                    best = (gap, a, b, aff.loc[a, b], aff.loc[b, a])
    return best[1:] if best and best[0] > 0 else None


def tab_divisions(f, cur, prev, windows, f_cross):
    dv = f.groupby("Division").agg(Sales=("Sales", "sum"), GrossProfit=("Gross Profit", "sum"),
                                   Units=("Units", "sum"), Products=("Product Name", "nunique"),
                                   AvgLineMargin=("Margin %", "mean")).reset_index()
    dv = dv.rename(columns={"AvgLineMargin": "Avg line margin % (unweighted)"})
    dv["Weighted Margin %"] = dv["GrossProfit"] / dv["Sales"] * 100
    dv["Revenue Share %"] = dv["Sales"] / dv["Sales"].sum() * 100
    dv["Profit Share %"] = dv["GrossProfit"] / dv["GrossProfit"].sum() * 100
    dv["Profit - Revenue Share (pp)"] = dv["Profit Share %"] - dv["Revenue Share %"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue vs profit")
        fig = px.bar(dv.melt(id_vars="Division", value_vars=["Sales", "GrossProfit"], var_name="Metric"),
                     x="Division", y="value", color="Metric", barmode="group")
        fig.update_layout(legend_title=None, height=380, yaxis_title="$", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        leader = dv.loc[dv["GrossProfit"].idxmax()]
        show_observation(f"{leader['Division']} generates the highest gross profit (${leader['GrossProfit']:,.0f}) "
                         f"with a weighted margin of {leader['Weighted Margin %']:.1f}%.")
    with c2:
        st.subheader("Margin distribution by division")
        fig = px.box(f, x="Division", y="Margin %", color="Division", points=False, color_discrete_map=DIV_COLORS)
        fig.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        med = f.groupby("Division")["Margin %"].median().sort_values()
        show_observation(f"{med.index[-1]} has the highest median order-line margin ({med.iloc[-1]:.1f}%), "
                         f"while {med.index[0]} has the lowest ({med.iloc[0]:.1f}%).")

    st.subheader("Division efficiency")
    st.dataframe(dv.style.format({
        "Sales": "${:,.0f}", "GrossProfit": "${:,.0f}", "Units": "{:,.0f}",
        "Avg line margin % (unweighted)": "{:.1f}", "Weighted Margin %": "{:.1f}",
        "Revenue Share %": "{:.1f}", "Profit Share %": "{:.1f}", "Profit - Revenue Share (pp)": "{:+.1f}"}),
        width="stretch", hide_index=True)
    for _, r in dv.iterrows():
        tag = "strong financial efficiency" if r["Profit - Revenue Share (pp)"] > 0.5 else (
            "structural margin issue" if r["Profit - Revenue Share (pp)"] < -0.5 else "in line with its revenue share")
        st.markdown(f"- **{r['Division']}**: {r['Weighted Margin %']:.1f}% margin, "
                    f"{r['Revenue Share %']:.1f}% of revenue vs {r['Profit Share %']:.1f}% of profit → {tag}.")

    st.subheader("Trend over time by division")
    metric = st.radio("Metric", ["Sales", "Gross Profit"], horizontal=True)
    tr = f.groupby(["Month", "Division"])[metric].sum().reset_index()
    fig = px.line(tr, x="Month", y=metric, color="Division", markers=True, color_discrete_map=DIV_COLORS)
    fig.update_layout(height=340, yaxis_title="$", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    monthly = f.groupby("Month")[metric].sum()
    top3 = monthly.nlargest(min(3, len(monthly)))
    show_observation(
        f"{metric} peaks in {top3.index[0]:%b %Y} (${top3.iloc[0]:,.0f}); the {len(top3)} strongest months "
        f"({', '.join(top3.index.strftime('%b %Y'))}) account for {top3.sum() / monthly.sum() * 100:.0f}% of the total.")

    if "Region" in f.columns:
        st.subheader("Where growth came from: region")
        _growth_chart(f, cur, prev, windows, "Region", height=340)
        if margin_is_fixed(f):
            st.caption("Regional margin is not shown: every product has one fixed margin, so regional margin "
                       "differences are product mix, not regional pricing or cost.")

        st.subheader("Gross profit by region and division")
        hm = f.pivot_table(index="Region", columns="Division", values="Gross Profit", aggfunc="sum", fill_value=0)
        fig = px.imshow(hm, text_auto=".3s", aspect="auto", color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Cross-division affinity")
    aff = division_affinity(f_cross)
    if len(aff) < 2:
        st.info("Affinity needs at least two divisions with buyers.")
        return
    st.caption("Of the customers who ever bought the division in each row, the % who also bought each column. "
               "Ignores the Division and Product filters (it compares across them); respects the date range.")
    st.dataframe(aff.style.format({c: "{:.1f}%" for c in aff.columns if c != "Buyers"} | {"Buyers": "{:,.0f}"}),
                 width="stretch")
    gap = _affinity_gap(aff)
    if gap:
        a, b, ab, ba = gap
        show_observation(f"{ab:.0f}% of {a} buyers also bought {b}, but only {ba:.1f}% of {b} buyers also bought {a}: "
                         "cross-buying is lopsided.")


def _growth_chart(f, cur, prev, windows, by, height=340, top=None):
    """Prior vs current sales by group with growth % labels; falls back to plain sales when no YoY exists."""
    if not windows or cur.empty or prev.empty:
        t = f.groupby(by)["Sales"].sum().sort_values(ascending=False)
        t = t.head(top) if top else t
        fig = px.bar(t.reset_index(), x=by, y="Sales")
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Growth needs a prior year of data; showing total sales for the selection instead.")
        return None
    g = growth_table(cur, prev, by)
    g = g.head(top) if top else g
    lc, lp = window_label(*windows["cur"]), window_label(*windows["prev"])
    fig = go.Figure()
    fig.add_bar(x=g[by], y=g["Prior"], name=lp, marker_color="#B0B7C3")
    fig.add_bar(x=g[by], y=g["Current"], name=lc, marker_color="#2A9D8F",
                text=["new" if np.isnan(x) else f"{x:+.0f}%" for x in g["Growth %"]], textposition="outside")
    fig.update_layout(barmode="group", height=height, yaxis_title="Sales ($)", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    valid = g.dropna(subset=["Growth %"])
    if len(valid) >= 2:
        hi, lo = valid.loc[valid["Growth %"].idxmax()], valid.loc[valid["Growth %"].idxmin()]
        big = g.loc[g["Change"].idxmax()]
        grew = "Every one grew" if (valid["Growth %"] > 0).all() else f"{int((valid['Growth %'] <= 0).sum())} did not grow"
        show_observation(
            f"{lc} vs {lp}: {grew}. Fastest {hi[by]} ({hi['Growth %']:+.0f}%), slowest {lo[by]} ({lo['Growth %']:+.0f}%). "
            f"{big[by]} added the most dollars (+${big['Change']:,.0f}, {big['Share of Growth %']:.0f}% of the growth).")
    return g


# --------------------------------------------------------------------------- tab 4: where to grow
def tab_grow(prod, min_lines, f_cross):
    if len(prod) < 3:
        st.info("Growth views need at least 3 products in view. Choose 'All Products' to see them.")
        return
    w = where_to_grow(prod, min_lines)
    rel = w[w["Lines"] >= min_lines]

    st.subheader("Profit per unit vs margin %")
    st.caption("Margin % says how much of each sale is kept; profit per unit says how much each unit adds to the "
               "bottom line. The two can rank products very differently.")
    if len(rel) < 2:
        st.info(f"Fewer than two products have {min_lines}+ order lines, so the comparison is not drawn.")
    else:
        port = prod["GrossProfit"].sum() / prod["Units"].sum()
        fig = px.scatter(rel, x="Gross Margin %", y="Profit per Unit", size="GrossProfit", color="Division",
                         text="Short", color_discrete_map=DIV_COLORS, size_max=45, hover_name="Product Name",
                         hover_data={"Short": False, "Lines per Month": ":.1f", "Division": False})
        fig.add_hline(y=port, line_dash="dot", line_color="grey", annotation_text=f"portfolio avg ${port:.2f}")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=460, yaxis_title="Profit per unit ($)", margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        dis = rel.assign(_d=(rel["Rank: Margin %"] - rel["Rank: Profit/Unit"]).abs())
        r = dis.loc[dis["_d"].idxmax()]
        thin = len(w) - len(rel)
        show_observation(
            f"{r['Product Name']} ranks #{int(r['Rank: Margin %'])} of {len(rel)} on margin % but "
            f"#{int(r['Rank: Profit/Unit'])} on profit per unit (${r['Profit per Unit']:.2f} vs a portfolio average of "
            f"${port:.2f}). Bubble size = gross profit."
            + (f" {thin} thin-data product(s) are not drawn." if thin else ""))

    st.subheader("Where to grow: ranking")
    st.caption("High profit/unit = at least 1.5x the portfolio average. Low volume = order lines per month below an equal "
               "share among products with enough data. 'If volume doubles' = the product's current share of total gross "
               "profit, i.e. what a second copy of today's profit would add.")
    cols = ["Where to Grow", "Product Name", "Profit per Unit", "Profit/Unit vs Portfolio (x)", "Gross Margin %",
            "Rank: Margin %", "Rank: Profit/Unit", "Lines per Month", "If Volume Doubles: +% of Total GP"]
    st.dataframe(w[cols].style.format({
        "Profit per Unit": "${:.2f}", "Profit/Unit vs Portfolio (x)": "{:.1f}x", "Gross Margin %": "{:.1f}",
        "Rank: Margin %": "{:.0f}", "Rank: Profit/Unit": "{:.0f}", "Lines per Month": "{:.1f}",
        "If Volume Doubles: +% of Total GP": "+{:.1f}%"}, na_rep="-"), width="stretch", hide_index=True)
    st.download_button("Download where-to-grow ranking (CSV)", w[cols].to_csv(index=False).encode("utf-8"),
                       file_name="nassau_candy_where_to_grow.csv", mime="text/csv")
    grow = w[w["Where to Grow"].str.startswith("Grow")]
    if not grow.empty:
        g = grow.iloc[0]
        show_observation(f"{g['Product Name']} is the clearest growth lever: {g['Profit/Unit vs Portfolio (x)']:.1f}x the "
                         f"portfolio's profit per unit at just {g['Lines per Month']:.1f} order lines a month; doubling it "
                         f"adds about {g['If Volume Doubles: +% of Total GP']:.1f}% to total gross profit.")
    else:
        show_observation("No product combines high profit per unit with low volume in this selection.")

    st.subheader("Cross-sell opportunity")
    cs = cross_sell(f_cross, min_lines)
    if not cs:
        st.info("No smaller-division product has enough order lines to measure cross-selling.")
        return
    a = cs["anchor"]
    m = st.columns(3)
    m[0].metric(f"Buyers of {', '.join(short(p) for p in cs['products'])}", f"{cs['group_buyers']:,}")
    m[1].metric(f"…who also bought {a}", f"{cs['pct_group_also_anchor']:.0f}%")
    m[2].metric(f"{a} buyers who bought any of them", f"{cs['pct_anchor_also_group']:.1f}%")
    st.dataframe(cs["per_product"].style.format({f"% also bought {a}": "{:.0f}%", "Buyers": "{:,.0f}"}),
                 width="stretch", hide_index=True)
    show_observation(
        f"Customers of these products clearly buy {a}, but the reverse is rare: {cs['anchor_buyers']:,} {a} buyers, "
        f"only {cs['pct_anchor_also_group']:.1f}% of whom bought one of them. Offering them to existing {a} customers "
        "is the concrete cross-sell opportunity. (Ignores the Division/Product filters.)")


# --------------------------------------------------------------------------- tab 5: concentration
def tab_concentration(f, prod, cur, prev, windows):
    n = len(prod)
    if n >= 3:
        ps, pp = pareto(prod.set_index("Short")["Sales"]), pareto(prod.set_index("Short")["GrossProfit"])
        c = st.columns(4)
        c[0].metric("Products for 80% of revenue", f"{n_for_80(ps)} of {n} ({n_for_80(ps) / n * 100:.0f}%)")
        c[1].metric("Products for 80% of profit", f"{n_for_80(pp)} of {n} ({n_for_80(pp) / n * 100:.0f}%)")
        c[2].metric("Top product share of profit", f"{pp['value'].iloc[0] / pp['value'].sum() * 100:.1f}%",
                    help=f"Top product: {pp.index[0]}")
        c[3].metric("Profit HHI (0-10,000)", f"{((pp['value'] / pp['value'].sum() * 100) ** 2).sum():,.0f}",
                    help="Herfindahl index of product profit shares. Higher = more concentrated.")
        for name, p, col in [("Revenue", ps, "Sales"), ("Profit", pp, "Gross Profit")]:
            fig = go.Figure()
            fig.add_bar(x=p.index, y=p["value"], name=col)
            fig.add_scatter(x=p.index, y=p["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers",
                            line=dict(color="crimson"))
            fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=80, y1=80,
                          line=dict(dash="dash", color="grey"))
            fig.update_layout(title=f"Pareto: {name}", height=400, yaxis=dict(title="$"),
                              yaxis2=dict(title="Cumulative % (dashed = 80%)", overlaying="y", side="right",
                                          range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100], showgrid=False),
                              margin=dict(l=0, r=0, t=60, b=0), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
            k = n_for_80(p)
            show_observation(f"{name}: {k} of {len(p)} products ({k / len(p) * 100:.1f}%) account for at least 80% of "
                             f"{name.lower()}; the leading product contributes {p['value'].iloc[0] / p['value'].sum() * 100:.1f}%.")
    else:
        st.info("Product concentration needs at least 3 products in view. Choose 'All Products' to see it.")

    st.subheader("Factory concentration")
    fac = factory_table(prod)
    conc = factory_concentration(fac)
    if len(fac) >= 3:
        names = " and ".join(f"**{nm}** ({s:.1f}%)" for nm, s in zip(conc["names"], conc["shares"]))
        msg = (f"{names} together produce **{conc['share']:.1f}%** of gross profit; the other "
               f"{conc['n_factories'] - 2} factories combined produce {100 - conc['share']:.1f}%. "
               "A stoppage or cost shock at either top factory would hit most of the profit.")
        (st.warning if conc["share"] >= 80 else st.info)(md(msg))
    fig = px.bar(fac, x="Factory", y="Profit Share %", text=fac["Profit Share %"].map("{:.1f}%".format),
                 color="Margin %", color_continuous_scale="RdYlGn", range_color=[0, 80])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fac.style.format({"Sales": "${:,.0f}", "GrossProfit": "${:,.0f}", "Margin %": "{:.1f}",
                                   "Sales Share %": "{:.1f}", "Profit Share %": "{:.1f}"}),
                 width="stretch", hide_index=True)

    if "State/Province" not in f.columns:
        return
    st.subheader("Growth by state/province")
    stt = f.groupby("State/Province")["Gross Profit"].sum().sort_values(ascending=False)
    top5 = stt.head(5).sum() / stt.sum() * 100
    st.markdown(f"Top 5 states/provinces generate **{top5:.1f}%** of gross profit "
                f"({stt.index[0]} alone: {stt.iloc[0] / stt.sum() * 100:.1f}%).")
    g = _growth_chart(f, cur, prev, windows, "State/Province", height=360, top=15)
    sa = f.groupby("State/Province").agg(p=("Gross Profit", "sum"), s=("Sales", "sum"), n=("Sales", "size"))
    sm, lines = sa["p"] / sa["s"] * 100, sa["n"]
    st.caption(f"Chart shows the 15 largest states by sales. State margins are not shown: they span {sm.min():.0f}% to "
               f"{sm.max():.0f}% only because of product mix on small samples (the median state has "
               f"{int(lines.median())} order lines).")
    if g is not None:
        full = growth_table(cur, prev, "State/Province")
        with st.expander("All states: sales growth table"):
            st.dataframe(full.style.format({"Current": "${:,.0f}", "Prior": "${:,.0f}", "Change": "${:+,.0f}",
                                            "Growth %": "{:+.0f}%", "Share of Growth %": "{:.1f}%"}, na_rep="new"),
                         width="stretch", hide_index=True)


# --------------------------------------------------------------------------- tab 7: data & notes
def _fmt_log(v) -> str:
    return f"${v:,.2f}" if isinstance(v, float) else f"{v:,}"


def tab_notes(df, log, dedupe, min_lines):
    st.subheader("Cleaning log")
    st.table(pd.DataFrame({"Step": list(log.keys()), "Value": [_fmt_log(v) for v in log.values()]}))

    lines = df.groupby("Product Name").size()
    thin = lines[lines < min_lines].index
    thin_share = df.loc[df["Product Name"].isin(thin), "Sales"].sum() / df["Sales"].sum() * 100
    remap = df[df["Division"] != df["Division (source)"]].groupby("Product Name").agg(
        csv=("Division (source)", "first"), brief=("Division", "first"))
    dups = log["Exact duplicate rows found (ignoring Row ID)"]

    notes = []
    if remap.empty:
        notes.append("- **Division labels** already agree with the product map in `product_map.json`.")
    else:
        items = "; ".join(f"*{p}*: CSV `{r.csv}` → map `{r.brief}`" for p, r in remap.iterrows())
        notes.append(f"- **Division labels** were standardised to the product map ({items}). "
                     "The original value is kept in `Division (source)`.")
    if "Ship Date" in df.columns:
        ship = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
        lag = (ship - df["Order Date"]).dt.days / 365.25
        if lag.notna().any() and lag.max() > 0.5:
            notes.append(f"- **Ship Date is not used.** It falls a median {lag.median():.1f} years after Order Date "
                         f"(range {lag.min():.1f}-{lag.max():.1f}; ship dates {ship.min():%Y}-{ship.max():%Y}), so it is unreliable.")
    oy = pd.to_numeric(df["Order ID"].str.extract(r"-(\d{4})-")[0], errors="coerce")
    id_match = (oy == df["Order Date"].dt.year).mean() * 100
    if id_match < 50:
        notes.append(f"- **Order ID** years match the Order Date year on only {id_match:.0f}% of rows, so Order Date is the time axis.")
    notes.append(f"- **Duplicates:** {dups:,} rows (${log['Sales value in those duplicate rows ($)']:,.0f} of sales) are identical "
                 f"to another row apart from Row ID; they are {'removed' if dedupe else 'kept'} (sidebar toggle).")
    notes.append(f"- **Thin products:** {len(thin)} product(s) have fewer than {min_lines} order lines "
                 f"({thin_share:.2f}% of sales); their margins are shown but not used for reprice/discontinue calls.")
    if margin_is_fixed(df):
        notes.append("- **Margins are fixed per product**: every product sells at one unit price and cost, so volatility is "
                     "zero and margin differences by month, region, state or ship mode come only from product mix.")
    notes.append("- **Gross margin is not net profit:** the data has no freight, overhead, returns or discount fields.")
    notes.append(f"- **Segments:** *high sales* = at/above an equal revenue share among products with ≥{min_lines} lines; "
                 "*high margin* = at/above the sidebar threshold.")
    notes.append("- **Customer definitions:** every Order ID embeds a product code, so an Order ID is one product line. "
                 "A **visit** is Customer ID + Order Date; **frequency** is distinct purchase dates; **average order "
                 "value** is sales per visit. See the Customers tab for the data-validity warning.")
    notes.append("- **Year-over-year:** the last 12 months of the selected range vs the same dates one year earlier, with the "
                 "same Division/Product filters. Shown only when the prior year exists in the data.")
    st.subheader("Notes and assumptions")
    st.markdown(md("\n".join(notes)))

    if "Ship Mode" in df.columns:
        st.subheader("Ship mode: mix, but no profitability")
        sm = ship_mode_table(df)
        st.dataframe(sm.style.format({"Lines": "{:,.0f}", "Sales": "${:,.0f}", "GrossProfit": "${:,.0f}",
                                      "Share of Sales %": "{:.1f}", "Gross Margin %": "{:.1f}"}),
                     width="stretch", hide_index=True)
        st.caption(f"{sm.iloc[0]['Ship Mode']} is {sm.iloc[0]['Share of Sales %']:.0f}% of sales. Gross margin is nearly the "
                   "same in every mode because it is product mix; without shipping cost you cannot tell which mode is "
                   "actually profitable.")

    st.subheader("Open questions for the business")
    st.markdown(md(
        "1. **Are Customer IDs persistent across years?** Near-zero cross-year retention is either real churn or IDs that "
        "are re-issued each year. The data cannot say which.\n"
        f"2. **Are the {dups:,} duplicate rows real?** They carry ${log['Sales value in those duplicate rows ($)']:,.0f} of "
        "sales. They are removed by default; confirm whether they are true repeat lines or double entries.\n"
        "3. **Can you supply freight, discount and returns data?** Ship Mode is in the file but shipping cost is not, so "
        "ship-mode profitability cannot be measured and gross margin overstates true profit."))
