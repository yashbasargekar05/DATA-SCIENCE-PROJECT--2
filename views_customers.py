"""Customers tab: new vs returning, cohort retention, products per visit, plus a data-validity warning."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from customers import (build_rfm, cohort_retention, customer_validity, new_vs_returning, products_per_visit,
                       validity_messages)
from views import md, show_observation


def render_validity_warning(v: dict) -> None:
    if v["reliable"]:
        st.success(f"Data-validity check passed: {v['multi_date']:,} customers ({v['multi_date_share']:.1f}%) bought on "
                   "more than one date, enough for repeat-purchase analysis. Note a visit is still Customer + Order Date, "
                   "not Order ID.")
        return
    msgs = validity_messages(v)
    st.warning(md("### ⚠️ Data validity: read this before using any customer metric\n"
                  + "\n".join(f"- {m}" for m in msgs)
                  + "\n\n**Ask the data owner:** are Customer IDs persistent across years, and does an order ever span "
                    "several dates? Until then, treat repeat-purchase and retention numbers as unproven."))


def tab_customers(df, start, end):
    """df = full cleaned history (so 'first visit' is never distorted by filters); start/end limit what is shown."""
    st.subheader("Customers: new vs returning, retention, basket size")
    st.caption("Uses each customer's full history. A **visit** = Customer ID + Order Date (Order IDs are single product "
               "lines here). Sidebar Division/Product filters do not apply on this tab; the date range limits what is shown.")
    v = customer_validity(df)
    render_validity_warning(v)

    # ---- new vs returning
    st.markdown("#### New vs returning customers")
    nv = new_vs_returning(df)
    nv = nv[(nv.index >= pd.Timestamp(start).to_period("M").to_timestamp()) & (nv.index <= pd.Timestamp(end))]
    if nv.empty:
        st.info("No customer activity in the selected date range.")
        return
    tot_new, tot_ret = int(nv["New customers"].sum()), int(nv["Returning customers"].sum())
    ret_sales = nv["Returning sales"].sum() / (nv["New sales"].sum() + nv["Returning sales"].sum()) * 100
    m = st.columns(3)
    m[0].metric("New customers", f"{tot_new:,}", help="Customers whose first-ever visit falls in the selected range.")
    m[1].metric("Returning customer-periods", f"{tot_ret:,}",
                help="Customer-months with a visit on a later date than their first. A customer can count in several months.")
    m[2].metric("Sales from returning visits", f"{ret_sales:.1f}%")
    fig = go.Figure()
    fig.add_bar(x=nv.index, y=nv["New customers"], name="New", marker_color="#2A9D8F")
    fig.add_bar(x=nv.index, y=nv["Returning customers"], name="Returning", marker_color="#E85D9E")
    fig.update_layout(barmode="stack", height=340, yaxis_title="Customers", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    show_observation(f"{tot_new:,} customers made a first purchase in this range; returning visits account for "
                     f"{ret_sales:.1f}% of sales. Growth here is almost entirely new customers. History starts with the "
                     "data, so everyone in the first month is New by construction.")

    # ---- cohort retention
    st.markdown("#### Cohort retention (quarterly cohorts)")
    size, counts, pct = cohort_retention(df, "Q")
    view = pct.drop(columns=[0]).copy()
    view.columns = [f"Q+{c}" for c in view.columns]
    if view.empty:
        st.info("Not enough history for cohort retention.")
    else:
        heat = px.imshow(view, text_auto=".1f", aspect="auto", color_continuous_scale="Blues", zmin=0,
                         zmax=max(5, float(view.max().max())) if view.notna().any().any() else 5,
                         labels=dict(x="Quarters since first purchase", y="Cohort", color="% still buying"))
        heat.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(heat, width="stretch")
        later = counts.drop(columns=[0])
        got, base = int(later.sum().sum()), int(size.sum())
        best = view.stack().max() if view.notna().any().any() else float("nan")
        show_observation(f"Of {base:,} customers in all cohorts, {got:,} customer-quarters show a later purchase. The best "
                         f"single cell is {best:.1f}% retained; blank cells are quarters not yet observable. "
                         "Retention near zero is either real churn or non-persistent Customer IDs.")
        with st.expander("Cohort sizes and counts"):
            st.dataframe(pd.concat([size, counts.drop(columns=[0]).add_prefix("Q+")], axis=1), width="stretch")

    # ---- products per visit
    st.markdown("#### Products per visit")
    sub = df[(df["Order Date"] >= pd.Timestamp(start)) & (df["Order Date"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    dist, yearly = products_per_visit(sub)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(dist, x="Products per visit", y="Visits", text=dist["Share %"].map("{:.0f}%".format))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(yearly.style.format({"Visits": "{:,.0f}", "Products per visit": "{:.2f}", "Sales per visit": "${:.2f}",
                                          "Units per visit": "{:.1f}"}), width="stretch", hide_index=True)
    multi = dist.loc[dist["Products per visit"] != "1", "Share %"].sum()
    show_observation(f"{multi:.0f}% of visits include two or more different products; the average visit is "
                     f"{sub.groupby(['Customer ID', 'Order Date'])['Product Name'].nunique().mean():.2f} products. "
                     "This is what the old 'frequency' was really counting.")

    # ---- RFM (only when repeat buying is real)
    st.markdown("#### RFM segments")
    if not v["reliable"]:
        st.info("RFM segmentation is hidden because the data-validity check failed: with so few repeat visits, segments "
                "like 'At Risk' or 'Cannot Lose Them' would describe nothing real (recency would mostly reflect which "
                "year the customer bought in). It returns automatically if repeat-visit data becomes available.")
        return
    rfm = build_rfm(sub, pd.Timestamp(end) + pd.Timedelta(days=1))
    seg = rfm.groupby("RFM Segment").agg(Customers=("Customer ID", "nunique"), Sales=("Monetary", "sum"),
                                         Avg_Visits=("Frequency", "mean")).sort_values("Customers", ascending=False)
    st.caption("Frequency = distinct purchase dates. Recency and Monetary use percentile ranks with ties sharing a score.")
    st.dataframe(seg.style.format({"Sales": "${:,.0f}", "Avg_Visits": "{:.2f}"}), width="stretch")
