# Nassau Candy: Product Line Profitability dashboard

```
streamlit run app.py     # put Nassau_Candy_Distributor.csv next to app.py (or upload it in the sidebar)
pytest -q                # unit tests + full-UI tests (Streamlit AppTest)
```

| File | Role |
|---|---|
| `app.py` | Entry point: sidebar filters, YoY windows, wiring |
| `config.py` | Constants and thresholds; loads the product map |
| `product_map.json` | Product → division and factory (edit here, not in code) |
| `logic.py` | Pure pandas: cleaning, product table, YoY/growth, where-to-grow, cross-sell, concentration |
| `customers.py` | Pure pandas: visits, data-validity checks, new vs returning, cohort retention, RFM |
| `findings.py` | Generates the "Key findings" sentences from the data in view |
| `views.py`, `views_customers.py` | Streamlit drawing code, one function per block or tab |
| `test_logic.py`, `test_ui.py` | Unit tests; end-to-end tests that drive `app.py` |

## Definitions that matter
- Every Order ID embeds a product code, so an Order ID is one product line, not a basket.
- **Visit** = Customer ID + Order Date. **Frequency** = distinct purchase dates. **Average order value** = sales per visit.
- **YoY** = last 12 months of the selected range vs the same dates a year earlier (same Division/Product filters); hidden when the prior year is not in the data.
- RFM segments are shown only when the data-validity check passes (≥100 customers and ≥10% of customers with 2+ purchase dates).
- `NASSAU_DATA_FILE` overrides the CSV path (the UI tests use it).
