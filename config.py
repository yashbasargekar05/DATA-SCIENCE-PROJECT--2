"""Settings and reference data for the Nassau Candy dashboard (no Streamlit imports)."""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_DATA_FILE = BASE_DIR / "Nassau_Candy_Distributor.csv"
PRODUCT_MAP_FILE = BASE_DIR / "product_map.json"


def data_file() -> Path:
    """CSV location; override with the NASSAU_DATA_FILE environment variable (used by the UI tests)."""
    return Path(os.environ.get("NASSAU_DATA_FILE", DEFAULT_DATA_FILE))


def load_product_map(path: Path = PRODUCT_MAP_FILE) -> dict[str, tuple[str, str]]:
    """Read product -> (division, factory) from JSON and fail loudly on a malformed entry."""
    products = json.loads(Path(path).read_text(encoding="utf-8")).get("products", {})
    out = {}
    for name, entry in products.items():
        if not isinstance(entry, dict) or not entry.get("division") or not entry.get("factory"):
            raise ValueError(f"{path.name}: '{name}' needs both a 'division' and a 'factory'.")
        out[name] = (entry["division"], entry["factory"])
    if not out:
        raise ValueError(f"{path.name}: no products defined.")
    return out


PRODUCT_MAP = load_product_map()

DIV_COLORS = {"Chocolate": "#6F4E37", "Sugar": "#E85D9E", "Other": "#2A9D8F"}
INSUFFICIENT = "Insufficient data"

# Columns the cleaning step cannot work without
CORE_COLUMNS = ["Row ID", "Order ID", "Order Date", "Customer ID", "Division", "Product Name",
                "Sales", "Units", "Gross Profit", "Cost"]

MIN_VOL_MONTHS = 6        # margin volatility needs at least this many active months
MIN_LINES_DEFAULT = 30    # order lines a product needs before its margin is judged reliable
HIGH_F, REPEAT_F = 3, 2   # purchase dates that count as "high frequency" / "repeat buyer" in RFM
YIELD_MULT = 1.5          # "high profit per unit" = at least this multiple of the portfolio average
FLAT_AOV_PCT = 5.0        # |change| below this counts as "flat" in generated findings
FLAT_MARGIN_PP = 1.0
MIN_REPEAT_CUSTOMERS = 100   # below this many customers with 2+ purchase dates, repeat-buying
MIN_REPEAT_SHARE = 10.0      # analysis (RFM, retention) is flagged as unreliable (% of customers)
