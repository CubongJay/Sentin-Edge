"""
The authority envelope: deterministic policy checks that bound what the
agent is permitted to trade. Called before any order; the LLM cannot bypass it.
"""

MAX_NOTIONAL = 100

ALLOWED_SYMBOLS = {"AAPL", "MSFT", "TSLA", "GOOGL"}


def check(symbol: str, side: str, notional: float):
    """Return {"ok": bool, "error": str | None, "data": None} for a proposed trade."""

    if notional > MAX_NOTIONAL:
        return {
            "ok": False,
            "error": f"The notional: {notional} is greater than the allowed maximum.",
            "data": None,
        }
    if symbol not in ALLOWED_SYMBOLS:
        return {
            "ok": False,
            "error": f"The stock: {symbol} is not allowed.",
            "data": None,
        }

    ok = True
    return {"ok": True, "error": None, "data": None}
