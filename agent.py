import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient
from tavily.errors import ForbiddenError

from envelope import check

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=True)

data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)


blob_service = BlobServiceClient.from_connection_string(
    os.getenv("BLOB_CONNECTION_STRING")
)
runs_container = blob_service.get_container_client("runs")


@tool
def get_news(stock: str) -> dict:
    """
    Search for the 5 most recent news articles about a specific stock ticker.

    Use this tool whenever you need up-to-date information, sentiment, or news
    events for a company before making a trading decision.

    Args:
        stock: The ticker symbol to search news for (e.g., "AAPL", "MSFT").

    Returns:
        A dict with this shape on success:
        {
            "ok": True,
            "error": None,
            "data": "<5 most recent articles, formatted as text, separated by '---'>"
        }
        On failure:
        {
            "ok": False,
            "error": "<human-readable reason>",
            "data": None
        }
    """

    try:

        news = tavily_client.search(
            query=f"latest news about {stock}",
            topic="news",
            search_depth="basic",
            days=1,
            max_results=5,
        )

        articles_text = []
        for article in news["results"]:

            date = article.get("published_date", "Date Not Found")
            title = article["title"]
            summary = article["content"]
            formatted_article = f"Date: {date}\nTitle: {title}\nSummary: {summary}\n---"
            articles_text.append(formatted_article)

        combined_articles = "\n".join(articles_text)
        return {"ok": True, "error": None, "data": combined_articles}
    except ForbiddenError as e:
        detail = str(e).strip()
        msg = (
            f"Tavily 403 Forbidden: {detail}"
            if detail
            else "Tavily 403 Forbidden (check plan/credits)"
        )
        return {"ok": False, "error": msg, "data": None}

    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = getattr(getattr(e, "response", None), "text", None)
        msg = str(e).strip() or f"{type(e).__name__} (status={status}): {body or e!r}"
        return {"ok": False, "error": msg, "data": None}


@tool
def get_price(symbol: str):
    """
    Get the latest ask price for a stock symbol.

    Use this before placing a trade to confirm the current market price.

    Args:
        symbol: The ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        A dict with this exact shape:
        {
            "ok": True,
            "error": None,
            "data": {"symbol": str, "price": float}
        }
        On failure:
        {
            "ok": False,
            "error": "<human-readable reason>",
            "data": None
        }
    """
    try:
        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )
        price = quote[symbol].ask_price

        return {
            "ok": True,
            "error": None,
            "data": {
                "price": price,
                "symbol": symbol,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}


@tool
def execute_trade(symbol: str, side: str, notional: float = 1.0):
    """
    Place a market order for a stock.

    Use this tool only when you have decided to trade, after reviewing news
    and the current price. side must be 'buy' or 'sell'.

    Uses notional (dollar amount) for fractional shares, whole shares otherwise.
    Will refuse if the market is closed, the side/symbol is invalid, or the
    notional is too small for a non-fractionable stock.

    Args:
        symbol: The ticker symbol (e.g., "AAPL", "MSFT").
        side:   "buy" or "sell".
        notional: Dollar amount to trade (default 1.0).

    Returns:
        A dict with this shape on success:
        {
            "ok": True,
            "error": None,
            "data": {
                "message": str,
                "order_id": str,
                "status": str
            }
        }
        On refusal:
        {
            "ok": False,
            "error": "<human-readable reason>",
            "data": None
            # or {"next_open": str} when the market is closed
        }
    """

    decision = check(symbol, side, notional)
    if not decision["ok"]:
        return decision
    try:

        clock = trading_client.get_clock()
        if not clock.is_open:

            return {
                "ok": False,
                "error": f"MARKET CLOSED: cannot place order for {symbol} right now",
                "data": {"next_open": str(clock.next_open)},
            }

        side_l = side.lower()
        if side_l not in ("buy", "sell"):

            return {
                "ok": False,
                "error": f"REJECTED: invalid side '{side}'. Must be 'buy' or 'sell'.",
                "data": None,
            }

        asset = trading_client.get_asset(symbol)
        if not asset.tradable:
            return {
                "ok": False,
                "error": f"REJECTED: {symbol} is not tradable on Alpaca.",
                "data": None,
            }

        order_side = OrderSide.BUY if side_l == "buy" else OrderSide.SELL
        if asset.fractionable:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            quote = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            price = quote[symbol].ask_price
            if notional < price:

                return {
                    "ok": False,
                    "error": f"REJECTED: {symbol} is not fractionable and notional ${notional:.2f} < 1 share (${price:.2f})",
                    "data": None,
                }

            qty = max(1, int(notional / price))
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        order = trading_client.submit_order(req)

        return {
            "ok": True,
            "error": None,
            "data": {
                "message": f"ORDER PLACED: {side_l} {symbol}.",
                "order_id": order.id,
                "status": order.status,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}


system_instruction = (
    "You are an elite hedge fund algorithmic trading agent named SentinEdge.\n"
    "Your goal is to evaluate a specific watchlist of stocks and make data-driven trading decisions.\n\n"
    "OPERATIONAL MANDATES:\n"
    "1. For each stock, you MUST first fetch the latest news using your search tool.\n"
    "2. Thoroughly analyze the sentiment, focusing heavily on revenue, lawsuits, and product launches.\n"
    "3. Based on your analysis, decide whether to buy, sell, or hold.\n"
    "4. If a trade is warranted, call the trading tool. If a tool returns ok: false, read the 'error' field and do not retry that tool. Report the failure in your final answer."
    "5. Be concise and execution-oriented. Do not hallucinate data."
)


llm = AzureAIChatCompletionsModel(
    endpoint=os.getenv("AZURE_INFERENCE_URL"),
    credential=os.getenv("AZURE_INFERENCE_CREDENTIAL"),
    model="gpt-4.1-mini",
)

tools = [execute_trade, get_news, get_price]


agent_executor = create_agent(model=llm, tools=tools, system_prompt=system_instruction)


def run_sentinedge_agent(watchlist_stocks: list):
    """
    Kicks off the autonomous agent workflow for a list of target tickers.
    """
    stocks_input = ", ".join(watchlist_stocks)
    timestamp_start = datetime.now(timezone.utc).isoformat()
    response = agent_executor.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Analyze this watchlist and execute trades if needed: {stocks_input}"
                )
            ]
        },
        config={"recursion_limit": 50},
    )
    timestamp_end = datetime.now(timezone.utc).isoformat()
    print("\n" + "=" * 40)
    print("AGENT EXECUTION PATH")
    print("=" * 40)
    for i, msg in enumerate(response["messages"]):
        print(f"\n--- Step {i+1} ({type(msg).__name__}) ---")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Tool Call: {tc['name']} with args {tc['args']}")
        if hasattr(msg, "content"):

            print(f"  Content: {msg.content[:500]}...")  # Truncate long content

    results_by_id = {
        msg.tool_call_id: msg.content
        for msg in response["messages"]
        if isinstance(msg, ToolMessage)
    }

    tool_calls = [
        {
            "name": tc["name"],
            "args": tc["args"],
            "result": (
                json.loads(results_by_id[tc["id"]])
                if tc["id"] in results_by_id
                else None
            ),
        }
        for msg in response["messages"]
        if isinstance(msg, AIMessage) and msg.tool_calls
        for tc in msg.tool_calls
    ]

    trade_results = [
        tc["result"]
        for tc in tool_calls
        if tc["name"] == "execute_trade" and tc["result"]
    ]
    if not trade_results:
        outcome = "held"
    elif any(r["ok"] for r in trade_results):
        outcome = "traded"
    else:
        outcome = "refused"

    final_report = response["messages"][-1].content

    envelope_decisions = [
        {
            "symbol": tc["args"]["symbol"],
            "side": tc["args"]["side"],
            "notional": tc["args"].get("notional", 1.0),
            "decision": check(
                tc["args"]["symbol"],
                tc["args"]["side"],
                tc["args"].get("notional", 1.0),
            ),
        }
        for tc in tool_calls
        if tc["name"] == "execute_trade"
    ]

    audit_trail = {
        "run_id": str(uuid.uuid4()),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "watch_list": watchlist_stocks,
        "tool_calls": tool_calls,
        "reasoning": [
            msg.content
            for msg in response["messages"]
            if isinstance(msg, AIMessage) and msg.content
        ],
        "final_report": final_report,
        "outcome": outcome,
        "envelope_decisions": envelope_decisions,
    }

    ts_for_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    blob_name = (
        f"{ts_for_name}_{'_'.join(watchlist_stocks)}_{audit_trail['run_id']}.json"
    )
    runs_container.upload_blob(
        name=blob_name,
        data=json.dumps(audit_trail, indent=2).encode("utf-8"),
        overwrite=False,
    )
    print(f"\n[AUDIT] Run persisted to runs/{blob_name}")
    return response["messages"][-1].content


if __name__ == "__main__":

    print("Launching SentinEdge Agent...")
    final_report = run_sentinedge_agent(["NVDA"])
    print("\n[FINAL REPORT]:\n", final_report)
