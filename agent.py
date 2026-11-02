import os
from pathlib import Path

from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from langchain.agents import create_agent

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading import TradingClient

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=True)

data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)


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
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}


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
    # 1. Market hours gate — the important bit
    try:
        clock = trading_client.get_clock()
        if not clock.is_open:

            return {
                "ok": False,
                "error": f"MARKET CLOSED: cannot place order for {symbol} right now",
                "data": {"next_open": str(clock.next_open)},
            }

        # 2. Validate side before touching Alpaca
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

    response = agent_executor.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Analyze this watchlist and execute trades if needed: {stocks_input}"
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    print("\n" + "=" * 40)
    print("AGENT EXECUTION PATH")
    print("=" * 40)
    for i, msg in enumerate(response["messages"]):
        print(f"\n--- Step {i+1} ({type(msg).__name__}) ---")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Tool Call: {tc['name']} with args {tc['args']}")
        if hasattr(msg, "content"):
            # For ToolMessage, this is the result. For AIMessage, this is text/reasoning.
            print(f"  Content: {msg.content[:500]}...")  # Truncate long content

    return response["messages"][-1].content


if __name__ == "__main__":
    # Test the entire autonomous flow for a single stock
    print("Launching SentinEdge Agent...")
    final_report = run_sentinedge_agent(["TSLA"])
    print("\n[FINAL REPORT]:\n", final_report)
