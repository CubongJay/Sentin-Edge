# SentinEdge

An autonomous trading agent that reads news, checks price, and decides whether to buy, sell, or hold a stock — with a uniform tool contract and paper trading by default.

---

## What it is

SentinEdge is an LLM-driven agent built with LangChain v1 (`create_agent`) on the LangGraph runtime. Given a watchlist of tickers, it:

1. Fetches the latest news for each symbol (`get_news`)
2. Checks the current ask price (`get_price`)
3. Decides whether to buy, sell, or hold based on the news
4. If trading, calls `execute_trade` (paper only), which enforces market hours, symbol validity, and fractionable-vs-whole-share logic
5. Reports the decision and reasoning

The model is `gpt-4.1-mini` via Azure AI Foundry inference. Market data and paper trading go through Alpaca; news search goes through Tavily.

---

## What it currently has

### Tools

All three tools return the same envelope: `{"ok": bool, "error": str | None, "data": <tool-specific>}`.

- **`get_news(stock: str)`** — returns the 5 most recent news articles for a ticker, formatted as text.
- **`get_price(symbol: str)`** — returns the latest ask price for a ticker.
- **`execute_trade(symbol: str, side: str, notional: float = 1.0)`** — places a paper market order. Refuses when the market is closed, the side is invalid, the symbol is not tradable, or the notional is too small for a non-fractionable stock.

### Agent

- Built with `create_agent` (LangChain v1), which runs on LangGraph.
- The agent decides which tool to call and when to stop. The execution transcript is printed to stdout for each run.
- **Current configuration:** `tools = [get_news, get_price]`, so runs are **decision-only** — the agent decides buy/sell/hold and reports its reasoning, but does not call `execute_trade`. The execution path (including the `MARKET CLOSED` refusal) was tested earlier with `execute_trade` enabled; it did not retry and did not narrate past the refusal.

### Safety

- **Paper trading only.** `TradingClient` is initialized with `paper=True`.
- The agent cannot trade when the market is closed; `execute_trade` returns an envelope refusal with the next open time.
- The agent does not retry a refused tool call; it reads `ok: false`, stops, and reports.

### Model and integrations

- LangChain v1 (`create_agent`) on LangGraph
- Azure AI Foundry inference (`gpt-4.1-mini`)
- Alpaca (paper trading + market data)
- Tavily (news)

## Setup

### 1. Prerequisites

- Python 3.11+
- Accounts and API keys for:
  - [Alpaca](https://alpaca.markets) (paper trading — use paper keys, not live keys)
  - [Tavily](https://tavily.com) (news search)
  - Azure AI Foundry (for the `gpt-4.1-mini` inference endpoint)

### 2. Clone and install

```bash
git clone <your-repo-url>
cd sentinedge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET=your_paper_secret
TAVILY_API_KEY=your_tavily_key
AZURE_INFERENCE_URL=https://<your-resource>.services.ai.azure.com/models
AZURE_INFERENCE_CREDENTIAL=your_azure_key
```

Notes:

- **Alpaca:** use your **paper trading** keys. The code also sets `paper=True`, but the keys should be paper keys as a second safeguard.
- **Azure AI Foundry:** `AZURE_INFERENCE_URL` is the **inference endpoint** (`...services.ai.azure.com/models`), not the classic Azure OpenAI endpoint (`...openai.azure.com`). The code uses `AzureAIChatCompletionsModel`, which speaks the inference endpoint's protocol. Using the wrong endpoint produces a 401.
- The model name (`gpt-4.1-mini`) must match a deployment that exists on your Azure resource.

### 4. Run

```bash
python agent.py
```

The script runs the agent on a single symbol (currently `TSLA`) and prints the execution path (tool calls and results) followed by the final report.

To change the watchlist, edit the call at the bottom of `agent.py`:

```python
final_report = run_sentinedge_agent(["AAPL", "MSFT"])
```

### 5. Verify it's working

A successful run prints something like:

```
AGENT EXECUTION PATH
========================================
--- Step 1 (HumanMessage) ---
  Content: Analyze this watchlist and execute trades if needed: TSLA...
--- Step 2 (AIMessage) ---
  🔧 Tool Call: get_news with args {'stock': 'TSLA'}
--- Step 3 (ToolMessage) ---
  Content: {'ok': True, 'error': None, 'data': 'Date: ...'}...
...
[FINAL REPORT]:
 <the agent's decision and reasoning>
```

If you see tool calls with `ok: True` / `ok: False` in the results, the agent is running. If the market is closed and `execute_trade` is enabled, you'll see `ok: False` with `error: "MARKET CLOSED..."` — that's the refusal path working, not a bug.

### Troubleshooting

- **401 from Azure:** check that `AZURE_INFERENCE_URL` is the inference endpoint (`services.ai.azure.com/models`), not the OpenAI endpoint, and that the key is from the same resource.
- **401 from Alpaca:** check that `ALPACA_API_KEY` / `ALPACA_SECRET` are paper keys, and that `.env` is in the project root.
- **No tool calls in the transcript:** the model may have decided to hold. Run with a symbol that has active news, or check the final report for the reasoning.
```

