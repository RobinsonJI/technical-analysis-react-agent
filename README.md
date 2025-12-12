# Technical Analysis ReAct Agent

A stock technical analysis agent powered by LLMs and LangGraph, providing comprehensive trading signals and recommendations using multiple technical indicators.

## Overview

This project implements a **ReAct (Reasoning + Acting) agent** that performs detailed technical analysis on stocks using 13+ technical indicators across 5 categories:

- **Volatility**: ATR, Bollinger Bands (with squeeze detection)
- **Trend**: SMA, EMA, MACD, ROC
- **Momentum**: RSI, Stochastic Oscillator, MACD
- **Volume**: Volume Analysis, OBV (On-Balance Volume)
- **Support & Resistance**: Pivot Points, Fibonacci Retracement, Fibonacci Extension
- **Basic Internet Search**: Searches the internet to find more information if required

The agent automatically selects and executes relevant tools, synthesizes findings, and generates comprehensive trading summaries with actionable signals and risk management recommendations.

## Features

### Core Capabilities

- ✅ **Multi-Indicator Analysis** - Uses 13+ technical indicators across 5 categories
- ✅ **Intelligent Tool Selection** - LLM decides which tools to use based on query
- ✅ **Squeeze Detection** - Identifies pre-breakout volatility squeezes
- ✅ **Volume Divergence Detection** - Warns of price-volume misalignment
- ✅ **Risk Assessment** - Calculates position sizing and stop-loss levels based on ATR
- ✅ **Comprehensive Summaries** - 11-section technical analysis reports
- ✅ **Signal Generation** - Identifies high-probability trade setups with confidence levels
- ✅ **Entry/Exit Recommendations** - Specific prices for entries, stops, and profit targets

### Advanced Features

- **Text-Based Tool Calling** - Avoids LLM provider API limitations
- **Fibonacci Analysis** - Retracement and extension levels for trade zones
- **Pivot Point Trading** - Daily support/resistance calculations
- **OBV Divergence** - Accumulation/distribution phase detection
- **Bollinger Band Squeeze + Short Setups** - Pre-breakout identification
- **Debug Logging** - Terminal-based logging with DEBUG mode support

## Installation

### Prerequisites

- Python 3.9+
- UV package manager (recommended) or pip
- API key for LLM service ([OpenRouter](https://openrouter.ai/), OpenAI, etc.)

### Setup

1. **Clone the repository**
```bash copy
git clone https://github.com/RobinsonJI/trading.git
cd trading
```

2. **Install uv (recommended)**

Install uv [here](https://docs.astral.sh/uv/getting-started/installation/).

3. **Install dependencies**
```bash copy
uv sync
```

4. **Configure environment variables**
```bash copy
cp .env.example .env  # If available
# Then edit .env with your settings:
```

**.env file:**
```env
TRADING_API_KEY="sk-or-v1-xxxxx"           # Your LLM API key
TRADING_AGENT_MODEL_URL="https://api.openrouter.ai/api/v1"
MODEL_NAME="openai/gpt-5.1-codex-mini"     # LLM model to use
TEMPERATURE=0.7                            # Model creativity (0-2)
RANDOM_SEED=123                            # For reproducibility
TOP_P=1.0                                  # Sampling diversity (0-1)
ENABLE_LOGGING=true                        # Enable detailed logging
```

## Usage

### Interactive Mode

```bash
# Basic usage with .env defaults
uv run agent

# Or with custom settings
uv run agent --temperature 0.5 --enable-logging
```

Then enter queries like:
```
Enter stock symbol and analysis details: Analyse AAPL stock for upside potential
Enter stock symbol and analysis details: TSLA 5-day analysis with support/resistance
Enter stock symbol and analysis details: Check AMD for squeeze + breakout setup
```


## Agent Workflow

```
User Query
    ↓
[trading_agent node]
    ├─ Initialise LLM with system message
    ├─ Parse tool calls from response
    └─ Emit tool calls if needed
        ↓
   [should_continue router]
    ├─ If tool_calls exist → "tools"
    └─ If no tool_calls → "generate_summary"
        ↓
    [tools node]
    ├─ Execute all requested tools
    ├─ Collect results
    └─ Return to trading_agent for next iteration
        ↓
    [Loop until no more tool_calls]
        ↓
    [generate_summary node]
    ├─ Synthesize all findings
    └─ Output 11-section technical analysis
        ↓
   Final Summary
```

## Configuration

### Model Selection
The agent works with any OpenAI-compatible API (OpenRouter, OpenAI, Azure, etc.):

```bash
# Use different models
uv run agent --model "gpt-4o-mini"
uv run agent --model "claude-3-sonnet"
uv run agent --base-url "https://api.openai.com/v1"
```

### Temperature Tuning
- **0.0** - Deterministic, repeatable analysis
- **0.5-0.7** - Balanced (recommended for trading)
- **1.0+** - Creative, more varied suggestions

## Limitations

- **Not Financial Advice** - This tool provides technical analysis only, not investment advice
- **Historical Data Only** - Analyses past data; cannot predict future markets
- **Internet Dependent** - Requires API access to fetch stock data from yfinance and call LLM

## Project Structure

```
trading/
.env                                    # Configuration (API keys, model settings)
main.py                                 # Entry point
pyproject.toml                          # Dependencies
README.md                               # This file

src/
├── agents/
│   └── trading_agent.py               # Main agent with tool execution
│
├── analysis/indicators/
│   ├── fibonacci.py                   # Fibonacci retracement & extension
│   ├── momentum.py                    # RSI, Stochastic, MACD
│   ├── support_and_resistance.py      # Pivot Points
│   ├── trend.py                       # SMA, EMA, MACD, ROC
│   ├── volume.py                      # Volume, OBV analysis
│   └── volatility.py                  # ATR, Bollinger Bands
│
├── config/
│   ├── logging_config.py              # Terminal logging setup
│   └── __init__.py
│
├── data/
│   ├── basemodels.py                  # Pydantic models for tools
│   ├── enums.py                       # TimePeriod, Interval enums
│   └── fetcher.py                     # Yahoo Finance data fetching
│
├── graph/
│   └── react_graph.py                 # LangGraph workflow
│
├── models/
│   └── client.py                      # LLM client wrapper
│
├── prompts/
│   └── system.py                      # Agent system message
│
└── state/
    └── states.py                      # Agent state definition
```

## License

This project is licensed under the MIT License - see [LICENSE](/LICENSE) file for details.
