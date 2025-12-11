from typing import List

from pydantic import BaseModel, Field

from src.data.enums import Interval, TimePeriod, FibonacciExtensionPattern

class TickData(BaseModel):
    """The ticker symbol date pulled from the yfinance API."""
    stock_name : str = Field(description="The full name of the stock")
    ticker_symbol : str = Field(description="The stock ticker (symbol)")
    opening_price : List[float] = Field(description="The opening price of the stock ticker symbol for the specified time period.")
    closing_price : List[float] = Field(description="The closing price of the stock ticker symbol for the specified time period.")
    volume: List[int] = Field(description="The volume (or amount of stock traded) for the ticker symbol for the specified time period.")
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH, 
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
        )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
        )
    
class GetTickerSymbolDateRangeAndInterval(BaseModel):
    """Input to make a GET request to yfinance for tick data for a specific stock."""
    stock_name : str = Field(description="The full name of the stock")
    ticker_symbol : str = Field(description="The stock ticker (symbol)")
    start_date : str = Field(description="The start date of ticker data (format: YYYY-MM-DD)")
    end_date : str = Field(description="The end date of ticker data (format: YYYY-MM-DD)")
    time_period: TimePeriod = Field(default=TimePeriod.ONE_MONTH)
    interval: Interval = Field(default=Interval.ONE_DAY)

class FibExtensionDataAndParams(BaseModel):
    """Input to compute Fibonacci Extensions using tick data from yfinance."""
    stock_name : str = Field(description="The full name of the stock")
    ticker_symbol : str = Field(description="The stock ticker (symbol)")
    opening_price : List[float] = Field(description="The opening price of the stock ticker symbol for the specified time period.")
    closing_price : List[float] = Field(description="The closing price of the stock ticker symbol for the specified time period.")
    volume: List[int] = Field(description="The volume (or amount of stock traded) for the ticker symbol for the specified time period.")
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH, 
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
        )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
        )
    pattern : FibonacciExtensionPattern  = Field(
        default=FibonacciExtensionPattern.BULLISH,
        description="Pattern type: BULLISH (low_high_low) or BEARISH (high_low_high)"
    )

class MovingAverages(BaseModel):
    """Input to compute Fibonacci Extensions using tick data from yfinance."""
    stock_name : str = Field(description="The full name of the stock")
    ticker_symbol : str = Field(description="The stock ticker (symbol)")
    opening_price : List[float] = Field(description="The opening price of the stock ticker symbol for the specified time period.")
    closing_price : List[float] = Field(description="The closing price of the stock ticker symbol for the specified time period.")
    volume: List[int] = Field(description="The volume (or amount of stock traded) for the ticker symbol for the specified time period.")
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH, 
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
        )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
        )
    period : int  = Field(
        description="The number of bars used in the moving average calculation."
    )

class MACD(BaseModel):
    """Input to compute MACD (Moving Average Convergence Divergence) using tick data from yfinance."""
    stock_name : str = Field(
        description="The full name of the stock"
    )
    ticker_symbol : str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price : List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price : List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH, 
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    fast: int = Field(
        default=12,
        description="Fast EMA period for MACD line calculation (typical range: 5-14, default: 12)",
        ge=5,
        le=50
    )
    slow: int = Field(
        default=26,
        description="Slow EMA period for MACD line calculation (typical range: 20-35, default: 26, must be > fast period)",
        ge=20,
        le=100
    )
    signal: int = Field(
        default=9,
        description="Signal line EMA period (trigger line, typical range: 5-14, default: 9)",
        ge=3,
        le=20
    )

class StochasticOscillator(BaseModel):
    """Input to compute Stochastic Oscillator using tick data from yfinance."""
    stock_name : str = Field(
        description="The full name of the stock"
    )
    ticker_symbol : str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price : List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price : List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH, 
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    period: int = Field(
        default=14,
        description="Lookback period for stochastic calculation (typical range: 5-21, default: 14)",
        ge=5,
        le=50
    )
    smooth_k: int = Field(
        default=3,
        description="K line smoothing period - creates Fast %K (typical range: 1-5, default: 3)",
        ge=1,
        le=10
    )
    smooth_d: int = Field(
        default=3,
        description="D line smoothing period - SMA of Fast %K, creates Slow %K (typical range: 1-5, default: 3)",
        ge=1,
        le=10
    )

class PivotPoints(BaseModel):
    """Input to compute Pivot Points for support and resistance levels using tick data from yfinance."""
    stock_name: str = Field(
        description="The full name of the stock"
    )
    ticker_symbol: str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price: List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price: List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH,
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    high_prices: List[float] = Field(
        default=None,
        description="List of high prices for each period (optional - will derive from opening/closing if not provided)"
    )
    low_prices: List[float] = Field(
        default=None,
        description="List of low prices for each period (optional - will derive from opening/closing if not provided)"
    )


class BollingerBands(BaseModel):
    """Input to compute Bollinger Bands with squeeze + short setup detection using tick data from yfinance."""
    stock_name: str = Field(
        description="The full name of the stock"
    )
    ticker_symbol: str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price: List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price: List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH,
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    period: int = Field(
        default=20,
        description="SMA period for middle band calculation (typical range: 10-50, default: 20)",
        ge=10,
        le=50
    )
    std_dev: int = Field(
        default=2,
        description="Number of standard deviations for band calculation (typical range: 1-3, default: 2)",
        ge=1,
        le=3
    )

class ATR(BaseModel):
    """Input to compute Average True Range (ATR) for volatility measurement using tick data from yfinance."""
    stock_name: str = Field(
        description="The full name of the stock"
    )
    ticker_symbol: str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price: List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price: List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH,
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    high_prices: List[float] = Field(
        default=None,
        description="List of high prices for each period (optional - will derive from max(open, close) if not provided)"
    )
    low_prices: List[float] = Field(
        default=None,
        description="List of low prices for each period (optional - will derive from min(open, close) if not provided)"
    )
    period: int = Field(
        default=14,
        description="ATR calculation period (typical range: 5-21, default: 14)",
        ge=5,
        le=21
    )


class Volume(BaseModel):
    """Input to compute Volume Analysis and OBV using tick data from yfinance."""
    stock_name: str = Field(
        description="The full name of the stock"
    )
    ticker_symbol: str = Field(
        description="The stock ticker (symbol)"
    )
    opening_price: List[float] = Field(
        description="The opening price of the stock ticker symbol for the specified time period."
    )
    closing_price: List[float] = Field(
        description="The closing price of the stock ticker symbol for the specified time period."
    )
    volume: List[int] = Field(
        description="The volume (or amount of stock traded) for the ticker symbol for the specified time period."
    )
    timestamps: List[str] = Field(
        description="List of timestamps for each data point (ISO 8601 format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD for daily data)"
    )
    time_period: TimePeriod = Field(
        default=TimePeriod.ONE_MONTH,
        description="""The time period for which the analysis is being conducted. 
        Use one month if the user is asking about long term investment. Use shorter timeframes if the user is asking about trading or scalping."""
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="""The intervals between ticker symbol opens and closes. 
        Use daily intervals if the user is asking about long term investment. Use shorter intervals if the user is asking about trading or scalping."""
    )
    period: int = Field(
        default=20,
        description="Volume moving average period (typical range: 5-50, default: 20)",
        ge=5,
        le=50
    )

class WebSearch(BaseModel):
    """Input to perform internet search using DuckDuckGo for news and market information."""
    query: str = Field(
        description="The search query string to find relevant information (e.g., 'Apple stock news', 'market conditions today', 'company earnings report')"
    )
    news: bool = Field(
        default=False,
        description="If True, search specifically for news articles. If False, perform a general web search"
    )