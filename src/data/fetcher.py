import yfinance as yf
from langchain.tools import tool

from .basemodels import GetTickerSymbolDateRangeAndInterval, TickData
from .enums import Interval, TimePeriod

@tool(args_schema=GetTickerSymbolDateRangeAndInterval)
def fetch_tick_data(
    stock_name: str,
    ticker_symbol: str,
    start_date: str,
    end_date: str,
    time_period: TimePeriod = TimePeriod.ONE_MONTH,
    interval: Interval = Interval.ONE_DAY
) -> TickData:
    """
    Fetch historical tick data for a stock using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AMC", "AAPL")
        start_date: Start date as string (format: "YYYY-MM-DD")
        end_date: End date as string (format: "YYYY-MM-DD")
        period: TimePeriod enum (overridden if start/end dates provided)
        interval: Interval enum for data granularity
    
    Returns:
        TickData object with OHLCV data
    
    Example:
        >>> data = fetch_tick_data(
        ...     ticker="AMC",
        ...     start_date="2022-01-01",
        ...     end_date="2025-09-12",
        ...     period=TimePeriod.ONE_YEAR,
        ...     interval=Interval.ONE_DAY
        ... )
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")

        # Download data using yfinance
        df = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            interval=interval.value,
            progress=False
        )
        
        # Extract OHLCV data
        open_prices = df["Open"][ticker_symbol].tolist()
        close_prices = df["Close"][ticker_symbol].tolist()
        volume = df["Volume"][ticker_symbol].tolist()


        # Extract timestamps from index
        timestamps = [idx.strftime("%Y-%m-%dT%H:%M:%S") for idx in df.index]
        
        # Create TickData object
        tick_data = TickData(
            stock_name=stock_name,
            ticker_symbol=ticker_symbol,
            opening_price=open_prices,
            closing_price=close_prices,
            volume=volume,
            timestamps=timestamps,
            period=time_period,
            interval=interval
        )
        
        return tick_data
    
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {str(e)}")
        raise