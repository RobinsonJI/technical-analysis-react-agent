from typing import List
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.data.basemodels import MovingAverages
from src.data.enums import TimePeriod, Interval

@tool(args_schema=MovingAverages)
def relative_strength_index(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval, 
    period: int = 14) -> List[float]:
    """
    Calculate the Relative Strength Index (RSI) for a given ticker.
    
    RSI measures momentum by comparing the magnitude of recent gains to recent losses.
    Values range from 0 to 100, where:
        - RSI > 70: Stock may be overbought (potential sell signal)
        - RSI < 30: Stock may be oversold (potential buy signal)
        - RSI 40-60: Neutral momentum
    
    Use this tool when the user asks about:
        - Momentum analysis
        - Overbought/oversold conditions
        - Trend strength confirmation
        - Divergence detection (price vs RSI)
    
    Args:
        data: TickData object containing closing prices
        period: RSI lookback period in intervals (default 14)
                Must have at least period+1 closing prices to calculate
                Shorter periods (7-9) for faster, more reactive signals
                Longer periods (21-28) for slower, less reactive signals
    
    Returns:
        List of RSI values (0-100 scale)
        Note: First period-1 values will be NaN due to calculation requirements
    
    Example:
        - 14-period RSI: Standard momentum (most common)
        - 7-period RSI: Faster signals (day trading/scalping)
        - 21-period RSI: Slower signals (swing trading)
    
    Note:
        Requires sufficient data points. For 14-period RSI, ensure at least 
        15 closing prices in your dataset.
    """
    logger.info(f"Running Relative Strength Index analysis for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    closes = np.array(closing_price)
    
    # Calculate price changes
    deltas = np.diff(closes)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gain and loss using EMA (Smooth Moving Average)
    avg_gains = np.zeros_like(closes)
    avg_losses = np.zeros_like(closes)
    
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # Smooth the averages using Wilder's smoothing
    for i in range(period + 1, len(closes)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, where=avg_losses != 0, out=np.zeros_like(avg_gains))
    rsi = 100 - (100 / (1 + rs))

    logger.info(f"Relative Strength Index analysis for {stock_name} (${ticker_symbol}) complete")

    
    # Return only valid RSI values (starting from period index)
    return rsi[period:].tolist()



@tool(args_schema=MovingAverages)
def simple_moving_average(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 20
    ) -> List[float]:
    """
    Calculate the Simple Moving Average (SMA) for a given ticker.
    
    Use this tool when the user asks about moving averages, trend analysis,
    or wants to smooth price data to identify trends over a specific period.

    Use SMA (Simple Moving Average) when:
        - You want equal weight to all periods
        - Analyzing long-term trends (50-day, 200-day)
        - Support/resistance levels (slower response)
        - Less sensitive to recent price spikes
        - Good for swing trading or position trading
    
    Args:
        data: TickData object containing closing prices
        period: SMA period in days (default 20 for short-term, use 50 or 200 for long-term)
            The period is the number of time intervals (days, hours, minutes, etc.) used to calculate the moving average or indicator.
    
    Returns:
        List of SMA values aligned with the closing prices
    
    Example:
        - 20-period SMA: Short-term trend
        - 50-period SMA: Medium-term trend
        - 200-period SMA: Long-term trend
    """
    logger.info(f"Computing Simple Moving Average for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    closes = np.array(closing_price, dtype=float)
    sma = np.full_like(closes, np.nan)
    
    for i in range(period - 1, len(closes)):
        sma[i] = np.mean(closes[i - period + 1:i + 1])
    
    logger.info(f"Computed Simple Moving Average for {stock_name} (${ticker_symbol})")
    return sma.tolist()



@tool(args_schema=MovingAverages)
def exponential_moving_average(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 12) -> List[float]:
    """
    Calculate the Exponential Moving Average (EMA) for a given ticker.
    
    Use this tool when the user asks about short-term trends, day trading signals,
    or wants to detect quick price reversals with emphasis on recent prices.
    
    Use EMA (Exponential Moving Average) when:
        - You want recent prices weighted more heavily than older prices
        - Day trading or scalping (short-term trading)
        - Need faster response to price changes
        - Detecting early trend reversals
        - Less suited for long-term support/resistance levels
    
    Args:
        data: TickData object containing closing prices
        period: EMA period in days (default 12 for short-term, use 26 for medium-term)
    
    Returns:
        List of EMA values aligned with the closing prices (first period-1 values are NaN)
    
    Example:
        - 12-period EMA: Short-term trend (day trading)
        - 26-period EMA: Medium-term trend (swing trading)
    """

    logger.info(f"Computing Exponential Moving Average for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")


    closes = np.array(closing_price, dtype=float)
    ema = np.full_like(closes, np.nan)
    
    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(closes[:period])
    
    for i in range(period, len(closes)):
        ema[i] = closes[i] * multiplier + ema[i-1] * (1 - multiplier)

    logger.info(f"Computed Exponential Moving Average for {stock_name} (${ticker_symbol})")
    return ema.tolist()


