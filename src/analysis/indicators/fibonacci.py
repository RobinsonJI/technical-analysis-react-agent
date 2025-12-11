from typing import Dict, Literal, List
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.data.basemodels import TickData, FibExtensionDataAndParams
from src.data.enums import Interval, TimePeriod, FibonacciExtensionPattern
from src.utils.helpers import get_current_date, get_date_from_index

@tool(args_schema=FibExtensionDataAndParams)
def fibonacci_extension(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    pattern : FibonacciExtensionPattern,
) -> Dict:
    """
    Calculate Fibonacci Extension levels using TickData.
    
    Supports two patterns:
    - low_high_low: Bullish extension (price breaks above the high)
    - high_low_high: Bearish extension (price breaks below the low)
    
    Use for identifying potential price targets beyond the original trend.
    Extensions show where price may go after breaking through retracement levels
    and starting a new impulse wave.
    
    Args:
        data: TickData object containing price data
        pattern: Either "low_high_low" (bullish) or "high_low_high" (bearish)
        first_index: Index of first swing point (default: auto-detect)
        second_index: Index of second swing point (default: auto-detect)
        third_index: Index of third swing point (default: auto-detect)
    
    Returns:
        Dictionary with extension levels (161.8%, 261.8%, 361.8%)
        Also includes the swing prices and pattern details

    Limitations:
        - Works best on significant short-term trends with clear pivot points
        - Requires longer lookback window for longer-term trends
        - Poor performance during consolidation periods with no clear trend
        - Fundamental events (black swan, policy changes) can invalidate projections
        
    Example:
        >>> # Bullish pattern: Low -> High -> Low, expect break above high
        >>> extension = fibonacci_extension(
        ...     data=tick_data,
        ...     pattern="low_high_low"
        ... )
        
        >>> # Bearish pattern: High -> Low -> High, expect break below low
        >>> extension = fibonacci_extension(
        ...     data=tick_data,
        ...     pattern="high_low_high"
        ... )
    """
    logger.info(f"Computing Fibonacci Extension for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Fibonacci pattern: {pattern.value}")
    closes = np.array(closing_price, dtype=float)
    
    if pattern == "low_high_low":
        # Bullish pattern: Low -> High -> Low
        # Extensions calculate targets above the high
        first_index = np.argmin(closes)  # First low
        second_index = np.argmax(closes[first_index:]) + first_index  # High after first low
        third_index = first_index + np.argmin(closes[first_index:]) if first_index < len(closes) - 1 else first_index  # New low
    
        first_price = closes[first_index]    # Initial low
        second_price = closes[second_index]  # High
        third_price = closes[third_index]    # New low (retracement)
        
        # Original trend range (low to high)
        trend_range = second_price - first_price
        
        # Extension levels from the new low (third point)
        extension_127_2 = third_price + (trend_range * 0.272)
        extension_161_8 = third_price + (trend_range * 0.618)
        extension_261_8 = third_price + (trend_range * 1.618)
        extension_361_8 = third_price + (trend_range * 2.618)
        
        pattern_name = "Low-High-Low (Bullish)"
        direction = "Upward"
        
    elif pattern == "high_low_high":
        # Bearish pattern: High -> Low -> High
        # Extensions calculate targets below the low
        first_index = np.argmax(closes)  # First high
        second_index = np.argmin(closes[first_index:]) + first_index  # Low after first high
        third_index = first_index + np.argmax(closes[first_index:]) if first_index < len(closes) - 1 else first_index  # New high
        
        first_price = closes[first_index]    # Initial high
        second_price = closes[second_index]  # Low
        third_price = closes[third_index]    # New high (retracement)
        
        # Original trend range (high to low)
        trend_range = first_price - second_price
        
        # Extension levels from the new high (third point)
        extension_127_2 = third_price - (trend_range * 0.272)
        extension_161_8 = third_price - (trend_range * 0.618)
        extension_261_8 = third_price - (trend_range * 1.618)
        extension_361_8 = third_price - (trend_range * 2.618)
        
        pattern_name = "High-Low-High (Bearish)"
        direction = "Downward"
    
    else:
        raise ValueError("Pattern must be either 'low_high_low' or 'high_low_high'")
    
     # Calculate wave sizes
    first_wave = abs(second_price - first_price)
    second_wave = abs(third_price - second_price)

     # Calculate price distance from current close to extensions
    current_close = closes[-1]
    distance_to_161_8 = abs(extension_161_8 - current_close)
    distance_to_261_8 = abs(extension_261_8 - current_close)
    distance_to_361_8 = abs(extension_361_8 - current_close)

    logger.info(f"Fibonacci Extension for {stock_name} (${ticker_symbol}) computed")
    return {
        "pattern": pattern_name,
        "direction": direction,
        "first_price": float(first_price),
        "first_date": get_date_from_index(timestamps, first_index),
        "second_price": float(second_price),
        "second_date": get_date_from_index(timestamps, second_index),
        "third_price": float(third_price),
        "third_date": get_date_from_index(timestamps, third_index),
        "current_close": float(current_close),
        "current_date": get_current_date(),
        "trend_range": float(trend_range),
        "extension_127_2": float(extension_127_2),
        "extension_161_8": float(extension_161_8),
        "extension_261_8": float(extension_261_8),
        "extension_361_8": float(extension_361_8),
        "metadata": {
            "ticker": ticker_symbol,
            "stock_name": stock_name,
            "pattern_type": pattern,
            "time_period": time_period.value,
            "interval": interval.value,
            "analysis_date": get_current_date(),
            # Indices
            "first_index": int(first_index),
            "second_index": int(second_index),
            "third_index": int(third_index),
            "total_data_points": len(closes),
            # Wave analysis
            "first_wave_size": float(first_wave),
            "second_wave_size": float(second_wave),
            "wave_ratio": float(second_wave / first_wave) if first_wave != 0 else 0.0,
            # Distance metrics (for proximity alerts)
            "distance_to_127_2": float(abs(extension_127_2 - current_close)),
            "distance_to_161_8": float(distance_to_161_8),
            "distance_to_261_8": float(distance_to_261_8),
            "distance_to_361_8": float(distance_to_361_8),
            # Completion percentage
            "completion_to_161_8_pct": float((distance_to_161_8 / trend_range) * 100) if trend_range != 0 else 0.0,
            "completion_to_261_8_pct": float((distance_to_261_8 / trend_range) * 100) if trend_range != 0 else 0.0,
        }
    }



@tool(args_schema=TickData)
def fibonacci_retracement(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval
) -> Dict:
    """
    Calculate Fibonacci Retracement levels using TickData.
    
    Use for identifying potential support and resistance levels during price pullbacks.
    Retracement levels show where price may find support before continuing the trend.
    
    Args:
        data: TickData object containing price data
        high_index: Index of the swing high (default: highest close)
        low_index: Index of the swing low (default: lowest close)
    
    Returns:
        Dictionary with Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        Also includes the high and low prices used in calculation

    Limitations:
        - Works best on significant short-term trends with clear pivot points
        - Requires longer lookback window for longer-term trends
        - Poor performance during consolidation periods with no clear trend
        - Fundamental events (black swan, policy changes) can invalidate projections
    
    Example:
        >>> retracement = fibonacci_retracement(
        ...     data=tick_data,
        ...     high_index=50,
        ...     low_index=30
        ... )
        >>> # Shows support/resistance levels during pullback
    """
    logger.info(f"Computing Fibonacci Retracement for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    closes = np.array(closing_price, dtype=float)
    
    # Find swing points if not provided
    high_index = np.argmax(closes)
    low_index = np.argmin(closes)
    
    # Extract prices
    high_price = closes[high_index]
    low_price = closes[low_index]
    current_close = closes[-1]

    # Price difference
    diff = high_price - low_price
    
    # Calculate retracement levels
    level_23_6 = high_price - (diff * 0.236)
    level_38_2 = high_price - (diff * 0.382)
    level_50_0 = high_price - (diff * 0.500)
    level_61_8 = high_price - (diff * 0.618)
    level_78_6 = high_price - (diff * 0.786)

    # Find closest support/resistance level
    levels = {
        "23.6%": level_23_6,
        "38.2%": level_38_2,
        "50.0%": level_50_0,
        "61.8%": level_61_8,
        "78.6%": level_78_6,
    }
    
    closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_close))
    
    # Calculate distance to levels
    distances = {
        "distance_to_23_6": abs(level_23_6 - current_close),
        "distance_to_38_2": abs(level_38_2 - current_close),
        "distance_to_50_0": abs(level_50_0 - current_close),
        "distance_to_61_8": abs(level_61_8 - current_close),
        "distance_to_78_6": abs(level_78_6 - current_close),
    }

    # Calculate retracement percentage (where are we in the pullback)
    if high_price != low_price:
        retracement_pct = ((high_price - current_close) / diff) * 100
    else:
        retracement_pct = 0.0
    logger.info(f"Fibonacci Retracement for {stock_name} (${ticker_symbol}) computed")
    return {
        "high_price": float(high_price),
        "high_date": get_date_from_index(timestamps, high_index),
        "low_price": float(low_price),
        "low_date": get_date_from_index(timestamps, low_index),
        "current_close": float(current_close),
        "current_date": get_current_date(),
        "level_0": float(high_price),
        "level_23_6": float(level_23_6),
        "level_38_2": float(level_38_2),
        "level_50_0": float(level_50_0),
        "level_61_8": float(level_61_8),
        "level_78_6": float(level_78_6),
        "level_100": float(low_price),
        "metadata": {
            "ticker": ticker_symbol,
            "stock_name": stock_name,
            "pattern_type": "Fibonacci Retracement",
            "time_period": time_period.value,
            "interval": interval.value,
            "analysis_date": get_current_date(),
            # Indices
            "high_index": int(high_index),
            "low_index": int(low_index),
            "total_data_points": len(closes),
            # Price metrics
            "range": float(diff),
            "current_retracement_pct": float(retracement_pct),
            "closest_level": closest_level[0],
            "closest_level_price": float(closest_level[1]),
            # Distance metrics
            **{f"distance_to_{k.replace('.', '_')}": float(v) for k, v in distances.items()},
        }
    }