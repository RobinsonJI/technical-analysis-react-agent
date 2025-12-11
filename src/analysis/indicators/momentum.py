from typing import List, Dict
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.analysis.indicators.trend import exponential_moving_average
from src.data.enums import Interval, TimePeriod
from src.data.basemodels import MACD, StochasticOscillator, MovingAverages


@tool(args_schema=MACD)
def macd(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict:
    """
    Calculate MACD (Moving Average Convergence Divergence) with momentum change timestamps.
    
    MACD is a trend-following momentum indicator that shows the relationship between 
    two moving averages. It consists of three components:
    - MACD Line: The difference between fast (12-period) and slow (26-period) EMAs
    - Signal Line: A 9-period EMA of the MACD line (acts as a trigger)
    - Histogram: The difference between MACD line and signal line (momentum strength)
    
    **Signal Interpretation:**
    - MACD > 0: Bullish momentum (price above average)
    - MACD < 0: Bearish momentum (price below average)
    - MACD crossover above signal line: BUY signal (bullish crossover)
    - MACD crossover below signal line: SELL signal (bearish crossover)
    - Histogram > 0 and increasing: Strong bullish momentum
    - Histogram < 0 and decreasing: Strong bearish momentum
    - Histogram near zero: Momentum weakening, potential reversal
    
    **Best Used For:**
    - Trend confirmation and identification
    - Momentum divergence detection
    - Entry/exit signal generation
    - Trend reversal early warning
    
    **Limitations:**
    - Lagging indicator (uses past data)
    - Prone to false signals in sideways/consolidating markets
    - Less effective in ranging markets
    - Requires confirmation from other indicators
    
    Args:
        closing_prices: List of closing prices (minimum 26 periods required)
        timestamps: List of ISO 8601 timestamp strings (same length as closing_prices)
        fast: Fast EMA period (default 12, typical range 5-14)
        slow: Slow EMA period (default 26, typical range 20-35)
        signal: Signal line EMA period (default 9, typical range 5-14)
    
    Returns:
        Dictionary containing:
            - macd_line (List[float]): MACD line values
            - signal_line (List[float]): Signal line values (trigger)
            - histogram (List[float]): MACD histogram (momentum strength)
            - latest_macd (float): Most recent MACD value
            - latest_signal (float): Most recent signal value
            - latest_histogram (float): Most recent histogram value
            - crossover_type (str): "BULLISH", "BEARISH", or "NONE"
            - momentum_strength (str): "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
            - histogram_trend (str): "EXPANDING_UP", "CONTRACTING", "EXPANDING_DOWN", "NEUTRAL"
            - divergence_detected (bool): True if potential divergence detected
            - event_timestamps (Dict): Key dates for momentum changes
            - metadata (Dict): Analysis metadata, timestamps, and statistics
    
    Raises:
        ValueError: If closing_prices has fewer than slow+1 periods
        TypeError: If closing_prices contains non-numeric values
    
    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112]
        >>> dates = ['2025-01-01', '2025-01-02', ..., '2025-01-13']
        >>> result = macd(prices, dates, fast=12, slow=26, signal=9)
        >>> print(f"MACD: {result['latest_macd']:.4f}")
        >>> print(f"Last Bullish Crossover: {result['event_timestamps']['last_bullish_crossover']}")
    """
    logger.info(f"Computing Moving Average Convergence Divergence for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")
    calculation_start = datetime.now()
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    closes = np.array(closing_price, dtype=float)

    logger.info(f"Finished computing Moving Average Convergence Divergence for {stock_name} (${ticker_symbol})")

    tick_data = {
        "stock_name" : stock_name,
        "ticker_symbol" : ticker_symbol,
        "opening_price" : opening_price,
        "closing_price" : closing_price,
        "volume": volume,
        "timestamps": timestamps,
        "time_period": time_period,
        "interval": interval,
    }
    
    # Calculate EMAs
    fast_tick_data = tick_data
    fast_tick_data["period"] = fast
    fast_ema = np.array(exponential_moving_average.func(**fast_tick_data), float)

    slow_tick_data = tick_data
    slow_tick_data["period"] = slow
    slow_ema = np.array(exponential_moving_average.func(**slow_tick_data), float)
    
    # MACD line
    macd_line = np.array(fast_ema, float) - np.array(slow_ema, float)
    
    # Signal line
    signal_tick_data = tick_data
    signal_tick_data["period"] = signal
    signal_line = np.array(exponential_moving_average.func(**signal_tick_data), float)
    
    # Histogram
    histogram = macd_line - signal_line
    
    # Get latest values
    latest_macd = float(macd_line[-1])
    latest_signal = float(signal_line[-1])
    latest_histogram = float(histogram[-1])
    
    # Find crossover events and momentum changes
    crossover_events = _find_macd_crossovers(macd_line, signal_line, histogram, timestamps)
    momentum_changes = _find_macd_momentum_changes(macd_line, histogram, timestamps)
    zero_crossovers = _find_zero_crossovers(histogram, timestamps)
    
    # Determine crossover type
    prev_histogram = float(histogram[-2]) if len(histogram) > 1 else 0
    if latest_histogram > 0 and prev_histogram <= 0:
        crossover_type = "BULLISH"
    elif latest_histogram < 0 and prev_histogram >= 0:
        crossover_type = "BEARISH"
    else:
        crossover_type = "NONE"
    
    # Determine momentum strength
    if latest_macd > latest_signal and latest_histogram > 0:
        if latest_histogram > abs(latest_macd) * 0.1:
            momentum_strength = "STRONG_BULLISH"
        else:
            momentum_strength = "BULLISH"
    elif latest_macd < latest_signal and latest_histogram < 0:
        if abs(latest_histogram) > abs(latest_macd) * 0.1:
            momentum_strength = "STRONG_BEARISH"
        else:
            momentum_strength = "BEARISH"
    else:
        momentum_strength = "NEUTRAL"
    
    # Histogram trend
    if len(histogram) > 2:
        prev_prev_histogram = float(histogram[-3])
        if latest_histogram > prev_histogram > prev_prev_histogram:
            histogram_trend = "EXPANDING_UP"
        elif latest_histogram < prev_histogram < prev_prev_histogram:
            histogram_trend = "EXPANDING_DOWN"
        elif abs(latest_histogram) < abs(prev_histogram):
            histogram_trend = "CONTRACTING"
        else:
            histogram_trend = "NEUTRAL"
    else:
        histogram_trend = "NEUTRAL"
    
    # Detect divergence
    if len(closes) > 5:
        bearish_divergence = (closes[-1] > closes[-5] and latest_macd < macd_line[-5])
        bullish_divergence = (closes[-1] < closes[-5] and latest_macd > macd_line[-5])
        divergence_detected = bearish_divergence or bullish_divergence
    else:
        divergence_detected = False
    
    # Calculate statistics - ensure all are numpy arrays
    macd_array = np.array(macd_line, dtype=float)
    signal_array = np.array(signal_line, dtype=float)
    histogram_array = np.array(histogram, dtype=float)
    
    macd_array = macd_array[~np.isnan(macd_array)]
    signal_array = signal_array[~np.isnan(signal_array)]
    histogram_array = histogram_array[~np.isnan(histogram_array)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    
    return {
        # Core values
        "macd_line": macd_line.tolist(),
        "signal_line": signal_line.tolist(),
        "histogram": histogram.tolist(),
        
        # Latest values
        "latest_macd": latest_macd,
        "latest_signal": latest_signal,
        "latest_histogram": latest_histogram,
        
        # Signal analysis
        "crossover_type": crossover_type,
        "momentum_strength": momentum_strength,
        "histogram_trend": histogram_trend,
        "divergence_detected": divergence_detected,
        
        # Event timestamps (KEY DATES FOR MOMENTUM CHANGES)
        "event_timestamps": {
            "last_bullish_crossover": crossover_events.get("last_bullish_crossover"),
            "last_bearish_crossover": crossover_events.get("last_bearish_crossover"),
            "bullish_crossover_dates": crossover_events.get("bullish_crossovers", []),
            "bearish_crossover_dates": crossover_events.get("bearish_crossovers", []),
            "last_momentum_acceleration": momentum_changes.get("last_acceleration_date"),
            "last_momentum_deceleration": momentum_changes.get("last_deceleration_date"),
            "momentum_acceleration_dates": momentum_changes.get("accelerations", []),
            "momentum_deceleration_dates": momentum_changes.get("decelerations", []),
            "last_zero_crossover": zero_crossovers.get("last_crossover_date"),
            "zero_crossover_direction": zero_crossovers.get("last_direction"),
            "all_zero_crossovers": zero_crossovers.get("all_crossovers", []),
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            # Data period info
            "time_period": time_period.value,
            "interval": interval.value,
            # Calculation info
            "calculation_type": "MACD (Moving Average Convergence Divergence)",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            # Parameters
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal,
            "total_data_points": len(closes),
            "valid_macd_points": int(np.sum(~np.isnan(macd_line))),
            
            # MACD statistics
            "macd_min": float(np.nanmin(macd_array)) if len(macd_array) > 0 else None,
            "macd_max": float(np.nanmax(macd_array)) if len(macd_array) > 0 else None,
            "macd_mean": float(np.nanmean(macd_array)) if len(macd_array) > 0 else None,
            "macd_std": float(np.nanstd(macd_array)) if len(macd_array) > 0 else None,
            "macd_current_from_mean": latest_macd - float(np.nanmean(macd_array)) if len(macd_array) > 0 else None,
            
            # Signal line statistics
            "signal_min": float(np.nanmin(signal_array)) if len(signal_array) > 0 else None,
            "signal_max": float(np.nanmax(signal_array)) if len(signal_array) > 0 else None,
            "signal_mean": float(np.nanmean(signal_array)) if len(signal_array) > 0 else None,
            
            # Histogram statistics
            "histogram_min": float(np.nanmin(histogram_array)) if len(histogram_array) > 0 else None,
            "histogram_max": float(np.nanmax(histogram_array)) if len(histogram_array) > 0 else None,
            "histogram_mean": float(np.nanmean(histogram_array)) if len(histogram_array) > 0 else None,
            
            # Distance metrics
            "macd_signal_distance": latest_macd - latest_signal,
            "macd_zero_distance": latest_macd,
            
            # Event counts
            "bullish_crossover_count": len(crossover_events.get("bullish_crossovers", [])),
            "bearish_crossover_count": len(crossover_events.get("bearish_crossovers", [])),
            "momentum_acceleration_count": len(momentum_changes.get("accelerations", [])),
            "momentum_deceleration_count": len(momentum_changes.get("decelerations", [])),
            "zero_crossover_count": len(zero_crossovers.get("all_crossovers", [])),
        }
    }


@tool(args_schema=StochasticOscillator)
def stochastic_oscillator(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Dict:
    """
    Calculate Stochastic Oscillator with momentum change timestamps.
    
    The Stochastic Oscillator measures the level of the close relative to the high-low 
    range over a set period. It's a momentum oscillator that shows overbought/oversold 
    conditions and potential reversals.
    
    **Signal Interpretation:**
    - %K > 80: Overbought (potential sell signal, short squeeze)
    - %K < 20: Oversold (potential buy signal, long squeeze)
    - %K in 20-80: Neutral zone
    - %K crosses above %D: BUY signal (bullish crossover)
    - %K crosses below %D: SELL signal (bearish crossover)
    
    Args:
        closing_prices: List of closing prices
        timestamps: List of ISO 8601 timestamp strings (same length as closing_prices)
        period: Lookback period (default 14)
        smooth_k: K line smoothing period (default 3)
        smooth_d: D line smoothing period (default 3)
    
    Returns:
        Dictionary with:
            - k_line, d_line values
            - Condition: "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
            - event_timestamps: Key dates for overbought/oversold conditions, crossovers
            - metadata: Statistics and event counts
    """
    logger.info(f"Computing Stochastic Oscillator for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")
    calculation_start = datetime.now()
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    closes = np.array(closing_price, dtype=float)
    
    # Calculate RAW %K (unsmoothed)
    k_line = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        highest = np.max(closes[i - period + 1:i + 1])
        lowest = np.min(closes[i - period + 1:i + 1])
        k_line[i] = 100 * (closes[i] - lowest) / (highest - lowest) if highest != lowest else 50
    
    # Smooth K line (Fast %K)
    k_smoothed = _moving_average(k_line, smooth_k)
    
    # D line (Slow %K - SMA of smoothed K)
    d_line = _moving_average(k_smoothed, smooth_d)
    
    # Get latest values
    latest_k = float(k_smoothed[-1]) if not np.isnan(k_smoothed[-1]) else None
    latest_d = float(d_line[-1]) if not np.isnan(d_line[-1]) else None
    
    # Find events - PASS CORRECT PARAMETERS
    overbought_events = _find_overbought_oversold_events(k_smoothed, timestamps, threshold_high=80, threshold_low=20)
    crossover_events = _find_stochastic_crossovers(k_smoothed, d_line, timestamps)
    divergence_events = _find_stochastic_divergence(k_line, k_smoothed, d_line, closes, timestamps)  # Pass k_line (raw)
    
    # Determine condition
    if latest_k is not None:
        if latest_k > 80:
            condition = "OVERBOUGHT"
        elif latest_k < 20:
            condition = "OVERSOLD"
        else:
            condition = "NEUTRAL"
    else:
        condition = "UNKNOWN"
    
    # Crossover detection
    if len(k_smoothed) > 1 and len(d_line) > 1:
        prev_k = float(k_smoothed[-2]) if not np.isnan(k_smoothed[-2]) else latest_k
        prev_d = float(d_line[-2]) if not np.isnan(d_line[-2]) else latest_d
        
        if prev_k <= prev_d and latest_k > latest_d:
            crossover_type = "BULLISH"
        elif prev_k >= prev_d and latest_k < latest_d:
            crossover_type = "BEARISH"
        else:
            crossover_type = "NONE"
    else:
        crossover_type = "NONE"
    
    # Calculate statistics
    k_valid = k_smoothed[~np.isnan(k_smoothed)]
    d_valid = d_line[~np.isnan(d_line)]
    k_raw_valid = k_line[~np.isnan(k_line)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    logger.info(f"Finished computing Stochastic Oscillator for {stock_name} (${ticker_symbol})")
    return {
        # Core values - RETURN BOTH RAW AND SMOOTHED
        "k_line_raw": k_line.tolist(),           # Raw %K (unsmoothed)
        "k_line_smoothed": k_smoothed.tolist(),  # Fast %K (smoothed)
        "d_line": d_line.tolist(),               # Slow %K (D line)
        
        # Latest values
        "latest_k": latest_k,                    # Latest Fast %K
        "latest_d": latest_d,                    # Latest Slow %K (D line)
        "latest_k_raw": float(k_line[-1]) if not np.isnan(k_line[-1]) else None,  # Latest raw
        
        # Signal analysis
        "condition": condition,
        "crossover_type": crossover_type,
        "strength_score": float(latest_k) if latest_k else None,
        
        # Event timestamps (KEY DATES FOR MOMENTUM CHANGES)
        "event_timestamps": {
            "last_overbought_entry": overbought_events.get("last_overbought_date"),
            "last_oversold_entry": overbought_events.get("last_oversold_date"),
            "last_overbought_exit": overbought_events.get("last_overbought_exit_date"),
            "last_oversold_exit": overbought_events.get("last_oversold_exit_date"),
            "overbought_entry_dates": overbought_events.get("overbought_entries", []),
            "oversold_entry_dates": overbought_events.get("oversold_entries", []),
            "last_bullish_crossover": crossover_events.get("last_bullish_crossover"),
            "last_bearish_crossover": crossover_events.get("last_bearish_crossover"),
            "bullish_crossover_dates": crossover_events.get("bullish_crossovers", []),
            "bearish_crossover_dates": crossover_events.get("bearish_crossovers", []),
            "last_divergence": divergence_events[-1] if divergence_events else None,
            "all_divergences": divergence_events[-5:],  # Last 5 divergences
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            # Data period info
            "time_period": time_period.value,
            "interval": interval.value,

            # Calculation info
            "calculation_type": "Stochastic Oscillator",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            # Parameters
            "period": period,
            "smooth_k": smooth_k,
            "smooth_d": smooth_d,
            "total_data_points": len(closes),
            
            # Raw K statistics
            "k_raw_min": float(np.nanmin(k_raw_valid)) if len(k_raw_valid) > 0 else None,
            "k_raw_max": float(np.nanmax(k_raw_valid)) if len(k_raw_valid) > 0 else None,
            "k_raw_mean": float(np.nanmean(k_raw_valid)) if len(k_raw_valid) > 0 else None,
            
            # Smoothed K statistics
            "k_smoothed_min": float(np.nanmin(k_valid)) if len(k_valid) > 0 else None,
            "k_smoothed_max": float(np.nanmax(k_valid)) if len(k_valid) > 0 else None,
            "k_smoothed_mean": float(np.nanmean(k_valid)) if len(k_valid) > 0 else None,
            
            # D line statistics
            "d_min": float(np.nanmin(d_valid)) if len(d_valid) > 0 else None,
            "d_max": float(np.nanmax(d_valid)) if len(d_valid) > 0 else None,
            "d_mean": float(np.nanmean(d_valid)) if len(d_valid) > 0 else None,
            
            # Distance metrics
            "k_d_distance": latest_k - latest_d if latest_k and latest_d else None,
            "k_overbought_distance": 80 - latest_k if latest_k else None,
            "k_oversold_distance": latest_k - 20 if latest_k else None,
            
            # Event counts
            "overbought_count": len(overbought_events.get("overbought_entries", [])),
            "oversold_count": len(overbought_events.get("oversold_entries", [])),
            "bullish_crossover_count": len(crossover_events.get("bullish_crossovers", [])),
            "bearish_crossover_count": len(crossover_events.get("bearish_crossovers", [])),
            "divergence_count": len(divergence_events),
        }
    }


@tool(args_schema=MovingAverages)
def roc(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 12
) -> Dict:
    """
    Calculate Rate of Change (ROC) with momentum change timestamps and divergence detection.
    
    ROC measures the percentage change in price over a specified period. It shows 
    how fast the price is changing. Positive ROC indicates upward momentum, while 
    negative ROC indicates downward momentum.
    
    **Signal Interpretation:**
    - ROC > 0: Upward momentum (bullish)
    - ROC < 0: Downward momentum (bearish)
    - ROC crosses above zero: Momentum turning positive (BUY signal)
    - ROC crosses below zero: Momentum turning negative (SELL signal)
    - High positive ROC: Strong buying pressure, potential overbought
    - Low negative ROC: Strong selling pressure, potential oversold
    - **ROC Divergence: Price and ROC moving in opposite directions = reversal signal**
    
    Args:
        closing_prices: List of closing prices (minimum 'period' + 1 periods)
        timestamps: List of ISO 8601 timestamp strings (same length as closing_prices)
        time_period: Time period of data
        interval: Interval of data
        period: ROC period in bars (default 12)
    
    Returns:
        Dictionary with:
            - roc_values: ROC percentage values
            - latest_roc: Most recent ROC value
            - momentum_direction: Classification of current momentum
            - event_timestamps: Key dates when momentum changed direction or divergence detected
            - metadata: Statistics and event counts
    """
    logger.info(f"Computing Rate of Change for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")
    calculation_start = datetime.now()
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    if len(closing_price) < period + 1:
        raise ValueError(f"Requires at least {period + 1} periods, got {len(closing_price)}")
    
    closes = np.array(closing_price, dtype=float)
    roc_values = np.full_like(closes, np.nan)
    
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            roc_values[i] = ((closes[i] - closes[i - period]) / closes[i - period]) * 100
    
    # Get latest value
    latest_roc = float(roc_values[-1]) if not np.isnan(roc_values[-1]) else None
    
    # Find momentum change events
    momentum_changes = _find_roc_momentum_changes(roc_values, timestamps)
    zero_crossovers = _find_roc_zero_crossovers(roc_values, timestamps)
    divergences = _find_roc_divergence(closes, roc_values, timestamps)  # NEW: Divergence detection
    
    # Momentum direction
    if latest_roc is not None:
        if latest_roc > 20:
            momentum_direction = "STRONGLY_BULLISH"
        elif latest_roc > 5:
            momentum_direction = "BULLISH"
        elif latest_roc > -5:
            momentum_direction = "NEUTRAL"
        elif latest_roc > -20:
            momentum_direction = "BEARISH"
        else:
            momentum_direction = "STRONGLY_BEARISH"
    else:
        momentum_direction = "UNKNOWN"
    
    roc_valid = roc_values[~np.isnan(roc_values)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    logger.info(f"Finished computing Rate of Change for {stock_name} (${ticker_symbol})")
    return {
        # Core values
        "roc_values": roc_values.tolist(),
        
        # Latest value
        "latest_roc": latest_roc,
        "momentum_direction": momentum_direction,
        
        # Event timestamps (KEY DATES FOR MOMENTUM CHANGES)
        "event_timestamps": {
            "last_momentum_acceleration": momentum_changes.get("last_acceleration_date"),
            "last_momentum_deceleration": momentum_changes.get("last_deceleration_date"),
            "momentum_acceleration_dates": momentum_changes.get("accelerations", []),
            "momentum_deceleration_dates": momentum_changes.get("decelerations", []),
            "last_zero_crossover": zero_crossovers.get("last_crossover_date"),
            "last_zero_crossover_direction": zero_crossovers.get("last_direction"),
            "bullish_zero_crossover_dates": zero_crossovers.get("bullish_crossovers", []),
            "bearish_zero_crossover_dates": zero_crossovers.get("bearish_crossovers", []),
            "last_divergence": divergences[-1] if divergences else None,
            "all_divergences": divergences[-5:],  # Last 5 divergences
            "bearish_divergence_count": len([d for d in divergences if d["type"] == "BEARISH"]),
            "bullish_divergence_count": len([d for d in divergences if d["type"] == "BULLISH"]),
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            # Data period info
            "time_period": time_period.value,
            "interval": interval.value,
            
            # Calculation info
            "calculation_type": "Rate of Change (ROC)",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            # Parameters
            "period": period,
            "total_data_points": len(closes),
            "valid_roc_points": int(np.sum(~np.isnan(roc_values))),
            
            # Statistics
            "roc_min": float(np.nanmin(roc_valid)) if len(roc_valid) > 0 else None,
            "roc_max": float(np.nanmax(roc_valid)) if len(roc_valid) > 0 else None,
            "roc_mean": float(np.nanmean(roc_valid)) if len(roc_valid) > 0 else None,
            "roc_std": float(np.nanstd(roc_valid)) if len(roc_valid) > 0 else None,
            
            # Event counts
            "momentum_acceleration_count": len(momentum_changes.get("accelerations", [])),
            "momentum_deceleration_count": len(momentum_changes.get("decelerations", [])),
            "zero_crossover_count": len(zero_crossovers.get("all_crossovers", [])),
            "bullish_zero_crossover_count": len(zero_crossovers.get("bullish_crossovers", [])),
            "bearish_zero_crossover_count": len(zero_crossovers.get("bearish_crossovers", [])),
            "divergence_count": len(divergences),  # NEW
        }
    }

# Helper functions for event detection

def _find_roc_divergence(closes, roc_values, timestamps, lookback: int = 5):
    """Detect divergence between price and ROC."""
    divergences = []
    
    for i in range(lookback, len(closes)):
        price_lookback = float(closes[i - lookback])
        price_now = float(closes[i])
        roc_lookback = float(roc_values[i - lookback]) if not np.isnan(roc_values[i - lookback]) else 0
        roc_now = float(roc_values[i]) if not np.isnan(roc_values[i]) else 0
        
        # Bearish divergence: Price higher but ROC lower
        # Price makes new high but momentum weakening = potential top
        if price_now > price_lookback and roc_now < roc_lookback and roc_now > 0:
            divergences.append({
                "type": "BEARISH",
                "strength": "STRONG" if roc_now < roc_lookback * 0.5 else "REGULAR",
                "date": timestamps[i],
                "price": price_now,
                "price_change": price_now - price_lookback,
                "roc": roc_now,
                "roc_change": roc_now - roc_lookback,
                "description": "Price higher but ROC declining - momentum weakening, potential reversal"
            })
        
        # Bullish divergence: Price lower but ROC higher
        # Price makes new low but momentum improving = potential bottom
        elif price_now < price_lookback and roc_now > roc_lookback and roc_now < 0:
            divergences.append({
                "type": "BULLISH",
                "strength": "STRONG" if roc_now > roc_lookback * 0.5 else "REGULAR",
                "date": timestamps[i],
                "price": price_now,
                "price_change": price_now - price_lookback,
                "roc": roc_now,
                "roc_change": roc_now - roc_lookback,
                "description": "Price lower but ROC improving - momentum strengthening, potential reversal"
            })
    
    return divergences

def _find_macd_crossovers(macd_line, signal_line, histogram, timestamps):
    """Find all MACD/Signal crossover dates."""
    bullish_crossovers = []
    bearish_crossovers = []
    
    for i in range(1, len(histogram)):
        prev_histogram = float(histogram[i - 1]) if not np.isnan(histogram[i - 1]) else 0
        curr_histogram = float(histogram[i]) if not np.isnan(histogram[i]) else 0
        
        if prev_histogram <= 0 and curr_histogram > 0:
            bullish_crossovers.append(timestamps[i])
        elif prev_histogram >= 0 and curr_histogram < 0:
            bearish_crossovers.append(timestamps[i])
    
    return {
        "bullish_crossovers": bullish_crossovers[-5:],  # Last 5
        "bearish_crossovers": bearish_crossovers[-5:],
        "last_bullish_crossover": bullish_crossovers[-1] if bullish_crossovers else None,
        "last_bearish_crossover": bearish_crossovers[-1] if bearish_crossovers else None,
    }


def _find_macd_momentum_changes(macd_line, histogram, timestamps):
    """Find acceleration and deceleration in MACD histogram."""
    accelerations = []
    decelerations = []
    
    for i in range(2, len(histogram)):
        prev_prev = float(histogram[i - 2]) if not np.isnan(histogram[i - 2]) else 0
        prev = float(histogram[i - 1]) if not np.isnan(histogram[i - 1]) else 0
        curr = float(histogram[i]) if not np.isnan(histogram[i]) else 0
        
        # Acceleration: histogram magnitude increasing
        if abs(curr) > abs(prev) > abs(prev_prev):
            accelerations.append(timestamps[i])
        # Deceleration: histogram magnitude decreasing
        elif abs(curr) < abs(prev) < abs(prev_prev):
            decelerations.append(timestamps[i])
    
    return {
        "accelerations": accelerations[-5:],
        "decelerations": decelerations[-5:],
        "last_acceleration_date": accelerations[-1] if accelerations else None,
        "last_deceleration_date": decelerations[-1] if decelerations else None,
    }


def _find_zero_crossovers(histogram, timestamps):
    """Find when MACD histogram crosses zero."""
    crossovers = []
    bullish = []
    bearish = []
    
    for i in range(1, len(histogram)):
        prev = float(histogram[i - 1]) if not np.isnan(histogram[i - 1]) else 0
        curr = float(histogram[i]) if not np.isnan(histogram[i]) else 0
        
        if prev <= 0 and curr > 0:
            crossovers.append({"date": timestamps[i], "direction": "BULLISH"})
            bullish.append(timestamps[i])
        elif prev >= 0 and curr < 0:
            crossovers.append({"date": timestamps[i], "direction": "BEARISH"})
            bearish.append(timestamps[i])
    
    return {
        "all_crossovers": crossovers[-5:],
        "bullish_crossovers": bullish[-5:],
        "bearish_crossovers": bearish[-5:],
        "last_crossover_date": crossovers[-1]["date"] if crossovers else None,
        "last_direction": crossovers[-1]["direction"] if crossovers else None,
    }


def _find_overbought_oversold_events(k_line, timestamps, threshold_high=80, threshold_low=20):
    """Find when stochastic enters/exits overbought/oversold zones."""
    overbought_entries = []
    oversold_entries = []
    overbought_exits = []
    oversold_exits = []
    
    for i in range(1, len(k_line)):
        prev_k = float(k_line[i - 1]) if not np.isnan(k_line[i - 1]) else 50
        curr_k = float(k_line[i]) if not np.isnan(k_line[i]) else 50
        
        # Overbought entry
        if prev_k <= threshold_high < curr_k:
            overbought_entries.append(timestamps[i])
        # Overbought exit
        if prev_k >= threshold_high > curr_k:
            overbought_exits.append(timestamps[i])
        # Oversold entry
        if prev_k >= threshold_low > curr_k:
            oversold_entries.append(timestamps[i])
        # Oversold exit
        if prev_k <= threshold_low < curr_k:
            oversold_exits.append(timestamps[i])
    
    return {
        "overbought_entries": overbought_entries[-5:],
        "oversold_entries": oversold_entries[-5:],
        "overbought_exits": overbought_exits[-5:],
        "oversold_exits": oversold_exits[-5:],
        "last_overbought_date": overbought_entries[-1] if overbought_entries else None,
        "last_oversold_date": oversold_entries[-1] if oversold_entries else None,
        "last_overbought_exit_date": overbought_exits[-1] if overbought_exits else None,
        "last_oversold_exit_date": oversold_exits[-1] if oversold_exits else None,
    }


def _find_stochastic_crossovers(k_line, d_line, timestamps):
    """Find K/D crossover dates."""
    bullish_crossovers = []
    bearish_crossovers = []
    
    for i in range(1, len(k_line)):
        prev_k = float(k_line[i - 1]) if not np.isnan(k_line[i - 1]) else 0
        prev_d = float(d_line[i - 1]) if not np.isnan(d_line[i - 1]) else 0
        curr_k = float(k_line[i]) if not np.isnan(k_line[i]) else 0
        curr_d = float(d_line[i]) if not np.isnan(d_line[i]) else 0
        
        if prev_k <= prev_d and curr_k > curr_d:
            bullish_crossovers.append(timestamps[i])
        elif prev_k >= prev_d and curr_k < curr_d:
            bearish_crossovers.append(timestamps[i])
    
    return {
        "bullish_crossovers": bullish_crossovers[-5:],
        "bearish_crossovers": bearish_crossovers[-5:],
        "last_bullish_crossover": bullish_crossovers[-1] if bullish_crossovers else None,
        "last_bearish_crossover": bearish_crossovers[-1] if bearish_crossovers else None,
    }


def _find_stochastic_divergence(k_line_raw, k_line_smoothed, d_line, closes, timestamps):
    """Detect divergence between price and K/D lines."""
    divergences = []
    
    # Use raw K line for more accurate divergence detection (more responsive to price)
    for i in range(5, len(closes)):
        price_5_ago = float(closes[i-5]) if i >= 5 else 0
        price_now = float(closes[i])
        k_5_ago = float(k_line_raw[i-5]) if not np.isnan(k_line_raw[i-5]) else 0
        k_now = float(k_line_raw[i]) if not np.isnan(k_line_raw[i]) else 0
        
        # Bearish divergence: price makes higher high, but K line makes lower high
        # Indicates momentum is weakening despite price strength
        if price_now > price_5_ago and k_now < k_5_ago and k_now > 50:
            divergences.append({
                "type": "BEARISH",
                "strength": "HIDDEN" if k_now > 70 else "REGULAR",
                "date": timestamps[i],
                "price": price_now,
                "price_change": price_now - price_5_ago,
                "k_line": k_now,
                "k_change": k_now - k_5_ago,
                "description": "Price higher but momentum declining - potential reversal"
            })
        
        # Bullish divergence: price makes lower low, but K line makes higher low
        # Indicates momentum is strengthening despite price weakness
        elif price_now < price_5_ago and k_now > k_5_ago and k_now < 50:
            divergences.append({
                "type": "BULLISH",
                "strength": "HIDDEN" if k_now < 30 else "REGULAR",
                "date": timestamps[i],
                "price": price_now,
                "price_change": price_now - price_5_ago,
                "k_line": k_now,
                "k_change": k_now - k_5_ago,
                "description": "Price lower but momentum improving - potential reversal"
            })
    
    return divergences


def _find_roc_momentum_changes(roc_values, timestamps):
    """Find ROC acceleration and deceleration points."""
    accelerations = []
    decelerations = []
    
    for i in range(2, len(roc_values)):
        prev_prev = float(roc_values[i - 2]) if not np.isnan(roc_values[i - 2]) else 0
        prev = float(roc_values[i - 1]) if not np.isnan(roc_values[i - 1]) else 0
        curr = float(roc_values[i]) if not np.isnan(roc_values[i]) else 0
        
        if curr > prev > prev_prev:
            accelerations.append(timestamps[i])
        elif curr < prev < prev_prev:
            decelerations.append(timestamps[i])
    
    return {
        "accelerations": accelerations[-5:],
        "decelerations": decelerations[-5:],
        "last_acceleration_date": accelerations[-1] if accelerations else None,
        "last_deceleration_date": decelerations[-1] if decelerations else None,
    }


def _find_roc_zero_crossovers(roc_values, timestamps):
    """Find when ROC crosses zero."""
    bullish = []
    bearish = []
    all_crossovers = []
    
    for i in range(1, len(roc_values)):
        prev = float(roc_values[i - 1]) if not np.isnan(roc_values[i - 1]) else 0
        curr = float(roc_values[i]) if not np.isnan(roc_values[i]) else 0
        
        if prev <= 0 and curr > 0:
            bullish.append(timestamps[i])
            all_crossovers.append({"date": timestamps[i], "direction": "BULLISH"})
        elif prev >= 0 and curr < 0:
            bearish.append(timestamps[i])
            all_crossovers.append({"date": timestamps[i], "direction": "BEARISH"})
    
    return {
        "bullish_crossovers": bullish[-5:],
        "bearish_crossovers": bearish[-5:],
        "all_crossovers": all_crossovers[-5:],
        "last_crossover_date": all_crossovers[-1]["date"] if all_crossovers else None,
        "last_direction": all_crossovers[-1]["direction"] if all_crossovers else None,
    }


def _moving_average(values: np.ndarray, period: int) -> np.ndarray:
    """Helper function to calculate simple moving average with NaN handling."""
    result = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        valid_values = values[i - period + 1:i + 1]
        valid_values = valid_values[~np.isnan(valid_values)]
        if len(valid_values) > 0:
            result[i] = np.mean(valid_values)
    return result