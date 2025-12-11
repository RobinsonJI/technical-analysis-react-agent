from typing import List, Dict
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.data.enums import Interval, TimePeriod
from src.data.basemodels import ATR, BollingerBands


@tool(args_schema=ATR)
def atr(
    stock_name: str,
    ticker_symbol: str,
    opening_price: List[float],
    closing_price: List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    high_prices: List[float] = None,
    low_prices: List[float] = None,
    period: int = 14
) -> Dict:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    ATR measures market volatility by analyzing the range of price movement.
    It's the average of true range values over a specified period.
    Used by traders to set stop-loss levels and position sizing.
    
    **True Range (TR)** is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    
    **Signal Interpretation:**
    - High ATR (>2% of price): High volatility, wider stops needed
    - Low ATR (<1% of price): Low volatility, tighter stops possible
    - ATR increasing: Volatility expanding (potential breakout)
    - ATR decreasing: Volatility contracting (potential squeeze)
    
    **Trading Applications:**
    - Set stop-loss: Price ± (2 × ATR)
    - Position sizing: Inverse of ATR (higher volatility = smaller position)
    - Breakout confirmation: Price moves > ATR value
    - Volatility squeeze: ATR near 52-week lows
    
    Args:
        stock_name: Full name of the stock
        ticker_symbol: Stock ticker symbol
        opening_price: List of opening prices for each period
        closing_price: List of closing prices for each period
        volume: List of volumes for each period
        timestamps: List of ISO 8601 timestamp strings
        time_period: Time period of data
        interval: Data interval
        high_prices: List of high prices (optional - will derive from open/close if not provided)
        low_prices: List of low prices (optional - will derive from open/close if not provided)
        period: ATR calculation period (default 14, range: 5-21)
    
    Returns:
        Dictionary with:
            - atr_values: ATR for each period
            - latest_atr: Current ATR value
            - atr_trend: Whether ATR is increasing/decreasing
            - volatility_assessment: HIGH/MEDIUM/LOW classification
            - stop_loss_suggestions: Suggested stop levels
            - metadata: Statistics and calculations info
    """

    logger.info(f"Running Average True Range analysis for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    calculation_start = datetime.now()
    
    # Input validation
    if len(opening_price) != len(closing_price):
        raise ValueError("opening_price and closing_price must have same length")
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_price and timestamps must have same length")
    
    if len(closing_price) < period + 1:
        raise ValueError(f"Requires at least {period + 1} periods, got {len(closing_price)}")
    
    opens = np.array(opening_price, dtype=float)
    closes = np.array(closing_price, dtype=float)
    
    # Derive high/low from open/close if not provided
    if high_prices is None:
        highs = np.maximum(opens, closes)  # High = max(open, close)
    else:
        highs = np.array(high_prices, dtype=float)
    
    if low_prices is None:
        lows = np.minimum(opens, closes)   # Low = min(open, close)
    else:
        lows = np.array(low_prices, dtype=float)
    
    # Validate arrays
    if len(highs) != len(closes) or len(lows) != len(closes):
        raise ValueError("high_prices, low_prices, and closing_price must have same length")
    
    # Calculate True Range
    tr = np.zeros_like(closes)
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
    
    # Calculate ATR using Wilder's smoothing method
    atr_values = np.full_like(closes, np.nan)
    atr_values[period - 1] = np.mean(tr[:period])
    
    for i in range(period, len(closes)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
    
    # Get latest ATR
    latest_atr = float(atr_values[-1]) if not np.isnan(atr_values[-1]) else None
    latest_price = float(closes[-1])
    latest_high = float(highs[-1])
    latest_low = float(lows[-1])
    
    # ATR as percentage of price
    atr_percentage = (latest_atr / latest_price * 100) if latest_price > 0 else 0
    
    # ATR trend (comparing last 5 periods)
    if len(atr_values) >= 6:
        atr_prev_5 = float(atr_values[-6]) if not np.isnan(atr_values[-6]) else latest_atr
        if latest_atr > atr_prev_5:
            atr_trend = "EXPANDING (Volatility increasing)"
        elif latest_atr < atr_prev_5:
            atr_trend = "CONTRACTING (Volatility decreasing)"
        else:
            atr_trend = "STABLE"
    else:
        atr_trend = "UNKNOWN"
    
    # Volatility assessment
    if atr_percentage > 2.0:
        volatility_assessment = "HIGH"
    elif atr_percentage > 1.0:
        volatility_assessment = "MEDIUM"
    else:
        volatility_assessment = "LOW"
    
    # Stop-loss suggestions
    stop_loss_tight = latest_price - latest_atr  # 1 ATR below
    stop_loss_conservative = latest_price - (1.5 * latest_atr)  # 1.5 ATR below
    take_profit = latest_price + (2 * latest_atr)  # 2 ATR above
    
    # Statistics
    valid_atr = atr_values[~np.isnan(atr_values)]
    tr_valid = tr[~np.isnan(tr)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    
    return {
        # Core values
        "atr_values": atr_values.tolist(),
        "true_range_values": tr.tolist(),
        
        # Latest metrics
        "latest_atr": latest_atr,
        "latest_atr_percentage": float(atr_percentage),
        "latest_price": latest_price,
        
        # Volatility analysis
        "atr_trend": atr_trend,
        "volatility_assessment": volatility_assessment,
        "volatility_level": f"{atr_percentage:.2f}% of price",
        
        # Trading recommendations
        "risk_management": {
            "stop_loss_tight": float(stop_loss_tight),
            "stop_loss_conservative": float(stop_loss_conservative),
            "take_profit_target": float(take_profit),
            "position_size_guidance": "Increase position size when ATR is LOW, reduce when ATR is HIGH"
        },
        
        # Statistics
        "atr_statistics": {
            "atr_min": float(np.nanmin(valid_atr)) if len(valid_atr) > 0 else None,
            "atr_max": float(np.nanmax(valid_atr)) if len(valid_atr) > 0 else None,
            "atr_mean": float(np.nanmean(valid_atr)) if len(valid_atr) > 0 else None,
            "atr_std": float(np.nanstd(valid_atr)) if len(valid_atr) > 0 else None,
        },
        
        "true_range_statistics": {
            "tr_min": float(np.nanmin(tr_valid)) if len(tr_valid) > 0 else None,
            "tr_max": float(np.nanmax(tr_valid)) if len(tr_valid) > 0 else None,
            "tr_mean": float(np.nanmean(tr_valid)) if len(tr_valid) > 0 else None,
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            "time_period": time_period.value,
            "interval": interval.value,
            
            "calculation_type": "Average True Range (ATR)",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            "parameters": {
                "period": period,
                "high_prices_source": "User provided" if high_prices else "Derived from max(open, close)",
                "low_prices_source": "User provided" if low_prices else "Derived from min(open, close)",
            },
            
            "data_info": {
                "total_data_points": len(closes),
                "valid_atr_points": int(np.sum(~np.isnan(atr_values))),
                "current_high": latest_high,
                "current_low": latest_low,
                "current_close": latest_price,
                "last_timestamp": timestamps[-1],
            },
            
            "note": "ATR calculated using Wilder's smoothing method (EMA-style). High/Low prices derived from open/close if not provided.",
        }
    }




@tool(args_schema=BollingerBands)
def bollinger_bands(
    stock_name : str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 20,
    std_dev: int = 2
) -> Dict:
    """
    Calculate Bollinger Bands with squeeze + short setup detection.
    
    Bollinger Bands consist of three lines:
    - **Upper Band**: SMA + (std_dev × StdDev)
    - **Middle Band**: Simple Moving Average (SMA)
    - **Lower Band**: SMA - (std_dev × StdDev)
    
    **NEW: Squeeze + Short Setup Detection**
    Identifies periods when:
    - Bands are extremely tight (volatility squeeze)
    - Price consolidates near lower band (short accumulation)
    - High breakout potential (often precedes explosive moves)
    
    The bands expand when volatility increases and contract when it decreases.
    Price typically bounces between the bands and reverts to the middle.
    
    **Signal Interpretation:**
    - Price touches UPPER band: Potentially overbought (short/sell signal)
    - Price touches LOWER band: Potentially oversold (long/buy signal)
    - Price inside bands: Normal trading range
    - Bands widening: Volatility increasing (breakout potential)
    - Bands narrowing: Volatility decreasing (squeeze, consolidation)
    - **SQUEEZE DETECTED + LOWER ZONE: Pre-breakout setup** 
    
    **Trading Strategies:**
    1. **Mean Reversion**: Buy at lower band, sell at upper band
    2. **Breakout**: Price close above/below bands = trend continuation
    3. **Squeeze Play**: Wait for bands to expand after squeeze (explosive move)
    4. **Support/Resistance**: Bands act as dynamic levels
    
    Args:
        stock_name: Stock name
        ticker_symbol: Stock ticker
        closing_prices: List of closing prices
        timestamps: List of ISO 8601 timestamp strings
        time_period: Time period of data
        interval: Data interval
        period: SMA period for middle band (default 20, range: 10-50)
        std_dev: Number of standard deviations (default 2, range: 1-3)
    
    Returns:
        Dictionary with:
            - Bollinger Band values (upper, middle, lower)
            - **Squeeze + short setup detection signals**
            - Current price position relative to bands
            - Band width and volatility metrics
            - Trading signals
            - Metadata
    """
    logger.info(f"Running Bollinger Band analysis for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    calculation_start = datetime.now()
    
    # Input validation
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    if len(closing_price) < period:
        raise ValueError(f"Requires at least {period} periods, got {len(closing_price)}")
    
    if std_dev < 1 or std_dev > 3:
        raise ValueError("std_dev should be between 1 and 3 (default 2)")
    
    closes = np.array(closing_price, dtype=float)
    
    # Middle band (SMA)
    middle_band = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        middle_band[i] = np.mean(closes[i - period + 1:i + 1])
    
    # Standard deviation
    std_deviation = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        std_deviation[i] = np.std(closes[i - period + 1:i + 1])
    
    # Bands
    upper_band = middle_band + (std_dev * std_deviation)
    lower_band = middle_band - (std_dev * std_deviation)
    band_width = upper_band - lower_band
    
    # NEW: Detect squeeze + short setups
    squeeze_analysis = _detect_squeeze_short_setup(
        closing_price, upper_band, lower_band, middle_band, band_width, timestamps
    )
    
    # Get latest values
    latest_close = float(closes[-1])
    latest_upper = float(upper_band[-1]) if not np.isnan(upper_band[-1]) else None
    latest_middle = float(middle_band[-1]) if not np.isnan(middle_band[-1]) else None
    latest_lower = float(lower_band[-1]) if not np.isnan(lower_band[-1]) else None
    latest_width = float(band_width[-1]) if not np.isnan(band_width[-1]) else None
    
    # Determine price position relative to bands
    if latest_close > latest_upper:
        position = "ABOVE_UPPER_BAND (Overbought/Strong Bullish)"
        proximity_pct = ((latest_close - latest_upper) / latest_width * 100) if latest_width > 0 else 0
    elif latest_close > latest_middle:
        position = "UPPER_HALF (Bullish bias)"
        proximity_pct = ((latest_close - latest_middle) / (latest_upper - latest_middle) * 100) if latest_upper > latest_middle else 0
    elif latest_close > latest_lower:
        position = "LOWER_HALF (Bearish bias)"
        proximity_pct = ((latest_close - latest_lower) / (latest_middle - latest_lower) * 100) if latest_middle > latest_lower else 0
    else:
        position = "BELOW_LOWER_BAND (Oversold/Strong Bearish)"
        proximity_pct = ((latest_lower - latest_close) / latest_width * 100) if latest_width > 0 else 0
    
    # Band squeeze detection
    valid_widths = band_width[~np.isnan(band_width)]
    median_width = float(np.median(valid_widths)) if len(valid_widths) > 0 else latest_width
    
    if latest_width < median_width * 0.5:
        squeeze_status = "SQUEEZE (Volatility very low - breakout potential)"
    elif latest_width < median_width:
        squeeze_status = "CONTRACTING (Volatility low)"
    elif latest_width > median_width * 1.5:
        squeeze_status = "EXPANDING (Volatility high)"
    else:
        squeeze_status = "NORMAL (Average volatility)"
    
    # Trading signals
    if latest_close > latest_upper:
        signal = "OVERBOUGHT - Consider selling or taking profits"
    elif latest_close < latest_lower:
        signal = "OVERSOLD - Consider buying or covering shorts"
    elif latest_close > latest_middle and latest_width < median_width:
        signal = "SQUEEZE + BULLISH - Potential upside breakout"
    elif latest_close < latest_middle and latest_width < median_width:
        signal = "SQUEEZE + BEARISH - Potential downside breakout"
    else:
        signal = "NEUTRAL - Continue monitoring"
    
    # NEW: Add squeeze + short signal
    if squeeze_analysis["squeeze_detected"] and squeeze_analysis["current_squeeze_score"] >= 6:
        if position == "LOWER_HALF (Bearish bias)" or "LOWER_ZONE" in str(squeeze_analysis["squeeze_signals"][-1].get("position_zone", "")):
            signal = "⚠️ SQUEEZE + SHORT SETUP DETECTED - Pre-breakout high volatility move likely"
    
    # Statistics
    valid_upper = upper_band[~np.isnan(upper_band)]
    valid_middle = middle_band[~np.isnan(middle_band)]
    valid_lower = lower_band[~np.isnan(lower_band)]
    valid_std = std_deviation[~np.isnan(std_deviation)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000

    logger.info(f"Bollinger Band analysis for {stock_name} (${ticker_symbol}) complete")
    
    return {
        # Core values
        "upper_band": upper_band.tolist(),
        "middle_band": middle_band.tolist(),
        "lower_band": lower_band.tolist(),
        "band_width": band_width.tolist(),
        
        # Latest metrics
        "latest_values": {
            "price": latest_close,
            "upper_band": latest_upper,
            "middle_band": latest_middle,
            "lower_band": latest_lower,
            "band_width": latest_width,
        },
        
        # Position analysis
        "price_position": {
            "position": position,
            "distance_from_middle": float(latest_close - latest_middle),
            "proximity_percentage": float(proximity_pct),
            "position_description": f"Price is {position}"
        },
        
        # Volatility analysis
        "volatility": {
            "squeeze_status": squeeze_status,
            "band_width_median": float(median_width),
            "band_width_current": float(latest_width),
            "width_ratio": float(latest_width / median_width) if median_width > 0 else 0,
        },
        
        # NEW: Squeeze + Short Setup Detection
        "squeeze_short_setup": squeeze_analysis,
        
        # Trading signals
        "trading_signal": signal,
        "signal_type": {
            "overbought": latest_close > latest_upper,
            "oversold": latest_close < latest_lower,
            "in_squeeze": latest_width < median_width * 0.5,
            "volatility_expanding": latest_width > median_width * 1.5,
            "squeeze_short_setup_detected": squeeze_analysis["squeeze_detected"] and squeeze_analysis["current_squeeze_score"] >= 6,
        },
        
        # Statistics
        "statistics": {
            "upper_band_min": float(np.nanmin(valid_upper)) if len(valid_upper) > 0 else None,
            "upper_band_max": float(np.nanmax(valid_upper)) if len(valid_upper) > 0 else None,
            "upper_band_mean": float(np.nanmean(valid_upper)) if len(valid_upper) > 0 else None,
            
            "middle_band_min": float(np.nanmin(valid_middle)) if len(valid_middle) > 0 else None,
            "middle_band_max": float(np.nanmax(valid_middle)) if len(valid_middle) > 0 else None,
            "middle_band_mean": float(np.nanmean(valid_middle)) if len(valid_middle) > 0 else None,
            
            "lower_band_min": float(np.nanmin(valid_lower)) if len(valid_lower) > 0 else None,
            "lower_band_max": float(np.nanmax(valid_lower)) if len(valid_lower) > 0 else None,
            "lower_band_mean": float(np.nanmean(valid_lower)) if len(valid_lower) > 0 else None,
            
            "std_dev_mean": float(np.nanmean(valid_std)) if len(valid_std) > 0 else None,
            "std_dev_max": float(np.nanmax(valid_std)) if len(valid_std) > 0 else None,
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            "time_period": time_period.value,
            "interval": interval.value,
            
            "calculation_type": "Bollinger Bands with Squeeze + Short Setup Detection",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            "parameters": {
                "sma_period": period,
                "standard_deviations": std_dev,
            },
            
            "data_info": {
                "total_data_points": len(closes),
                "valid_band_points": int(np.sum(~np.isnan(upper_band))),
                "current_close": latest_close,
                "last_timestamp": timestamps[-1],
            },
            
            "note": "Bollinger Bands are dynamic - bands adjust as volatility changes. Squeeze + Short setups often precede explosive moves.",
        }
    }



def _detect_squeeze_short_setup(
    closing_prices: List[float],
    upper_band: np.ndarray,
    lower_band: np.ndarray,
    middle_band: np.ndarray,
    band_width: np.ndarray,
    timestamps: List[str],
    lookback: int = 20
) -> Dict:
    """
    Detect squeeze + short setup patterns.
    
    **Squeeze + Short Setup indicates:**
    - Bollinger Bands extremely narrow (volatility squeeze)
    - Price consolidating near lower band (short accumulation)
    - RSI or momentum compressed (ready to explode)
    - Volume often low during squeeze
    
    These setups often precede explosive moves (either direction).
    """
    closes = np.array(closing_prices, dtype=float)
    valid_widths = band_width[~np.isnan(band_width)]
    median_width = float(np.median(valid_widths))
    std_width = float(np.std(valid_widths))
    
    squeeze_signals = []
    
    for i in range(max(0, len(closes) - lookback), len(closes)):
        current_width = float(band_width[i]) if not np.isnan(band_width[i]) else 0
        current_close = float(closes[i])
        current_upper = float(upper_band[i]) if not np.isnan(upper_band[i]) else 0
        current_lower = float(lower_band[i]) if not np.isnan(lower_band[i]) else 0
        current_middle = float(middle_band[i]) if not np.isnan(middle_band[i]) else 0
        
        # Squeeze intensity score (0-10)
        squeeze_score = 0
        
        # How narrow are the bands?
        if current_width < median_width * 0.3:
            squeeze_score += 3
        elif current_width < median_width * 0.5:
            squeeze_score += 2
        elif current_width < median_width * 0.7:
            squeeze_score += 1
        
        # Is price near lower band (short accumulation)?
        distance_to_lower = current_close - current_lower
        distance_range = current_upper - current_lower
        price_position = distance_to_lower / distance_range if distance_range > 0 else 0.5
        
        if price_position < 0.3:  # In lower 30% = short setup
            squeeze_score += 2
        elif price_position < 0.4:  # In lower 40%
            squeeze_score += 1
        
        # Is there a multi-period consolidation?
        if i >= 4:
            recent_range = max(closes[i-4:i+1]) - min(closes[i-4:i+1])
            if recent_range < (current_close * 0.02):  # Range < 2% consolidation
                squeeze_score += 2
        
        if squeeze_score >= 4:  # Only report significant squeezes
            squeeze_signals.append({
                "date": timestamps[i],
                "squeeze_score": squeeze_score,
                "band_width": current_width,
                "band_width_vs_median": f"{(current_width / median_width * 100):.1f}%",
                "price_position_in_bands": f"{(price_position * 100):.1f}%",
                "position_zone": "LOWER_ZONE (Short Setup)" if price_position < 0.4 else "MIDDLE_ZONE" if price_position < 0.6 else "UPPER_ZONE (Bullish Setup)",
                "distance_to_lower": distance_to_lower,
                "distance_to_upper": current_upper - current_close,
                "squeeze_intensity": "EXTREME" if squeeze_score >= 8 else "HIGH" if squeeze_score >= 6 else "MODERATE",
                "breakout_potential": "VERY_HIGH" if squeeze_score >= 8 else "HIGH" if squeeze_score >= 6 else "MODERATE",
            })
    
    return {
        "squeeze_detected": len(squeeze_signals) > 0,
        "squeeze_count": len(squeeze_signals),
        "current_squeeze_score": squeeze_signals[-1]["squeeze_score"] if squeeze_signals else 0,
        "squeeze_signals": squeeze_signals,
        "median_band_width": median_width,
        "band_width_std_dev": std_width,
    }