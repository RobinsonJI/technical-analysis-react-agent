from typing import List, Dict
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.data.enums import Interval, TimePeriod
from src.data.basemodels import TickData, Volume


@tool(args_schema=Volume)
def volume_analysis(
    stock_name: str,
    ticker_symbol: str,
    opening_price : List[float],
    closing_price: List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    period: int = 20
) -> Dict:
    """
    Analyze volume trends and volume-price confirmation.
    
    Volume analysis reveals the **strength** behind price moves:
    - High volume on up days: Strong bullish confirmation
    - High volume on down days: Strong bearish confirmation
    - Low volume on moves: Weak, likely to reverse
    - Volume spikes: Institutional buying/selling or breakouts
    
    **Key Metrics:**
    - **Volume MA**: Average volume over period (baseline)
    - **Volume Trend**: Positive on up days, negative on down days
    - **Price-Volume Ratio**: How much volume per unit price
    - **Volume Rate of Change**: Is volume expanding or contracting?
    
    **Signal Interpretation:**
    - Volume > MA on breakouts: Confirmation (strong move)
    - Volume < MA on retracements: Weak selling/buying (reversal likely)
    - Volume spikes 2x+ MA: Extreme interest, potential reversal
    - Divergence: Volume low but price rising = weak rally (bearish)
    
    Args:
        stock_name: Full name of the stock
        ticker_symbol: Stock ticker symbol
        closing_prices: List of closing prices for each period
        volumes: List of trading volumes for each period
        timestamps: List of ISO 8601 timestamp strings
        time_period: Time period of data
        interval: Data interval
        period: Volume moving average period (default 20, range: 5-50)
    
    Returns:
        Dictionary with:
            - volume_ma: Volume moving average
            - volume_trend: Signed volume (positive/negative)
            - price_volume_ratio: Price per unit of volume
            - volume_signals: Trading signals based on volume
            - volume_divergence: Price vs volume confirmation
            - metadata: Calculation info and statistics
    """

    logger.info(f"Running volume analysis for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    calculation_start = datetime.now()
    
    # Input validation
    if len(closing_price) != len(volume):
        raise ValueError("closing_prices and volumes must have same length")
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    if len(closing_price) < period:
        raise ValueError(f"Requires at least {period} periods, got {len(closing_price)}")
    
    closes = np.array(closing_price, dtype=float)
    vols = np.array(volume, dtype=float)
    
    # Volume moving average
    volume_ma = np.full_like(vols, np.nan)
    for i in range(period - 1, len(vols)):
        volume_ma[i] = np.mean(vols[i - period + 1:i + 1])
    
    # Price changes
    price_changes = np.diff(closes)
    
    # Volume trend (positive on up days, negative on down days)
    volume_trend = np.concatenate([[np.nan], np.where(price_changes > 0, vols[1:], -vols[1:])])
    
    # Price-volume ratio
    price_volume_ratio = np.divide(closes, vols, where=vols != 0, out=np.zeros_like(closes))
    
    # Volume Rate of Change (comparing current to MA)
    volume_roc = np.full_like(vols, np.nan)
    for i in range(period - 1, len(vols)):
        if volume_ma[i] > 0:
            volume_roc[i] = ((vols[i] - volume_ma[i]) / volume_ma[i]) * 100
    
    # Get latest values
    latest_volume = float(vols[-1])
    latest_ma = float(volume_ma[-1]) if not np.isnan(volume_ma[-1]) else None
    latest_roc = float(volume_roc[-1]) if not np.isnan(volume_roc[-1]) else None
    latest_close = float(closes[-1])
    latest_price_change = float(price_changes[-1]) if len(price_changes) > 0 else 0
    
    # Volume confirmation
    volume_vs_ma = "ABOVE_AVERAGE" if latest_volume > latest_ma else "BELOW_AVERAGE"
    
    # Detect volume spikes
    valid_ma = volume_ma[~np.isnan(volume_ma)]
    ma_mean = float(np.mean(valid_ma)) if len(valid_ma) > 0 else latest_ma
    ma_std = float(np.std(valid_ma)) if len(valid_ma) > 0 else 0
    
    if latest_volume > ma_mean + (2 * ma_std):
        volume_spike = "EXTREME (2+ std deviations)"
    elif latest_volume > ma_mean + ma_std:
        volume_spike = "HIGH (1+ std deviations)"
    elif latest_volume > ma_mean * 1.5:
        volume_spike = "ABOVE_AVERAGE (1.5x MA)"
    elif latest_volume < ma_mean * 0.5:
        volume_spike = "VERY_LOW (0.5x MA)"
    else:
        volume_spike = "NORMAL"
    
    # Volume-Price Divergence Detection
    divergence_signal = _detect_volume_divergence(closes, vols, volume_ma, period)
    
    # Volume signals
    signals = _generate_volume_signals(
        latest_volume, latest_ma, latest_roc, latest_price_change,
        volume_spike, divergence_signal, closes, vols
    )
    
    # Statistics
    valid_vols = vols[~np.isnan(vols)]
    valid_roc = volume_roc[~np.isnan(volume_roc)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000

    logger.info(f"Volume analysis for {stock_name} (${ticker_symbol}) complete")
    
    return {
        # Core values
        "volume_ma": volume_ma.tolist(),
        "volume_trend": volume_trend.tolist(),
        "price_volume_ratio": price_volume_ratio.tolist(),
        "volume_roc": volume_roc.tolist(),
        
        # Latest metrics
        "latest_volume": latest_volume,
        "latest_volume_ma": latest_ma,
        "latest_volume_roc": latest_roc,
        "volume_vs_ma": volume_vs_ma,
        "volume_spike_status": volume_spike,
        
        # Price-Volume Analysis
        "price_volume_analysis": {
            "latest_price_change": latest_price_change,
            "price_direction": "UP" if latest_price_change > 0 else "DOWN" if latest_price_change < 0 else "FLAT",
            "volume_confirmation": "CONFIRMED" if (latest_price_change > 0 and latest_volume > latest_ma) or (latest_price_change < 0 and latest_volume > latest_ma) else "WEAK",
        },
        
        # Divergence Detection
        "volume_divergence": divergence_signal,
        
        # Trading Signals
        "volume_signals": signals,
        
        # Statistics
        "volume_statistics": {
            "volume_min": float(np.nanmin(valid_vols)) if len(valid_vols) > 0 else None,
            "volume_max": float(np.nanmax(valid_vols)) if len(valid_vols) > 0 else None,
            "volume_mean": float(np.nanmean(valid_vols)) if len(valid_vols) > 0 else None,
            "volume_std": float(np.nanstd(valid_vols)) if len(valid_vols) > 0 else None,
        },
        
        "volume_ma_statistics": {
            "ma_min": float(np.nanmin(valid_ma)) if len(valid_ma) > 0 else None,
            "ma_max": float(np.nanmax(valid_ma)) if len(valid_ma) > 0 else None,
            "ma_mean": float(np.nanmean(valid_ma)) if len(valid_ma) > 0 else None,
        },
        
        "volume_roc_statistics": {
            "roc_min": float(np.nanmin(valid_roc)) if len(valid_roc) > 0 else None,
            "roc_max": float(np.nanmax(valid_roc)) if len(valid_roc) > 0 else None,
            "roc_mean": float(np.nanmean(valid_roc)) if len(valid_roc) > 0 else None,
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            "time_period": time_period.value,
            "interval": interval.value,
            
            "calculation_type": "Volume Analysis",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            "parameters": {
                "period": period,
            },
            
            "data_info": {
                "total_data_points": len(closes),
                "current_close": latest_close,
                "current_volume": latest_volume,
                "last_timestamp": timestamps[-1],
            },
            
            "note": "Volume confirms price trends. High volume = strong move. Low volume = weak move.",
        }
    }


def _detect_volume_divergence(closes: np.ndarray, volumes: np.ndarray, volume_ma: np.ndarray, lookback: int = 5) -> Dict:
    """
    Detect bullish and bearish volume divergences.
    
    Bullish Divergence: Price makes lower low, but volume makes higher low (reversal signal)
    Bearish Divergence: Price makes higher high, but volume makes lower high (reversal signal)
    """
    if len(closes) < lookback + 1:
        return {"divergence_detected": False, "type": "UNKNOWN"}
    
    recent_closes = closes[-lookback:]
    recent_volumes = volumes[-lookback:]
    recent_ma = volume_ma[-lookback:]
    
    price_lower_low = recent_closes[-1] < np.min(recent_closes[:-1])
    price_higher_high = recent_closes[-1] > np.max(recent_closes[:-1])
    
    volume_higher_low = recent_volumes[-1] > np.min(recent_volumes[:-1])
    volume_lower_high = recent_volumes[-1] < np.max(recent_volumes[:-1])
    
    if price_lower_low and volume_higher_low:
        return {
            "divergence_detected": True,
            "type": "BULLISH",
            "description": "Price made lower low but volume increased - reversal up likely",
            "strength": "HIGH"
        }
    elif price_higher_high and volume_lower_high:
        return {
            "divergence_detected": True,
            "type": "BEARISH",
            "description": "Price made higher high but volume decreased - reversal down likely",
            "strength": "HIGH"
        }
    else:
        return {
            "divergence_detected": False,
            "type": "NONE",
            "description": "Price and volume in agreement"
        }


def _generate_volume_signals(
    current_vol: float,
    volume_ma: float,
    volume_roc: float,
    price_change: float,
    spike_status: str,
    divergence: Dict,
    closes: np.ndarray,
    volumes: np.ndarray
) -> List[Dict]:
    """Generate trading signals from volume analysis."""
    signals = []
    
    # Signal 1: Volume Confirmation
    if price_change > 0 and current_vol > volume_ma:
        signals.append({
            "signal": "✅ BULLISH VOLUME CONFIRMATION",
            "confidence": "HIGH",
            "description": "Price up + Volume above MA = Strong bullish move",
            "action": "Continue holding or add to long positions",
            "strength": "STRONG"
        })
    elif price_change < 0 and current_vol > volume_ma:
        signals.append({
            "signal": "✅ BEARISH VOLUME CONFIRMATION",
            "confidence": "HIGH",
            "description": "Price down + Volume above MA = Strong bearish move",
            "action": "Consider selling or short positions",
            "strength": "STRONG"
        })
    
    # Signal 2: Weak Move
    if abs(price_change) > 0 and current_vol < volume_ma * 0.8:
        direction = "Up" if price_change > 0 else "Down"
        signals.append({
            "signal": "⚠️ WEAK MOVE - Low Volume",
            "confidence": "MODERATE",
            "description": f"Price {direction} but volume below MA - move lacks conviction",
            "action": "Expect reversal or consolidation",
            "strength": "WEAK"
        })
    
    # Signal 3: Volume Spike
    if "EXTREME" in spike_status or "HIGH" in spike_status:
        signals.append({
            "signal": "🔥 VOLUME SPIKE DETECTED",
            "confidence": "HIGH",
            "description": f"Volume spike: {spike_status}. Institutional activity or breakout",
            "action": "Watch for potential reversal or trend acceleration",
            "spike_level": spike_status
        })
    
    # Signal 4: Volume Divergence
    if divergence["divergence_detected"]:
        signals.append({
            "signal": f"🔄 {divergence['type']} DIVERGENCE DETECTED",
            "confidence": "HIGH",
            "description": divergence["description"],
            "action": f"Prepare for reversal in {divergence['type']} direction",
            "type": divergence["type"],
            "strength": divergence["strength"]
        })
    
    # Signal 5: Very Low Volume
    if "VERY_LOW" in spike_status:
        signals.append({
            "signal": "😴 VOLUME DRYING UP",
            "confidence": "MODERATE",
            "description": "Volume at 50% of MA - very light trading, expect breakout soon",
            "action": "Prepare for potential gap move or explosive breakout",
            "volatility_expectation": "LIKELY_TO_INCREASE"
        })
    
    # Signal 6: Volume Expansion
    if volume_roc and volume_roc > 50:
        signals.append({
            "signal": "📈 VOLUME EXPANDING FAST",
            "confidence": "MODERATE",
            "description": f"Volume up {volume_roc:.1f}% above MA - acceleration in progress",
            "action": "Trend is strengthening, momentum may continue",
            "roc": f"{volume_roc:.1f}%"
        })
    
    # Signal 7: Volume Contraction
    if volume_roc and volume_roc < -50:
        signals.append({
            "signal": "📉 VOLUME CONTRACTING FAST",
            "confidence": "MODERATE",
            "description": f"Volume down {abs(volume_roc):.1f}% below MA - losing momentum",
            "action": "Trend weakening, potential reversal coming",
            "roc": f"{volume_roc:.1f}%"
        })
    
    return signals


@tool(args_schema=TickData)
def obv(
    stock_name: str,
    ticker_symbol : str,
    opening_price : List[float],
    closing_price : List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
) -> Dict:
    """
    Calculate On-Balance Volume (OBV) indicator.
    
    OBV measures cumulative buying and selling pressure by adding volume on up days
    and subtracting volume on down days.
    
    **OBV Interpretation:**
    - OBV rising: Accumulation (institutional buying) = Bullish
    - OBV falling: Distribution (institutional selling) = Bearish
    - OBV flat: Indecision, consolidation
    - OBV vs price divergence: Early reversal signal
    
    **Key Concepts:**
    - OBV > long-term trend = Uptrend strength confirmed
    - OBV < long-term trend = Downtrend strength confirmed
    - OBV breaks to new high = Breakout confirmed
    - OBV flat while price rises = Weakening bullish pressure
    
    **Trading Signals:**
    - OBV crossover of MA: Trend change confirmation
    - Divergence: Price and OBV moving opposite = reversal likely
    - OBV spike: Institutional move, breakout confirmation
    
    Args:
        stock_name: Full name of the stock
        ticker_symbol: Stock ticker symbol
        closing_prices: List of closing prices
        volumes: List of trading volumes
        timestamps: List of ISO 8601 timestamp strings
        time_period: Time period of data
        interval: Data interval
    
    Returns:
        Dictionary with:
            - obv_values: Cumulative OBV
            - obv_ma: Moving average of OBV
            - obv_trend: Direction of OBV
            - obv_signals: Trading signals
            - metadata: Calculation info
    """
    logger.info(f"Computing OBV and running analysis for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    calculation_start = datetime.now()
    
    # Input validation
    if len(closing_price) != len(volume):
        raise ValueError("closing_prices and volumes must have same length")
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_prices and timestamps must have same length")
    
    closes = np.array(closing_price, dtype=float)
    vols = np.array(volume, dtype=float)
    
    # Calculate OBV
    obv_values = np.zeros_like(closes)
    obv_values[0] = vols[0]
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv_values[i] = obv_values[i-1] + vols[i]
        elif closes[i] < closes[i-1]:
            obv_values[i] = obv_values[i-1] - vols[i]
        else:
            obv_values[i] = obv_values[i-1]
    
    # OBV Moving Average (20-period)
    period = 20
    obv_ma = np.full_like(obv_values, np.nan)
    for i in range(period - 1, len(obv_values)):
        obv_ma[i] = np.mean(obv_values[i - period + 1:i + 1])
    
    # OBV Trend
    obv_trend_values = np.diff(obv_values)
    
    # Get latest values
    latest_obv = float(obv_values[-1])
    latest_obv_ma = float(obv_ma[-1]) if not np.isnan(obv_ma[-1]) else None
    latest_obv_trend = "UP" if obv_trend_values[-1] > 0 else "DOWN" if obv_trend_values[-1] < 0 else "FLAT"
    latest_close = float(closes[-1])
    latest_price_change = float(closes[-1] - closes[-2]) if len(closes) > 1 else 0
    
    # OBV vs MA confirmation
    obv_confirmation = "BULLISH" if latest_obv > latest_obv_ma else "BEARISH" if latest_obv < latest_obv_ma else "NEUTRAL"
    
    # Detect OBV divergence
    obv_divergence = _detect_obv_divergence(closes, obv_values)
    
    # Generate OBV signals
    obv_signals = _generate_obv_signals(latest_obv, latest_obv_ma, latest_obv_trend, obv_divergence, latest_price_change, closes, obv_values)
    
    # Statistics
    valid_obv = obv_values[~np.isnan(obv_values)]
    valid_obv_ma = obv_ma[~np.isnan(obv_ma)]
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    
    logger.info(f"OBV analysis for {stock_name} (${ticker_symbol}) complete")
    return {
        # Core values
        "obv_values": obv_values.tolist(),
        "obv_ma": obv_ma.tolist(),
        "obv_trend": obv_trend_values.tolist(),
        
        # Latest metrics
        "latest_obv": latest_obv,
        "latest_obv_ma": latest_obv_ma,
        "obv_direction": latest_obv_trend,
        "obv_vs_ma": obv_confirmation,
        
        # Price-OBV Analysis
        "price_obv_analysis": {
            "latest_price_change": latest_price_change,
            "price_direction": "UP" if latest_price_change > 0 else "DOWN" if latest_price_change < 0 else "FLAT",
            "obv_confirmation": "CONFIRMED" if (latest_price_change > 0 and latest_obv > latest_obv_ma) or (latest_price_change < 0 and latest_obv < latest_obv_ma) else "DIVERGENCE",
        },
        
        # Divergence Detection
        "obv_divergence": obv_divergence,
        
        # Trading Signals
        "obv_signals": obv_signals,
        
        # Statistics
        "obv_statistics": {
            "obv_min": float(np.nanmin(valid_obv)) if len(valid_obv) > 0 else None,
            "obv_max": float(np.nanmax(valid_obv)) if len(valid_obv) > 0 else None,
            "obv_mean": float(np.nanmean(valid_obv)) if len(valid_obv) > 0 else None,
            "obv_std": float(np.nanstd(valid_obv)) if len(valid_obv) > 0 else None,
        },
        
        "obv_ma_statistics": {
            "ma_min": float(np.nanmin(valid_obv_ma)) if len(valid_obv_ma) > 0 else None,
            "ma_max": float(np.nanmax(valid_obv_ma)) if len(valid_obv_ma) > 0 else None,
            "ma_mean": float(np.nanmean(valid_obv_ma)) if len(valid_obv_ma) > 0 else None,
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            "time_period": time_period.value,
            "interval": interval.value,
            
            "calculation_type": "On-Balance Volume (OBV)",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            "data_info": {
                "total_data_points": len(closes),
                "current_close": latest_close,
                "current_obv": latest_obv,
                "last_timestamp": timestamps[-1],
            },
            
            "note": "OBV confirms trends. Rising OBV = Bullish. Falling OBV = Bearish. Divergence = Reversal likely.",
        }
    }


def _detect_obv_divergence(closes: np.ndarray, obv_values: np.ndarray, lookback: int = 5) -> Dict:
    """
    Detect bullish and bearish OBV divergences.
    
    Bullish Divergence: Price lower low, OBV higher low (reversal up)
    Bearish Divergence: Price higher high, OBV lower high (reversal down)
    """
    if len(closes) < lookback + 1:
        return {"divergence_detected": False, "type": "UNKNOWN"}
    
    recent_closes = closes[-lookback:]
    recent_obv = obv_values[-lookback:]
    
    price_lower_low = recent_closes[-1] < np.min(recent_closes[:-1])
    price_higher_high = recent_closes[-1] > np.max(recent_closes[:-1])
    
    obv_higher_low = recent_obv[-1] > np.min(recent_obv[:-1])
    obv_lower_high = recent_obv[-1] < np.max(recent_obv[:-1])
    
    if price_lower_low and obv_higher_low:
        return {
            "divergence_detected": True,
            "type": "BULLISH",
            "description": "Price lower low but OBV higher low - accumulation, reversal up likely",
            "strength": "VERY_HIGH"
        }
    elif price_higher_high and obv_lower_high:
        return {
            "divergence_detected": True,
            "type": "BEARISH",
            "description": "Price higher high but OBV lower high - distribution, reversal down likely",
            "strength": "VERY_HIGH"
        }
    else:
        return {
            "divergence_detected": False,
            "type": "NONE",
            "description": "Price and OBV in agreement"
        }


def _generate_obv_signals(
    current_obv: float,
    obv_ma: float,
    obv_trend: str,
    divergence: Dict,
    price_change: float,
    closes: np.ndarray,
    obv_values: np.ndarray
) -> List[Dict]:
    """Generate trading signals from OBV analysis."""
    signals = []
    
    # Signal 1: OBV Confirmation
    if price_change > 0 and current_obv > obv_ma:
        signals.append({
            "signal": "✅ BULLISH OBV CONFIRMATION",
            "confidence": "HIGH",
            "description": "Price up + OBV above MA = Strong accumulation, bullish",
            "action": "Continue holding long positions or add to positions",
            "strength": "VERY_STRONG"
        })
    elif price_change < 0 and current_obv < obv_ma:
        signals.append({
            "signal": "✅ BEARISH OBV CONFIRMATION",
            "confidence": "HIGH",
            "description": "Price down + OBV below MA = Strong distribution, bearish",
            "action": "Consider selling or short positions",
            "strength": "VERY_STRONG"
        })
    
    # Signal 2: OBV Divergence
    if divergence["divergence_detected"]:
        signals.append({
            "signal": f"🔄 {divergence['type']} OBV DIVERGENCE",
            "confidence": "VERY_HIGH",
            "description": divergence["description"],
            "action": f"Early warning of reversal in {divergence['type']} direction",
            "type": divergence["type"],
            "strength": divergence["strength"]
        })
    
    # Signal 3: OBV Trend Up
    if obv_trend == "UP" and current_obv > obv_ma:
        signals.append({
            "signal": "📈 STRONG ACCUMULATION PHASE",
            "confidence": "HIGH",
            "description": "OBV rising above MA - institutions buying steadily",
            "action": "Bullish bias, look for breakout opportunities",
            "trend": "ACCUMULATING"
        })
    
    # Signal 4: OBV Trend Down
    if obv_trend == "DOWN" and current_obv < obv_ma:
        signals.append({
            "signal": "📉 STRONG DISTRIBUTION PHASE",
            "confidence": "HIGH",
            "description": "OBV falling below MA - institutions selling steadily",
            "action": "Bearish bias, watch for breakdown opportunities",
            "trend": "DISTRIBUTING"
        })
    
    # Signal 5: Weak Price Move
    if price_change > 0 and current_obv < obv_ma:
        signals.append({
            "signal": "⚠️ WEAK BULLISH MOVE",
            "confidence": "MODERATE",
            "description": "Price up but OBV below MA - rally lacks volume support",
            "action": "Expect pullback or reversal",
            "strength": "WEAK"
        })
    elif price_change < 0 and current_obv > obv_ma:
        signals.append({
            "signal": "⚠️ WEAK BEARISH MOVE",
            "confidence": "MODERATE",
            "description": "Price down but OBV above MA - selloff lacks conviction",
            "action": "Expect bounce or reversal",
            "strength": "WEAK"
        })
    
    return signals