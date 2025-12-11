from typing import List, Dict
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from langchain.tools import tool

from src.data.basemodels import PivotPoints
from src.data.enums import Interval, TimePeriod


@tool(args_schema=PivotPoints)
def pivot_points(
    stock_name: str,
    ticker_symbol: str,
    opening_price: List[float],
    closing_price: List[float],
    volume: List[int],
    timestamps: List[str],
    time_period: TimePeriod,
    interval: Interval,
    high_prices: List[float] = None,
    low_prices: List[float] = None
) -> Dict:
    """
    Calculate Pivot Points for all periods with support and resistance levels.
    
    This function computes pivot points for EVERY period in the data,
    showing how support/resistance levels evolved over time.
    
    **Interpretation:**
    - Pivot Point (P): Center line - key reference level
    - Resistance 1 (R1): First resistance level (typical target)
    - Resistance 2 (R2): Second resistance level (strong resistance)
    - Support 1 (S1): First support level (typical bounce zone)
    - Support 2 (S2): Second support level (strong support)
    
    **Trading Strategy:**
    - Price bounces off S1/S2 = Buy signal
    - Price bounces off R1/R2 = Sell signal
    - Breakout above R2 = Strong bullish (continuation)
    - Breakdown below S2 = Strong bearish (continuation)
    
    Args:
        stock_name: Full name of the stock
        ticker_symbol: Stock ticker symbol
        opening_price: List of opening prices (required)
        closing_price: List of closing prices (required)
        volume: List of volumes
        timestamps: List of ISO 8601 timestamps
        time_period: Time period of data
        interval: Data interval
        high_prices: List of high prices (optional - will use max(open, close) if not provided)
        low_prices: List of low prices (optional - will use min(open, close) if not provided)
    
    Returns:
        Dictionary with:
            - all_pivots: Complete list of pivot points for all periods
            - current_pivot_levels: Latest pivot levels (for today's trading)
            - pivot_summary: Statistics and counts
            - Metadata
    """
    logger.info(f"Computing Pivot Points for {stock_name} (${ticker_symbol})")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Interval: {interval}")

    calculation_start = datetime.now()
    
    # Input validation
    if len(closing_price) < 2:
        raise ValueError("Requires at least 2 data points")
    
    if len(closing_price) != len(timestamps):
        raise ValueError("closing_price and timestamps must have same length")
    
    if len(opening_price) != len(closing_price):
        raise ValueError("opening_price and closing_price must have same length")
    
    closes = np.array(closing_price, dtype=float)
    opens = np.array(opening_price, dtype=float)
    
    # Derive high/low from open/close if not provided
    if high_prices is None:
        high_prices = np.maximum(opens, closes)  # High = max(open, close)
    else:
        high_prices = np.array(high_prices, dtype=float)
    
    if low_prices is None:
        low_prices = np.minimum(opens, closes)   # Low = min(open, close)
    else:
        low_prices = np.array(low_prices, dtype=float)
    
    # Validate arrays
    if len(high_prices) != len(closes) or len(low_prices) != len(closes):
        raise ValueError("high_prices, low_prices, and closing_price must have same length")
    
    # Calculate pivot points for ALL periods
    all_pivots = []
    
    for i in range(len(closes) - 1):  # -1 because we need previous period to calculate
        high_prev = float(high_prices[i])
        low_prev = float(low_prices[i])
        close_prev = float(closes[i])
        close_current = float(closes[i + 1])
        
        # Calculate pivot levels from previous period
        pivot = (high_prev + low_prev + close_prev) / 3
        resistance1 = (2 * pivot) - low_prev
        support1 = (2 * pivot) - high_prev
        resistance2 = pivot + (high_prev - low_prev)
        support2 = pivot - (high_prev - low_prev)
        range_size = high_prev - low_prev
        
        # Calculate distances for next period
        distance_to_pivot = close_current - pivot
        distance_to_r1 = resistance1 - close_current
        distance_to_r2 = resistance2 - close_current
        distance_to_s1 = close_current - support1
        distance_to_s2 = close_current - support2
        
        # Determine position
        if close_current > resistance2:
            position = "ABOVE_R2"
        elif close_current > resistance1:
            position = "BETWEEN_R1_AND_R2"
        elif close_current > pivot:
            position = "BETWEEN_PIVOT_AND_R1"
        elif close_current > support1:
            position = "BETWEEN_S1_AND_PIVOT"
        elif close_current > support2:
            position = "BETWEEN_S2_AND_S1"
        else:
            position = "BELOW_S2"
        
        all_pivots.append({
            "period_index": i,
            "date": timestamps[i],
            "next_date": timestamps[i + 1],  # Date when these pivots are used
            "calculated_from": {
                "high": high_prev,
                "low": low_prev,
                "close": close_prev,
                "range": range_size,
            },
            "pivot_levels": {
                "pivot": pivot,
                "resistance_1": resistance1,
                "resistance_2": resistance2,
                "support_1": support1,
                "support_2": support2,
            },
            "next_period_price": close_current,
            "distances": {
                "to_pivot": distance_to_pivot,
                "to_r1": distance_to_r1,
                "to_r2": distance_to_r2,
                "to_s1": distance_to_s1,
                "to_s2": distance_to_s2,
            },
            "position": position,
            "strength": {
                "r1": "STRONG" if distance_to_r1 < range_size else "WEAK",
                "r2": "STRONG" if distance_to_r2 < range_size * 2 else "WEAK",
                "s1": "STRONG" if distance_to_s1 < range_size else "WEAK",
                "s2": "STRONG" if distance_to_s2 < range_size * 2 else "WEAK",
            }
        })
    
    # Get current/latest pivot (for today's trading)
    if len(closes) >= 2:
        high_prev = float(high_prices[-2])
        low_prev = float(low_prices[-2])
        close_prev = float(closes[-2])
    else:
        high_prev = float(high_prices[-1])
        low_prev = float(low_prices[-1])
        close_prev = float(closes[-1])
    
    close_current = float(closes[-1])
    
    pivot = (high_prev + low_prev + close_prev) / 3
    resistance1 = (2 * pivot) - low_prev
    support1 = (2 * pivot) - high_prev
    resistance2 = pivot + (high_prev - low_prev)
    support2 = pivot - (high_prev - low_prev)
    range_size = high_prev - low_prev
    
    distance_to_pivot = close_current - pivot
    distance_to_r1 = resistance1 - close_current
    distance_to_r2 = resistance2 - close_current
    distance_to_s1 = close_current - support1
    distance_to_s2 = close_current - support2
    
    # Determine current position
    if close_current > resistance2:
        position = "ABOVE_R2 (Strong Bullish)"
    elif close_current > resistance1:
        position = "BETWEEN_R1_AND_R2"
    elif close_current > pivot:
        position = "BETWEEN_PIVOT_AND_R1"
    elif close_current > support1:
        position = "BETWEEN_S1_AND_PIVOT"
    elif close_current > support2:
        position = "BETWEEN_S2_AND_S1"
    else:
        position = "BELOW_S2 (Strong Bearish)"
    
    # Calculate pivot statistics
    all_pivots_list = [p["pivot_levels"]["pivot"] for p in all_pivots]
    all_r1 = [p["pivot_levels"]["resistance_1"] for p in all_pivots]
    all_s1 = [p["pivot_levels"]["support_1"] for p in all_pivots]

    recent_pivots = _get_recent_pivots(all_pivots, num_periods=10)
    important_pivots = _get_important_pivots(all_pivots, threshold=1.5)
    pivot_levels_summary = _get_pivot_levels_by_type(all_pivots)
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000
    
 
    
    calculation_end = datetime.now()
    calculation_duration = (calculation_end - calculation_start).total_seconds() * 1000

    logger.info(f"Finished computing Pivot Points for {stock_name} (${ticker_symbol})")
    return {
        # All historical pivots
         # Recent pivots (last 10 periods)
        "recent_pivots": recent_pivots,
        
        # Important/high-impact pivots
        "important_pivots": important_pivots,
        
        # Pivot level summary by type
        "pivot_levels_summary": pivot_levels_summary,
        

        # Current pivot levels (latest - for today's trading)
        "current_pivot_levels": {
            "period_index": len(closes) - 1,
            "date_calculated": timestamps[-2] if len(timestamps) >= 2 else timestamps[-1],
            "valid_for_date": timestamps[-1],
            "pivot_point": float(pivot),
            "resistance_1": float(resistance1),
            "resistance_2": float(resistance2),
            "support_1": float(support1),
            "support_2": float(support2),
        },
        
        # Previous period data used for current calculation
        "previous_period_data": {
            "high": float(high_prev),
            "low": float(low_prev),
            "close": float(close_prev),
            "range": float(range_size),
        },
        
        # Current price analysis
        "current_price_analysis": {
            "current_price": float(close_current),
            "current_position": position,
            "daily_range": float(high_prices[-1] - low_prices[-1]),
            "range_percentage": float((high_prices[-1] - low_prices[-1]) / low_prices[-1] * 100) if low_prices[-1] > 0 else 0,
        },
        
        # Distance metrics for current period
        "distance_to_levels": {
            "to_pivot": float(distance_to_pivot),
            "to_resistance_1": float(distance_to_r1),
            "to_resistance_2": float(distance_to_r2),
            "to_support_1": float(distance_to_s1),
            "to_support_2": float(distance_to_s2),
        },
        
        # Strength assessment
        "level_strength": {
            "resistance_1_strength": "STRONG" if distance_to_r1 < range_size else "WEAK",
            "resistance_2_strength": "STRONG" if distance_to_r2 < range_size * 2 else "WEAK",
            "support_1_strength": "STRONG" if distance_to_s1 < range_size else "WEAK",
            "support_2_strength": "STRONG" if distance_to_s2 < range_size * 2 else "WEAK",
        },
        
        # Trading signals
        "trading_signals": {
            "nearest_support": float(support1) if distance_to_s1 < distance_to_s2 else float(support2),
            "nearest_resistance": float(resistance1) if distance_to_r1 < distance_to_r2 else float(resistance2),
            "breakout_potential": "UPSIDE" if close_current > pivot else "DOWNSIDE",
            "bounce_zone": f"S1 ({support1:.2f})" if close_current > support1 else f"S2 ({support2:.2f})",
        },
        
        # Pivot statistics and summary
        "pivot_summary": {
            "total_pivots_calculated": len(all_pivots),
            "data_points": len(closes),
            "pivot_min": float(np.nanmin(all_pivots_list)) if all_pivots_list else None,
            "pivot_max": float(np.nanmax(all_pivots_list)) if all_pivots_list else None,
            "pivot_mean": float(np.nanmean(all_pivots_list)) if all_pivots_list else None,
            "pivot_std": float(np.nanstd(all_pivots_list)) if all_pivots_list else None,
            "r1_min": float(np.nanmin(all_r1)) if all_r1 else None,
            "r1_max": float(np.nanmax(all_r1)) if all_r1 else None,
            "s1_min": float(np.nanmin(all_s1)) if all_s1 else None,
            "s1_max": float(np.nanmax(all_s1)) if all_s1 else None,
        },
        
        # Metadata
        "metadata": {
            "stock_name": stock_name,
            "ticker_symbol": ticker_symbol,
            "time_period": time_period.value,
            "interval": interval.value,
            
            "calculation_type": "Pivot Points (All Periods)",
            "calculation_timestamp": calculation_start.isoformat(),
            "calculation_completed_at": calculation_end.isoformat(),
            "calculation_duration_ms": round(calculation_duration, 2),
            
            "current_high": float(high_prices[-1]),
            "current_low": float(low_prices[-1]),
            "current_close": float(close_current),
            "last_timestamp": timestamps[-1],
            "note": "Pivot levels calculated from PREVIOUS period (standard practice)",
        }
    }


# Helpers for pivots

def _get_recent_pivots(all_pivots: List[Dict], num_periods: int = 10) -> List[Dict]:
    """Extract the most recent N pivot points."""
    return all_pivots[-num_periods:] if all_pivots else []


def _get_important_pivots(all_pivots: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    Extract 'important' pivots based on:
    - Large price ranges (volatile periods)
    - Significant distance from current price
    - Strong support/resistance levels
    """
    if not all_pivots:
        return []
    
    important = []
    current_price = all_pivots[-1]["next_period_price"]
    
    for pivot_data in all_pivots:
        range_size = pivot_data["calculated_from"]["range"]
        distances = pivot_data["distances"]
        
        # Calculate importance score
        score = 0
        
        # High volatility = important
        if range_size > np.nanmean([p["calculated_from"]["range"] for p in all_pivots]) * 1.5:
            score += 2
        
        # Price recently tested this level = important
        if abs(distances["to_pivot"]) < range_size * 0.5:
            score += 2
        
        # Multiple touches/bounces = important (simplified)
        if abs(distances["to_r1"]) < range_size * 0.3:
            score += 1.5
        if abs(distances["to_s1"]) < range_size * 0.3:
            score += 1.5
        
        if score >= threshold:
            important.append({
                **pivot_data,
                "importance_score": score
            })
    
    return sorted(important, key=lambda x: x["importance_score"], reverse=True)


def _get_pivot_levels_by_type(all_pivots: List[Dict]) -> Dict:
    """
    Extract and summarize each pivot level type across all periods.
    Shows historical ranges of S1, S2, R1, R2, P.
    """
    if not all_pivots:
        return {}
    
    pivots = [p["pivot_levels"]["pivot"] for p in all_pivots]
    r1_levels = [p["pivot_levels"]["resistance_1"] for p in all_pivots]
    r2_levels = [p["pivot_levels"]["resistance_2"] for p in all_pivots]
    s1_levels = [p["pivot_levels"]["support_1"] for p in all_pivots]
    s2_levels = [p["pivot_levels"]["support_2"] for p in all_pivots]
    
    return {
        "pivot_point": {
            "current": pivots[-1],
            "min": float(np.min(pivots)),
            "max": float(np.max(pivots)),
            "mean": float(np.mean(pivots)),
            "std": float(np.std(pivots)),
            "range": float(np.max(pivots) - np.min(pivots)),
        },
        "resistance_1": {
            "current": r1_levels[-1],
            "min": float(np.min(r1_levels)),
            "max": float(np.max(r1_levels)),
            "mean": float(np.mean(r1_levels)),
            "std": float(np.std(r1_levels)),
            "range": float(np.max(r1_levels) - np.min(r1_levels)),
        },
        "resistance_2": {
            "current": r2_levels[-1],
            "min": float(np.min(r2_levels)),
            "max": float(np.max(r2_levels)),
            "mean": float(np.mean(r2_levels)),
            "std": float(np.std(r2_levels)),
            "range": float(np.max(r2_levels) - np.min(r2_levels)),
        },
        "support_1": {
            "current": s1_levels[-1],
            "min": float(np.min(s1_levels)),
            "max": float(np.max(s1_levels)),
            "mean": float(np.mean(s1_levels)),
            "std": float(np.std(s1_levels)),
            "range": float(np.max(s1_levels) - np.min(s1_levels)),
        },
        "support_2": {
            "current": s2_levels[-1],
            "min": float(np.min(s2_levels)),
            "max": float(np.max(s2_levels)),
            "mean": float(np.mean(s2_levels)),
            "std": float(np.std(s2_levels)),
            "range": float(np.max(s2_levels) - np.min(s2_levels)),
        },
    }