from enum import Enum


class TimePeriod(Enum):
    """Valid time periods for yfinance data fetching"""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YEAR_TO_DATE = "ytd"
    MAX = "max"

class Interval(Enum):
    """Valid intervals for yfinance data fetching"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "60m"
    ONE_HOUR_ALT = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"

class FibonacciExtensionPattern(str, Enum):
    """Fibonacci pattern types."""
    BULLISH = "low_high_low"
    BEARISH = "high_low_high"
    
    def __str__(self) -> str:
        """Return human-readable name."""
        if self == FibonacciExtensionPattern.BULLISH:
            return "Low-High-Low (Bullish)"
        return "High-Low-High (Bearish)"