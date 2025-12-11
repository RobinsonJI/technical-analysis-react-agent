from datetime import datetime
from typing import List


def get_date_from_index(timestamps: List[str], index: int) -> str:
    """
    Get date/timestamp from a list of timestamps at the given index.
    
    Args:
        timestamps: List of timestamp strings (ISO 8601 format)
        index: Index position in the timestamps list
    
    Returns:
        Timestamp string at the given index, or "Index {index}" if out of bounds
    
    Example:
        >>> timestamps = ["2025-01-01T10:30:00", "2025-01-02T10:30:00"]
        >>> get_date_from_index(timestamps, 0)
        '2025-01-01T10:30:00'
        >>> get_date_from_index(timestamps, 10)
        'Index 10'
    """
    if timestamps and index < len(timestamps):
        return str(timestamps[index])
    
    return f"Index {index}"



def get_current_date() -> str:
    """
    Helper function to get the current/latest date.
    """    
    return datetime.now().isoformat()