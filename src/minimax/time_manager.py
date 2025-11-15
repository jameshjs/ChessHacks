"""
Time management for chess engine.
"""

import time


class TimeManager:
    """
    Manages time allocation for move search.
    """
    
    def __init__(self, time_left_ms: int, safety_margin: float = 0.1):
        """
        Args:
            time_left_ms: Time remaining in milliseconds
            safety_margin: Fraction of time to reserve (default 0.1 = 10%)
        """
        self.time_left_ms = time_left_ms
        self.safety_margin = safety_margin
        self.start_time = time.time()
        
        # Allocate time for this move (use 10% of remaining time, with safety margin)
        self.allocated_time = (time_left_ms / 1000.0) * 0.1 * (1 - safety_margin)
        self.end_time = self.start_time + self.allocated_time
    
    def should_stop(self) -> bool:
        """Check if search should stop due to time limit."""
        return time.time() >= self.end_time
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.end_time - time.time())
    
    def allocate_time(self) -> float:
        """Get allocated time for this move in seconds."""
        return self.allocated_time

