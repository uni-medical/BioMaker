import time

class Timer:
    def __init__(self, timeout_seconds=None):
        self.timeout_seconds = timeout_seconds
        self._start_time = time.monotonic() # Use monotonic clock for durations

    @property
    def elapsed(self) -> float:
        """Returns the elapsed time in seconds."""
        return time.monotonic() - self._start_time

    def reset(self):
        """Resets the timer to zero."""
        self._start_time = time.monotonic()

    def is_timeout(self, tolerance=0) -> bool:
        """Checks if the timer has expired based on the time limit."""
        if self.timeout_seconds is None:
            return False
        return self.elapsed >= self.timeout_seconds + tolerance

            

if __name__ == "__main__":
    timer = Timer(4)
    print(timer.elapsed)
    print(timer.is_timeout())
    time.sleep(5)
    print(timer.elapsed)
    print(timer.is_timeout())