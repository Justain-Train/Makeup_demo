"""In-process token-bucket limiter with a bounded key table (LRU eviction)."""

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from app.services.rate_limit.protocol import Decision


class InMemoryRateLimiter:
    def __init__(
        self,
        rate_per_min: float,
        burst: int,
        max_keys: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._rate = float(rate_per_min)
        self._burst = burst
        self._max_keys = max_keys
        self._clock = clock
        self._lock = threading.Lock()
        self._state: OrderedDict[tuple[str, str], tuple[float, float]] = OrderedDict()

    def check(self, key: str, bucket: str) -> Decision:
        entry = (bucket, key)
        with self._lock:
            now = self._clock()
            if entry in self._state:
                tokens, last = self._state[entry]
                tokens = min(float(self._burst), tokens + max(0.0, now - last) * self._rate / 60)
            else:
                tokens = float(self._burst)
            if tokens >= 1:
                self._state[entry] = (tokens - 1, now)
                decision = Decision(True, 0)
            else:
                self._state[entry] = (tokens, now)
                decision = Decision(False, max(1, math.ceil((1 - tokens) * 60 / self._rate)))
            self._state.move_to_end(entry)
            while len(self._state) > self._max_keys:
                self._state.popitem(last=False)
            return decision
