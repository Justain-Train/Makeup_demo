"""Rate limiter contract. Implementations must be thread-safe."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Decision:
    allowed: bool
    # Whole seconds to wait: >= 1 when denied, 0 when allowed.
    retry_after_s: int


class RateLimiter(Protocol):
    def check(self, key: str, bucket: str) -> Decision: ...
