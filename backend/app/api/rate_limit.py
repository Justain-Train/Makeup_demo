"""Per-client rate limiting dependency. Runs before upload reading and inference."""

import logging
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.services.rate_limit.protocol import RateLimiter

logger = logging.getLogger(__name__)

_DETAIL_LIMITED = "Too many requests. Please retry later."


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def client_key(request: Request) -> str:
    """Peer address only. X-Forwarded-For is client-controlled and is never read."""
    return request.client.host if request.client else "unknown"


def rate_limit(bucket: str) -> Callable[..., None]:
    def dependency(
        request: Request,
        limiter: RateLimiter = Depends(get_rate_limiter),
        settings: Settings = Depends(get_settings),
    ) -> None:
        if not settings.rate_limit_enabled:
            return
        decision = limiter.check(client_key(request), bucket)
        if not decision.allowed:
            logger.warning("Rate limit exceeded: bucket=%s", bucket)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                _DETAIL_LIMITED,
                headers={"Retry-After": str(decision.retry_after_s)},
            )

    return dependency
