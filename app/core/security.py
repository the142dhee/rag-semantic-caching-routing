"""
app/core/security.py
====================
PURPOSE: Security controls for API Key Authentication, Request Throttling (Rate Limiting),
and input sanitization.
"""

import time
import threading
import logging
from typing import Dict, List, Optional
from fastapi import Request, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> bool:
    """
    Validates the client's X-API-Key header if API authentication is enabled in settings.

    Returns:
        True if authentication succeeds or is disabled.

    Raises:
        HTTPException 401: If API key is missing or invalid.
    """
    settings = get_settings()

    # Bypassed if disabled in settings (e.g. during local dev or CI tests)
    if not settings.api_key_auth_enabled:
        return True

    if not api_key or api_key.strip() != settings.api_auth_key:
        logger.warning(f"Unauthorized API access attempt with key: '{api_key}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


class InMemoryRateLimiter:
    """
    Thread-safe sliding-window rate limiter per client IP.
    Ensures endpoints are protected against denial-of-service and brute-force attacks.
    """

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str, limit: int = 60, window_seconds: float = 60.0) -> bool:
        """
        Check if the request from client_ip is within the rate limit.

        Args:
            client_ip: Remote IP address of the caller.
            limit: Maximum allowed requests within the window.
            window_seconds: Window duration in seconds (default: 60).

        Returns:
            True if allowed, False if limit exceeded.
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            # Clean expired timestamps for this IP
            timestamps = self._requests.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if t > cutoff]

            if len(valid_timestamps) >= limit:
                self._requests[client_ip] = valid_timestamps
                return False

            valid_timestamps.append(now)
            self._requests[client_ip] = valid_timestamps
            return True

    def reset(self):
        """Clear all rate limit tracking records."""
        with self._lock:
            self._requests.clear()


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware to throttle requests based on client IP.
    Exempts documentation and health checks from throttling.
    """
    settings = get_settings()

    if not settings.rate_limit_enabled:
        return await call_next(request)

    path = request.url.path
    # Exempt health checks and documentation from rate limiting
    if path in ["/", "/health", "/ready", "/api/v1/health", "/api/v1/ready", "/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    limit = settings.rate_limit_requests_per_minute

    if not rate_limiter.is_allowed(client_ip, limit=limit, window_seconds=60.0):
        logger.warning(f"Rate limit exceeded for client IP: {client_ip} on {path}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Rate limit exceeded: maximum {limit} requests per minute allowed."},
            headers={"Retry-After": "60"},
        )

    return await call_next(request)
