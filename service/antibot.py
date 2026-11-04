import time
from collections import OrderedDict
from fastapi import Request, HTTPException
from typing import Optional


def get_client_ip(request: Request) -> str:
    """Extract real client IP address, supporting reverse proxies (X-Forwarded-For, X-Real-IP)."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For may contain a comma-separated list of client, proxy1, proxy2...
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


class BoundedSlidingWindowLimiter:
    """Sliding window rate limiter with LRU capacity bounding to prevent memory leaks."""

    def __init__(self, max_keys: int = 10000):
        self.max_keys = max_keys
        self._records: OrderedDict[str, list[float]] = OrderedDict()

    def is_allowed(self, key: str, times: int, seconds: int) -> bool:
        current_time = time.time()
        cutoff_time = current_time - seconds

        if key in self._records:
            # Move to end for LRU tracking
            self._records.move_to_end(key)
            timestamps = [t for t in self._records[key] if t > cutoff_time]
        else:
            timestamps = []

        if len(timestamps) >= times:
            self._records[key] = timestamps
            return False

        timestamps.append(current_time)
        self._records[key] = timestamps

        # Enforce LRU capacity bound
        while len(self._records) > self.max_keys:
            self._records.popitem(last=False)

        return True

    def clear_key(self, key: str) -> None:
        self._records.pop(key, None)


class RateLimiter:
    """Configurable FastAPI rate limiting dependency."""

    def __init__(
        self,
        times: int = 3,
        seconds: int = 30,
        scope: str = "default",
        detail: Optional[str] = None,
        max_keys: int = 10000,
    ):
        self.times = times
        self.seconds = seconds
        self.scope = scope
        self.detail = detail or "Request rate limit exceeded. Please try again later."
        self._limiter = BoundedSlidingWindowLimiter(max_keys=max_keys)

    async def __call__(self, request: Request) -> bool:
        client_ip = get_client_ip(request)
        key = f"{self.scope}:{client_ip}"

        if not self._limiter.is_allowed(key, self.times, self.seconds):
            raise HTTPException(status_code=429, detail=self.detail)

        return True


# Default instance preserving backwards compatibility
default_rate_limiter = RateLimiter(times=3, seconds=30, scope="global")


async def check_rate_limit(request: Request) -> bool:
    """Backward-compatible rate limit check (3 requests / 30 seconds)."""
    return await default_rate_limiter(request)


# Specialized rate limiter for verification code attempts (10 requests / 60 seconds per IP)
verify_rate_limiter = RateLimiter(
    times=10,
    seconds=60,
    scope="verify_ip",
    detail="Too many verification attempts from this IP. Please try again later.",
)


# --- Email-Level Brute Force Protection ---
class EmailAttemptTracker:
    """Track failed verification attempts per email to prevent distributed credential stuffing."""

    def __init__(
        self,
        max_attempts: int = 5,
        lock_seconds: int = 600,
        window_seconds: int = 600,
        max_keys: int = 10000,
    ):
        self.max_attempts = max_attempts
        self.lock_seconds = lock_seconds
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        # key -> {"attempts": [t1, t2], "locked_until": float}
        self._data: OrderedDict[str, dict] = OrderedDict()

    def _get_entry(self, email: str) -> dict:
        key = email.strip().lower()
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]

        entry = {"attempts": [], "locked_until": 0.0}
        self._data[key] = entry

        while len(self._data) > self.max_keys:
            self._data.popitem(last=False)

        return entry

    def is_locked(self, email: str) -> bool:
        key = email.strip().lower()
        if key not in self._data:
            return False

        entry = self._data[key]
        now = time.time()
        if now < entry.get("locked_until", 0.0):
            return True

        return False

    def record_failure(self, email: str) -> int:
        entry = self._get_entry(email)
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = [t for t in entry["attempts"] if t > cutoff]
        timestamps.append(now)
        entry["attempts"] = timestamps

        if len(timestamps) >= self.max_attempts:
            entry["locked_until"] = now + self.lock_seconds

        return len(timestamps)

    def clear(self, email: str) -> None:
        key = email.strip().lower()
        self._data.pop(key, None)


_email_tracker = EmailAttemptTracker()


def is_verify_locked(email: str) -> bool:
    """Check if an email is locked due to too many failed verification attempts."""
    return _email_tracker.is_locked(email)


def record_failed_verify(email: str) -> int:
    """Record a failed verification attempt for the given email."""
    return _email_tracker.record_failure(email)


def clear_failed_verify(email: str) -> None:
    """Clear failed verification attempts upon successful verification."""
    _email_tracker.clear(email)


# Honeypot verification
def check_honeypot(honeypot_value: str):
    if honeypot_value:
        raise HTTPException(status_code=400, detail="Bot behavior detected.")
