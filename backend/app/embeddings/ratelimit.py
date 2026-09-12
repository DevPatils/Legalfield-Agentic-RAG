"""Sliding-window rate limiter for embedding APIs.

Voyage caps accounts without a payment method at 3 requests/min and 10K tokens/min.
Both ceilings bind independently, so the limiter tracks each in its own window and
sleeps until whichever is tighter has room.

A sliding window (rather than a fixed one) matters here: with only 3 RPM, a fixed
window lets 3 requests fire at 0:59 and 3 more at 1:01, which the server sees as 6
requests in 2 seconds and rejects.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


def estimate_tokens(text: str) -> int:
    """Rough token count. ~4 chars/token is close enough for budgeting, and erring
    high is safe -- it just means throttling slightly harder than required."""
    return max(1, len(text) // 4 + 1)


class RateLimiter:
    def __init__(self, max_requests: int, max_tokens: int) -> None:
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests: deque[float] = deque()
        self.tokens: deque[tuple[float, int]] = deque()

    def _evict(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        while self.tokens and self.tokens[0][0] <= cutoff:
            self.tokens.popleft()

    def _wait_for(self, token_cost: int, now: float) -> float:
        """Seconds to sleep before a request of ``token_cost`` may proceed."""
        self._evict(now)
        waits = [0.0]
        if len(self.requests) >= self.max_requests:
            waits.append(self.requests[0] + WINDOW_SECONDS - now)
        used = sum(count for _, count in self.tokens)
        if used + token_cost > self.max_tokens and self.tokens:
            # Evicting the oldest entries frees budget; wait for the oldest to age out.
            waits.append(self.tokens[0][0] + WINDOW_SECONDS - now)
        return max(waits)

    def acquire(self, token_cost: int, on_wait: Callable[[float], None] | None = None) -> None:
        """Block until a request costing ``token_cost`` fits inside both windows."""
        while True:
            now = time.monotonic()
            delay = self._wait_for(token_cost, now)
            if delay <= 0:
                break
            if on_wait:
                on_wait(delay)
            time.sleep(min(delay, WINDOW_SECONDS) + 0.25)
        now = time.monotonic()
        self.requests.append(now)
        self.tokens.append((now, token_cost))


def batch_by_tokens(costs: list[int], max_tokens: int, max_items: int) -> list[list[int]]:
    """Group indices into batches that fit the per-request token ceiling.

    Takes per-item token *costs* rather than the texts, so the caller decides how to
    count. Use the provider's real tokenizer where one exists: measured against
    ``voyage-law-2``, legal text runs 3.64 chars/token, so the chars/4 heuristic
    undershoots by ~10% and the resulting over-large requests are rejected
    server-side, costing a full backoff cycle each time.

    Returns index batches rather than items so callers can keep results aligned with
    their source chunks.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, cost in enumerate(costs):
        too_many = len(current) >= max_items
        too_big = current and current_tokens + cost > max_tokens
        if too_many or too_big:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches
