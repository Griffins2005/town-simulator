"""
rate_limiter.py -- A real wall-clock rate limiter for free-tier LLM calls.

Why this exists at all: Groq's free tier caps llama-3.3-70b-versatile at
roughly 30 requests/minute, shared across the WHOLE organization (i.e.
the whole town, not per-agent) -- confirmed via Groq's own rate-limit
docs and multiple independent trackers as of June 2026. That is 1 request
every 2 seconds, total, for every agent combined. Phase 1's engine was
tickless-and-instant by design (no wall-clock pacing at all); an LLM
Decider makes wall-clock pacing load-bearing for the first time in this
codebase. Exceeding the limit doesn't degrade gracefully -- Groq returns
HTTP 429 and your request is simply rejected, not queued by them. So
*something* in this codebase has to be the thing standing between 16
agents trying to think at once and a wall of 429s -- this module is it.

Design choice: a simple token-bucket, not a sliding window. A token
bucket is simpler to reason about and verify, and slightly more
permissive at the margins (bursts up to the bucket size) than a strict
sliding window would be -- acceptable here since Groq's own limit is
already a safety margin, not a hard physical constraint we're trying to
exactly saturate.
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe-ish token bucket (this codebase is single-threaded by
    construction -- see engine.py's synchronous step() loop -- so the
    locking here is defensive, not load-bearing, but costs nothing and
    means this class is also safe to reuse if a future version DOES
    parallelize agent decisions).

    Usage: call `wait_for_token()` immediately before making an LLM call.
    It blocks (sleeps) until a token is available, then consumes one.
    This makes the caller's wall-clock pacing automatic and impossible
    to forget -- there's no separate "check if allowed" step to skip.
    """

    def __init__(self, max_per_minute, burst=None):
        """
        Args:
            max_per_minute: steady-state requests/minute allowed. Set
                this BELOW Groq's published limit, not equal to it --
                see RECOMMENDED_RPM_SAFETY_MARGIN below for why.
            burst: bucket capacity (how many requests can fire back-to-
                back before throttling kicks in). Defaults to a small
                burst (max_per_minute // 6, i.e. ~10 seconds' worth) --
                large enough that a brief flurry of agent interrupts
                doesn't stall immediately, small enough that it can't
                blow through a big chunk of the per-minute budget in one
                burst.
        """
        self.rate_per_second = max_per_minute / 60.0
        self.capacity = burst if burst is not None else max(1, max_per_minute // 6)
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def wait_for_token(self):
        """Block until a token is available, consume it, return how long
        we waited (seconds) -- the caller can log/print this so the
        wall-clock cost of LLM-backed ticks is visible, not silent.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                # Not enough tokens yet -- compute exactly how long until
                # there will be one, sleep that long (outside the lock),
                # then loop and re-check rather than assuming the sleep
                # was exact (clock granularity, scheduling jitter).
                deficit = 1.0 - self._tokens
                sleep_for = deficit / self.rate_per_second
            time.sleep(min(sleep_for, 1.0))  # cap each individual sleep
            # so the loop can re-check rather than oversleeping past a
            # token becoming available from a concurrent caller (again,
            # defensive given this codebase is single-threaded today).
            waited += min(sleep_for, 1.0)


# Set BELOW Groq's published 30 RPM, not equal to it. Reasons: (1) the
# limit is shared at the organization level, so anything else hitting
# the same API key (e.g. you testing manually in another terminal while
# the sim runs) eats into the same budget; (2) clock drift / network
# latency means a limiter ticking at exactly 30/min can still occasionally
# present 31 requests inside Groq's actual 60-second window; (3) the
# cost of being slightly too conservative (a few extra seconds of wall
# clock per tick) is far lower than the cost of 429s interrupting a long
# unattended run. 24 leaves real headroom while still using most of the
# free tier's budget.
RECOMMENDED_RPM_SAFETY_MARGIN = 24
