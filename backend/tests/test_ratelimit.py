import time

from app.embeddings.ratelimit import RateLimiter, batch_by_tokens, estimate_tokens


class TestEstimateTokens:
    def test_never_returns_zero(self):
        assert estimate_tokens("") >= 1

    def test_scales_with_length(self):
        assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)


class TestBatchByTokens:
    def test_splits_on_the_token_ceiling(self):
        batches = batch_by_tokens([1000] * 4, max_tokens=2500, max_items=100)
        assert len(batches) == 2
        assert [len(b) for b in batches] == [2, 2]

    def test_splits_on_the_item_ceiling(self):
        batches = batch_by_tokens([1] * 10, max_tokens=10**9, max_items=4)
        assert [len(b) for b in batches] == [4, 4, 2]

    def test_every_index_appears_exactly_once_in_order(self):
        costs = [25 * i for i in range(1, 40)]
        batches = batch_by_tokens(costs, max_tokens=600, max_items=5)
        assert [i for b in batches for i in b] == list(range(len(costs)))

    def test_oversized_item_still_gets_its_own_batch(self):
        # A single chunk larger than the ceiling must not be dropped.
        batches = batch_by_tokens([20_000, 1], max_tokens=1000, max_items=10)
        assert [i for b in batches for i in b] == [0, 1]

    def test_empty_input(self):
        assert batch_by_tokens([], max_tokens=100, max_items=10) == []


class TestRateLimiter:
    def test_allows_requests_under_both_ceilings(self):
        limiter = RateLimiter(max_requests=5, max_tokens=10_000)
        started = time.monotonic()
        for _ in range(5):
            limiter.acquire(100)
        assert time.monotonic() - started < 0.5

    def test_request_ceiling_would_block(self):
        limiter = RateLimiter(max_requests=2, max_tokens=10**9)
        limiter.acquire(1)
        limiter.acquire(1)
        # A third request must wait for the oldest to age out of the window.
        assert limiter._wait_for(1, time.monotonic()) > 0

    def test_token_ceiling_would_block(self):
        limiter = RateLimiter(max_requests=100, max_tokens=1000)
        limiter.acquire(900)
        assert limiter._wait_for(500, time.monotonic()) > 0

    def test_token_ceiling_allows_a_fitting_request(self):
        limiter = RateLimiter(max_requests=100, max_tokens=1000)
        limiter.acquire(400)
        assert limiter._wait_for(400, time.monotonic()) == 0

    def test_window_eviction_frees_budget(self):
        limiter = RateLimiter(max_requests=1, max_tokens=100)
        limiter.acquire(100)
        # Backdate the recorded entries so they fall outside the 60s window.
        limiter.requests[0] -= 61
        stamp, count = limiter.tokens[0]
        limiter.tokens[0] = (stamp - 61, count)
        assert limiter._wait_for(100, time.monotonic()) == 0

    def test_first_request_never_waits(self):
        limiter = RateLimiter(max_requests=1, max_tokens=10)
        assert limiter._wait_for(10**6, time.monotonic()) == 0
