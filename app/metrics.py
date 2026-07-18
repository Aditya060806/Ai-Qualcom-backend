"""Latency / throughput instrumentation for a single generation.

TTFT (time to first token) and tokens-per-second are the two numbers judges and
the dashboard care about, so every chat response carries them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationMetrics:
    start: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None
    tokens: int = 0

    def on_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.tokens += 1

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return round((self.first_token_time - self.start) * 1000, 2)

    def summary(self) -> dict:
        now = time.perf_counter()
        gen_time = (now - self.first_token_time) if self.first_token_time else 0.0
        tps = round(self.tokens / gen_time, 2) if gen_time > 0 else None
        return {
            "ttft_ms": self.ttft_ms,
            "tokens": self.tokens,
            "tokens_per_second": tps,
            "total_ms": round((now - self.start) * 1000, 2),
        }
