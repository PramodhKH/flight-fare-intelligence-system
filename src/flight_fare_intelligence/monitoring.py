"""Lightweight in-memory request telemetry for local production demonstrations."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RequestTelemetry:
    """Track bounded request counts, statuses, and recent latency without external services."""

    max_latency_samples: int = 2_000
    total_requests: int = 0
    status_counts: Counter[int] = field(default_factory=Counter)
    path_counts: Counter[str] = field(default_factory=Counter)
    latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=2_000))

    def __post_init__(self) -> None:
        if self.max_latency_samples < 1:
            raise ValueError("max_latency_samples must be positive")
        self.latency_ms = deque(self.latency_ms, maxlen=self.max_latency_samples)

    def observe(self, *, path: str, status_code: int, latency_ms: float) -> None:
        """Record one completed request."""
        self.total_requests += 1
        self.status_counts[int(status_code)] += 1
        self.path_counts[str(path)] += 1
        self.latency_ms.append(float(latency_ms))

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe bounded telemetry summary."""
        samples = np.asarray(self.latency_ms, dtype=float)
        errors = sum(count for code, count in self.status_counts.items() if code >= 400)
        if len(samples):
            p50 = float(np.quantile(samples, 0.50))
            p95 = float(np.quantile(samples, 0.95))
            mean = float(np.mean(samples))
        else:
            p50 = p95 = mean = 0.0
        error_rate = errors / self.total_requests * 100.0 if self.total_requests else 0.0
        top_paths = [
            {"path": path, "requests": count} for path, count in self.path_counts.most_common(10)
        ]
        return {
            "total_requests": self.total_requests,
            "error_requests": errors,
            "error_rate_percent": round(error_rate, 4),
            "latency_samples": len(samples),
            "mean_latency_ms": round(mean, 4),
            "p50_latency_ms": round(p50, 4),
            "p95_latency_ms": round(p95, 4),
            "status_counts": {
                str(code): count for code, count in sorted(self.status_counts.items())
            },
            "top_paths": top_paths,
            "scope": "in-memory local telemetry; resets when the API process restarts",
        }
