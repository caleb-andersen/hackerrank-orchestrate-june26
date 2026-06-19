"""Runtime instrumentation for model usage, latency, and image inspection cache.

Pricing is an estimate only. Values are assumed USD per 1K input/output tokens
for the configured Azure deployments; update PRICING after final model pricing
is confirmed for P4 operational reporting.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Iterator

PRICING = {
    # Assumed planning rates per 1K tokens, not a billing source of truth.
    "gpt-5.4": {"input": 0.0050, "output": 0.0150},
    "gpt-4.1": {"input": 0.0020, "output": 0.0080},
    "gpt-4o": {"input": 0.0050, "output": 0.0150},
}


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


@dataclass
class Instrument:
    """Accumulate call counts, token usage, estimated cost, and wall time."""

    _calls: dict[str, dict[str, int]] = field(default_factory=dict)
    _timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record_call(self, model: str, usage: Any) -> None:
        model_name = str(model or "unknown")
        bucket = self._calls.setdefault(
            model_name,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += _usage_value(usage, "prompt_tokens", "input_tokens")
        bucket["completion_tokens"] += _usage_value(
            usage,
            "completion_tokens",
            "output_tokens",
        )

    @contextmanager
    def track(self, name: str = "total") -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] += time.perf_counter() - start

    def snapshot(self) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = {}
        totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }
        for model, counts in sorted(self._calls.items()):
            pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
            cost = (
                counts["prompt_tokens"] * pricing["input"]
                + counts["completion_tokens"] * pricing["output"]
            ) / 1000
            models[model] = {**counts, "cost_usd": cost}
            totals["calls"] += counts["calls"]
            totals["prompt_tokens"] += counts["prompt_tokens"]
            totals["completion_tokens"] += counts["completion_tokens"]
            totals["cost_usd"] += cost
        return {
            "models": models,
            "totals": totals,
            "timings_seconds": dict(self._timings),
            "pricing_assumption": "USD per 1K input/output tokens in PRICING",
        }


class InspectionCache:
    """In-memory cache keyed by absolute path and, when readable, content hash."""

    def __init__(self, use_content_hash: bool = True) -> None:
        self.use_content_hash = use_content_hash
        self._items: dict[str, Any] = {}

    def key_for(self, path: str | Path) -> str:
        resolved = str(Path(path).resolve(strict=False))
        if not self.use_content_hash:
            return resolved
        try:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return resolved
        return f"{resolved}#{digest}"

    def get(self, key: str | Path) -> Any | None:
        return self._items.get(str(key))

    def set(self, key: str | Path, observation: Any) -> Any:
        self._items[str(key)] = observation
        return observation

    def get_path(self, path: str | Path) -> Any | None:
        return self.get(self.key_for(path))

    def set_path(self, path: str | Path, observation: Any) -> Any:
        return self.set(self.key_for(path), observation)

    def snapshot(self) -> dict[str, int]:
        return {"entries": len(self._items)}


instrument = Instrument()
inspection_cache = InspectionCache()


def _self_test() -> None:
    inst = Instrument()
    inst.record_call("gpt-4.1", {"prompt_tokens": 100, "completion_tokens": 25})
    with inst.track("unit"):
        pass
    snap = inst.snapshot()
    assert snap["totals"]["calls"] == 1, snap
    assert snap["totals"]["prompt_tokens"] == 100, snap
    assert snap["totals"]["completion_tokens"] == 25, snap
    assert "unit" in snap["timings_seconds"], snap

    cache = InspectionCache(use_content_hash=False)
    key = cache.key_for("/tmp/example.jpg")
    assert cache.get(key) is None
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}


if __name__ == "__main__":
    _self_test()
    print("instrument self-test passed")
