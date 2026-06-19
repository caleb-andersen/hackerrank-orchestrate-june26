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
import json
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

    def to_json(self) -> str:
        """Serialize raw counters/timings so accounting can resume after reload."""
        payload = {
            "calls": {
                str(model): {
                    "calls": int(counts.get("calls", 0)),
                    "prompt_tokens": int(counts.get("prompt_tokens", 0)),
                    "completion_tokens": int(counts.get("completion_tokens", 0)),
                }
                for model, counts in self._calls.items()
            },
            "timings_seconds": {
                str(name): float(seconds)
                for name, seconds in self._timings.items()
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Instrument":
        data = json.loads(text)
        inst = cls()

        calls = data.get("calls", data.get("_calls", {}))
        if isinstance(calls, dict):
            for model, counts in calls.items():
                if not isinstance(counts, dict):
                    continue
                inst._calls[str(model)] = {
                    "calls": int(counts.get("calls", 0) or 0),
                    "prompt_tokens": int(counts.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        counts.get("completion_tokens", 0) or 0
                    ),
                }

        timings = data.get("timings_seconds", data.get("_timings", {}))
        restored_timings: defaultdict[str, float] = defaultdict(float)
        if isinstance(timings, dict):
            for name, seconds in timings.items():
                try:
                    restored_timings[str(name)] += float(seconds)
                except (TypeError, ValueError):
                    continue
        inst._timings = restored_timings
        return inst

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Instrument":
        try:
            return cls.from_json(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def merge(self, other: "Instrument") -> None:
        """Fold another Instrument's raw counters/timings into this one."""
        for model, counts in other._calls.items():
            bucket = self._calls.setdefault(
                model,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            bucket["calls"] += int(counts.get("calls", 0))
            bucket["prompt_tokens"] += int(counts.get("prompt_tokens", 0))
            bucket["completion_tokens"] += int(counts.get("completion_tokens", 0))

        for name, seconds in other._timings.items():
            self._timings[name] += float(seconds)

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

    def save(self, path: str | Path) -> None:
        payload = {
            "use_content_hash": self.use_content_hash,
            "items": self._items,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        use_content_hash: bool = True,
    ) -> "InspectionCache":
        cache = cls(use_content_hash=use_content_hash)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cache

        items = data.get("items") if isinstance(data, dict) else None
        if items is None and isinstance(data, dict):
            # Backward-compatible plain key->observation cache.
            items = data
        if not isinstance(items, dict):
            return cache
        cache._items = {str(key): value for key, value in items.items()}
        return cache

    def snapshot(self) -> dict[str, int]:
        return {"entries": len(self._items)}


instrument = Instrument()
inspection_cache = InspectionCache()


def _self_test() -> None:
    import tempfile

    inst = Instrument()
    inst.record_call("gpt-4.1", {"prompt_tokens": 100, "completion_tokens": 25})
    with inst.track("unit"):
        pass
    snap = inst.snapshot()
    assert snap["totals"]["calls"] == 1, snap
    assert snap["totals"]["prompt_tokens"] == 100, snap
    assert snap["totals"]["completion_tokens"] == 25, snap
    assert "unit" in snap["timings_seconds"], snap

    with tempfile.TemporaryDirectory() as tmp_dir:
        inst_path = Path(tmp_dir) / "instrument.json"
        inst.save(inst_path)
        loaded = Instrument.load(inst_path)
        assert loaded.snapshot() == inst.snapshot()

        other = Instrument()
        other.record_call("gpt-4.1", {"prompt_tokens": 5, "completion_tokens": 7})
        other.record_call("gpt-5.4", {"input_tokens": 11, "output_tokens": 13})
        loaded.merge(other)
        merged = loaded.snapshot()
        assert merged["models"]["gpt-4.1"]["calls"] == 2, merged
        assert merged["models"]["gpt-4.1"]["prompt_tokens"] == 105, merged
        assert merged["models"]["gpt-4.1"]["completion_tokens"] == 32, merged
        assert merged["models"]["gpt-5.4"]["calls"] == 1, merged
        assert Instrument.load(Path(tmp_dir) / "missing.json").snapshot()["totals"][
            "calls"
        ] == 0

    cache = InspectionCache(use_content_hash=False)
    key = cache.key_for("/tmp/example.jpg")
    assert cache.get(key) is None
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir) / "inspection_cache.json"
        cache.save(cache_path)
        loaded_cache = InspectionCache.load(cache_path, use_content_hash=False)
        assert loaded_cache.get(key) == {"ok": True}
        assert loaded_cache.snapshot() == {"entries": 1}
        assert InspectionCache.load(Path(tmp_dir) / "missing.json").snapshot() == {
            "entries": 0
        }
        bad_path = Path(tmp_dir) / "bad.json"
        bad_path.write_text("{bad json", encoding="utf-8")
        assert InspectionCache.load(bad_path).snapshot() == {"entries": 0}


if __name__ == "__main__":
    _self_test()
    print("instrument self-test passed")
