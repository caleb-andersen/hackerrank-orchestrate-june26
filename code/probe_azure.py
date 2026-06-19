#!/usr/bin/env python3
"""Azure OpenAI capability probe  [Claude Code / P0].

Purpose
-------
Before we pin models in `config.py`, verify *empirically* which Azure OpenAI
deployments actually exist in this resource/region and which of them accept
**image input** (vision). We do NOT trust model-name assumptions; the plan's
top risk is "named model is text-only or region-locked", so we probe.

What it does
------------
1. Reads connection settings from environment variables only (never hardcoded):
       AZURE_OPENAI_ENDPOINT      e.g. https://my-res.openai.azure.com
       AZURE_OPENAI_API_KEY
       AZURE_OPENAI_API_VERSION   e.g. 2024-10-21 (defaults below if unset)
2. Determines the candidate deployment names to test:
       AZURE_PROBE_DEPLOYMENTS="gpt-4.1,gpt-4o,gpt-5,o4-mini"  (comma-separated)
   ...falling back to a sensible default candidate list if unset.
3. For each candidate, runs two tiny calls:
       - a text-only completion  -> proves the deployment is reachable
       - a single-image completion -> proves vision support
4. Prints a capability table + a recommended (inspection, synthesis) pairing,
   and writes the same to `code/azure_probe_report.json` for the record.

It NEVER prints secrets. Costs a few cents total.

Usage
-----
    pip install openai
    # set env vars (or use a .env you load yourself), then:
    python code/probe_azure.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

# --- Default API version & candidate deployments (override via env) -----------
DEFAULT_API_VERSION = "2025-06-01"
DEFAULT_CANDIDATES = ["gpt-4.1", "gpt-4o", "gpt-5", "gpt-5-mini", "o4-mini", "o3"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Loads <repo>/.env if present.

    Only sets vars that aren't already in the environment, so real env vars
    win. Lines are `KEY=VALUE`; `#` comments and blank lines are ignored;
    surrounding quotes on VALUE are stripped.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# A real sample image makes the vision test representative; fall back to a
# tiny inline PNG if the dataset isn't present where we're run.
SAMPLE_IMAGE = REPO_ROOT / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"
# 1x1 red PNG (used only if no sample image is available).
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _fail(msg: str, code: int = 1) -> None:
    print(f"[probe] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _load_image_data_url() -> str:
    """Return a data: URL for the vision test (real sample if available)."""
    if SAMPLE_IMAGE.is_file():
        raw = SAMPLE_IMAGE.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    return "data:image/png;base64," + _TINY_PNG_B64


def _client():
    try:
        from openai import AzureOpenAI
    except ImportError:
        _fail("`openai` not installed. Run: pip install openai")

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
    missing = [
        n
        for n, v in (
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        )
        if not v
    ]
    if missing:
        _fail(
            "missing env var(s): "
            + ", ".join(missing)
            + ". Set them (do not hardcode) and re-run."
        )
    from openai import AzureOpenAI  # noqa: F811  (re-import after import check)

    print(f"[probe] endpoint={endpoint!r}  api_version={api_version!r}")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


def _candidates() -> list[str]:
    env = os.environ.get("AZURE_PROBE_DEPLOYMENTS")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    return DEFAULT_CANDIDATES


def _try_text(client, deployment: str) -> tuple[bool, str]:
    try:
        r = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_completion_tokens=5,
        )
        return True, (r.choices[0].message.content or "").strip()[:40]
    except Exception as e:  # noqa: BLE001 - probe wants the reason, not a crash
        return False, type(e).__name__ + ": " + str(e)[:160]


def _try_vision(client, deployment: str, data_url: str) -> tuple[bool, str]:
    try:
        r = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with one word: what object type is this?"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_completion_tokens=10,
        )
        return True, (r.choices[0].message.content or "").strip()[:40]
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__ + ": " + str(e)[:160]


def main() -> None:
    _load_dotenv()
    client = _client()
    data_url = _load_image_data_url()
    using_real = SAMPLE_IMAGE.is_file()
    print(f"[probe] vision test image: {'real sample case_001/img_1.jpg' if using_real else 'inline 1x1 png'}")
    print(f"[probe] candidates: {_candidates()}\n")

    results: list[dict] = []
    for dep in _candidates():
        t_ok, t_msg = _try_text(client, dep)
        v_ok, v_msg = (False, "skipped (text failed)")
        if t_ok:
            v_ok, v_msg = _try_vision(client, dep, data_url)
        results.append(
            {"deployment": dep, "text_ok": t_ok, "text_detail": t_msg,
             "vision_ok": v_ok, "vision_detail": v_msg}
        )
        status = "TEXT+VISION" if v_ok else ("TEXT only" if t_ok else "UNAVAILABLE")
        print(f"  {dep:<14} {status:<12} | text: {t_msg!r} | vision: {v_msg!r}")
        time.sleep(0.3)  # be gentle on RPM

    vision_models = [r["deployment"] for r in results if r["vision_ok"]]
    text_models = [r["deployment"] for r in results if r["text_ok"]]

    print("\n[probe] === recommendation ===")
    if not vision_models:
        print("  No vision-capable deployment found. Inspection step cannot run.")
        print("  -> Provision a vision model (gpt-4o / gpt-4.1) before P2.")
    else:
        inspection = vision_models[0]
        # Prefer a stronger reasoner for synthesis if reachable; else reuse vision model.
        synth_pref = ["o3", "o4-mini", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o"]
        synthesis = next((m for m in synth_pref if m in text_models), inspection)
        print(f"  inspection (vision, high volume): {inspection}")
        print(f"  synthesis  (final decision)     : {synthesis}")
        print(f"  comparison config B             : gpt-4o (if present)")

    report = {
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        "vision_capable": vision_models,
        "text_capable": text_models,
        "results": results,
    }
    out = Path(__file__).resolve().parent / "azure_probe_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
