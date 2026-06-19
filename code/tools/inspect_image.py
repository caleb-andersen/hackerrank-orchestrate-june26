"""inspect_image — one vision call over a single image -> structured JSON.

This is the only place in the system that sends raw pixels to a model, and it
uses the cheap, high-volume vision deployment (`INSPECTION_MODEL`). The reasoner
that drives the loop never sees the image bytes; it only ever consumes the JSON
this returns (decision #2 + #6).

The function is defensive: a missing file, a decode failure, or a model/parse
error never raises into the loop. It returns a well-formed "unreadable"
observation instead, so the reasoner can still make a `not_enough_information`
decision rather than crashing the whole run.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from agent.prompts import INSPECTION_SYSTEM_PROMPT, build_inspection_user_text

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# Stable shape every observation conforms to, regardless of model output.
_OBSERVATION_KEYS = {
    "readable": bool,
    "object_type": str,
    "object_matches_claim": bool,
    "parts_visible": list,
    "damage_observed": str,
    "damage_parts": list,
    "damage_severity": str,
    "quality_flags": list,
    "authenticity_flags": list,
    "text_in_image": str,
    "text_instruction_present": bool,
    "notes": str,
}


def _unreadable(image_id: str, reason: str) -> dict[str, Any]:
    """A safe observation used when an image cannot be inspected at all."""
    return {
        "image_id": image_id,
        "readable": False,
        "object_type": "unknown",
        "object_matches_claim": False,
        "parts_visible": [],
        "damage_observed": "unknown",
        "damage_parts": [],
        "damage_severity": "unknown",
        "quality_flags": ["cropped_or_obstructed"],
        "authenticity_flags": [],
        "text_in_image": "",
        "text_instruction_present": False,
        "notes": f"Image could not be inspected: {reason}",
    }


def _coerce(observation: dict[str, Any], image_id: str) -> dict[str, Any]:
    """Force model output into the stable observation shape with safe defaults."""
    blank = _unreadable(image_id, "missing fields")
    out: dict[str, Any] = {"image_id": image_id}
    for key, kind in _OBSERVATION_KEYS.items():
        value = observation.get(key, blank[key])
        if kind is bool:
            out[key] = bool(value)
        elif kind is list:
            out[key] = [str(v).strip() for v in value] if isinstance(value, list) else []
        else:
            out[key] = str(value).strip()
    return out


def _data_url(path: Path) -> str:
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def inspect_image(
    image_id: str,
    image_path: Path,
    claim_object: str,
    *,
    client: Any,
    model: str,
    max_tokens: int = 600,
    instrument: Any = None,
) -> dict[str, Any]:
    """Inspect one image and return a structured observation dict.

    Never raises: file/model/parse failures degrade to an "unreadable"
    observation so the agent loop can keep going.

    `instrument` (optional) is a `code/instrument.py` Instrument; when supplied
    we record this call's token usage so the P4 operational report has real
    per-model numbers. It never changes the observation, only accounting.
    """
    path = Path(image_path)
    if not path.is_file():
        return _unreadable(image_id, "file not found")
    if path.suffix.lower() not in _MIME_BY_SUFFIX:
        return _unreadable(image_id, f"unsupported file type {path.suffix!r}")

    try:
        data_url = _data_url(path)
    except OSError as exc:  # unreadable bytes on disk
        return _unreadable(image_id, f"read error: {type(exc).__name__}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSPECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_inspection_user_text(claim_object, image_id),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        if instrument is not None:
            instrument.record_call(model, getattr(response, "usage", None))
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _unreadable(image_id, "model returned non-JSON")
    except Exception as exc:  # noqa: BLE001 - tool must never crash the loop
        return _unreadable(image_id, f"inspection call failed: {type(exc).__name__}")

    if not isinstance(parsed, dict):
        return _unreadable(image_id, "model returned non-object JSON")
    return _coerce(parsed, image_id)
