"""The agent loop: a single reasoner drives a real tool-calling loop per claim.

One claim row in -> one 14-column decision dict out (plus a `_routing` block the
evaluation harness consumes). The reasoner (`synth_model`) never sees pixels; it
calls tools and we feed back structured JSON:

  - inspect_image(image_id)            -> tools.inspect_image (cheap vision)
  - get_evidence_requirement(...)      -> injected lookup (Codex P2-CX)
  - get_user_history(user_id)          -> injected lookup (Codex P2-CX)
  - submit_decision(...)               -> terminates the loop (synthesis schema)

The lookups and the image inspector are **dependency-injected** as plain
callables, so this module imports nothing from the concrete lookup
implementations and is fully unit-testable with fakes (see `_self_test`). It is
also model-agnostic apart from the OpenAI chat-completions tool-calling shape.

Routing (decision #5 — measure it, collapse the design if it is flat):
  - early_stop:    decided without inspecting every available image
  - reinspected:   inspected more than one image
  - label_flipped: a re-inspection round changed the final claim_status

Safety: if the model never submits, loops past `max_iters`, or errors, we return
the `not_enough_information` + `manual_review_required` fallback row.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.prompts import (
    EVIDENCE_FAMILY_HINTS,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_user_text,
)
from agent.synthesis import SUBMIT_DECISION_TOOL, decision_to_row, fallback_row
from config import MAX_LOOP_ITERS

# Injected callable signatures (bound by main.py):
InspectFn = Callable[[str], dict]          # (image_id) -> observation
EvidenceFn = Callable[[str, str], dict]     # (claim_object, issue_family) -> req
HistoryFn = Callable[[str], dict]           # (user_id) -> history


def _evidence_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_evidence_requirement",
            "description": (
                "Return the minimum image evidence required to evaluate this kind "
                "of claim. Use it to ground evidence_standard_met."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_object": {
                        "type": "string",
                        "enum": ["car", "laptop", "package", "all"],
                    },
                    "issue_family": {
                        "type": "string",
                        "description": (
                            "The kind of issue, e.g. one of: "
                            + "; ".join(EVIDENCE_FAMILY_HINTS)
                        ),
                    },
                },
                "required": ["claim_object", "issue_family"],
            },
        },
    }


def _history_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_user_history",
            "description": (
                "Return the user's prior-claim risk context. Use it to decide "
                "user_history_risk (history adds risk but does not override clear "
                "visual evidence)."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    }


def _inspect_tool(image_ids: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": (
                "Run a vision inspection on ONE submitted image and return a "
                "structured observation. Inspect the images you need to decide."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "image_id": {
                        "type": "string",
                        "enum": image_ids or ["none"],
                    }
                },
                "required": ["image_id"],
            },
        },
    }


def _assistant_message(message: Any) -> dict:
    """Serialize an SDK assistant message (with tool calls) back into a dict."""
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in message.tool_calls
        ]
    return out


def _tool_result(call_id: str, payload: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def run_claim(
    row: dict[str, str],
    *,
    client: Any,
    synth_model: str,
    inspect_fn: InspectFn,
    evidence_fn: EvidenceFn,
    history_fn: HistoryFn,
    available_image_ids: list[str],
    max_iters: int = MAX_LOOP_ITERS,
    max_tokens: int = 3000,
) -> dict[str, str]:
    """Run one claim through the tool loop; return a 14-column row + `_routing`."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_synthesis_user_text(
                user_id=row.get("user_id", ""),
                claim_object=row.get("claim_object", ""),
                user_claim=row.get("user_claim", ""),
                image_ids=available_image_ids,
            ),
        },
    ]
    tools = [
        _inspect_tool(available_image_ids),
        _evidence_tool(),
        _history_tool(),
        SUBMIT_DECISION_TOOL,
    ]

    inspected: list[str] = []        # image ids inspected, in call order
    synth_calls = 0
    reinspect_budget = 1             # bounded re-inspection round (decision #5)
    first_status: str | None = None
    nudged = False

    def finalize(args: dict[str, Any], flipped: bool) -> dict[str, str]:
        decided = decision_to_row(args, row)
        decided["_routing"] = _routing(flipped)
        return decided

    def _routing(flipped: bool) -> dict[str, Any]:
        unique = list(dict.fromkeys(inspected))
        n_available = len(available_image_ids)
        return {
            "early_stop": bool(n_available > 1 and len(unique) < n_available
                               and len(unique) >= 1),
            "reinspected": len(inspected) > 1,
            "label_flipped": bool(flipped),
            "images_available": n_available,
            "images_inspected": len(inspected),
            "synth_calls": synth_calls,
            "inspect_calls": len(inspected),
        }

    for _ in range(max_iters):
        try:
            response = client.chat.completions.create(
                model=synth_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - any API failure -> safe fallback
            row_out = fallback_row(row, f"synthesis call failed: {type(exc).__name__}")
            row_out["_routing"] = _routing(False)
            return row_out

        synth_calls += 1
        message = response.choices[0].message

        if not message.tool_calls:
            # The model answered in prose without using a tool. Nudge once.
            if nudged:
                row_out = fallback_row(row, "model did not call submit_decision")
                row_out["_routing"] = _routing(False)
                return row_out
            nudged = True
            messages.append(_assistant_message(message))
            messages.append({
                "role": "user",
                "content": (
                    "You must finish by calling the submit_decision tool with the "
                    "structured fields. Do that now."
                ),
            })
            continue

        messages.append(_assistant_message(message))

        submit_args: dict[str, Any] | None = None
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_decision":
                submit_args = args
                # A tool result is required even though we may terminate.
                messages.append(_tool_result(call.id, {"status": "received"}))
            elif name == "inspect_image":
                image_id = str(args.get("image_id", "")).strip()
                observation = inspect_fn(image_id)
                inspected.append(image_id)
                messages.append(_tool_result(call.id, observation))
            elif name == "get_evidence_requirement":
                payload = evidence_fn(
                    str(args.get("claim_object", "")).strip(),
                    str(args.get("issue_family", "")).strip(),
                )
                messages.append(_tool_result(call.id, payload))
            elif name == "get_user_history":
                payload = history_fn(str(args.get("user_id", "")).strip())
                messages.append(_tool_result(call.id, payload))
            else:
                messages.append(_tool_result(call.id, {"error": f"unknown tool {name}"}))

        if submit_args is None:
            continue

        status = str(submit_args.get("claim_status", "")).strip()
        uninspected = [i for i in available_image_ids if i not in inspected]

        # Bounded re-inspection: if the model gives up (NEI) without having
        # looked at every image, push it to inspect the rest before finalizing.
        if (
            reinspect_budget > 0
            and status == "not_enough_information"
            and uninspected
        ):
            reinspect_budget -= 1
            first_status = status
            messages.append({
                "role": "user",
                "content": (
                    "Before finalizing: you have not inspected "
                    f"{', '.join(uninspected)} yet. Inspect the remaining "
                    "image(s), then call submit_decision again."
                ),
            })
            continue

        flipped = first_status is not None and first_status != status
        return finalize(submit_args, flipped)

    # Exhausted iterations without a final decision.
    row_out = fallback_row(row, f"exceeded {max_iters} loop iterations")
    row_out["_routing"] = _routing(False)
    return row_out


# --------------------------------------------------------------------------- #
# Offline self-test: drives the loop with a fake client + fake tools (no
# network, no Codex modules). Verifies tool dispatch, routing, and the
# fallback path.
# --------------------------------------------------------------------------- #

class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = type(
            "F", (), {"name": name, "arguments": json.dumps(arguments)}
        )()


class _FakeMessage:
    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeResponse:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


class _ScriptedClient:
    """Replays a scripted sequence of assistant turns."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **_kwargs):  # noqa: D401 - mimics client.chat.completions.create
        return _FakeResponse(self._turns.pop(0))


def _self_test() -> None:
    row = {
        "user_id": "user_x",
        "image_paths": "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg",
        "user_claim": "Customer: rear bumper dented.",
        "claim_object": "car",
    }
    ids = ["img_1", "img_2"]

    # Scenario A: inspect img_1, then submit supported -> early_stop (img_2 skipped).
    turns_a = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "inspect_image", {"image_id": "img_1"})]),
        _FakeMessage(tool_calls=[_FakeToolCall("c2", "submit_decision", {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "rear bumper visible",
            "risk_flags": [],
            "issue_type": "dent",
            "object_part": "rear_bumper",
            "claim_status": "supported",
            "claim_status_justification": "img_1 shows the dent",
            "supporting_image_ids": ["img_1"],
            "valid_image": True,
            "severity": "medium",
        })]),
    ]
    out_a = run_claim(
        row, client=_ScriptedClient(turns_a), synth_model="fake",
        inspect_fn=lambda i: {"image_id": i, "readable": True},
        evidence_fn=lambda o, f: {"requirement_id": "REQ", "minimum_image_evidence": "x"},
        history_fn=lambda u: {"user_id": u, "history_flags": "none"},
        available_image_ids=ids,
    )
    assert out_a["claim_status"] == "supported", out_a
    assert out_a["_routing"]["early_stop"] is True, out_a["_routing"]
    assert out_a["_routing"]["reinspected"] is False, out_a["_routing"]
    assert out_a["_routing"]["label_flipped"] is False

    # Scenario B: NEI without inspecting img_2 -> forced re-inspect -> flips to
    # contradicted. Tests the bounded re-inspection + label_flipped routing.
    turns_b = [
        _FakeMessage(tool_calls=[_FakeToolCall("c1", "inspect_image", {"image_id": "img_1"})]),
        _FakeMessage(tool_calls=[_FakeToolCall("c2", "submit_decision", {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "unclear",
            "risk_flags": [],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "cannot tell from img_1",
            "supporting_image_ids": [],
            "valid_image": True,
            "severity": "unknown",
        })]),
        _FakeMessage(tool_calls=[_FakeToolCall("c3", "inspect_image", {"image_id": "img_2"})]),
        _FakeMessage(tool_calls=[_FakeToolCall("c4", "submit_decision", {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "img_2 clear",
            "risk_flags": ["claim_mismatch", "manual_review_required"],
            "issue_type": "scratch",
            "object_part": "rear_bumper",
            "claim_status": "contradicted",
            "claim_status_justification": "img_2 shows only a scratch",
            "supporting_image_ids": ["img_2"],
            "valid_image": True,
            "severity": "low",
        })]),
    ]
    out_b = run_claim(
        row, client=_ScriptedClient(turns_b), synth_model="fake",
        inspect_fn=lambda i: {"image_id": i, "readable": True},
        evidence_fn=lambda o, f: {"requirement_id": "REQ"},
        history_fn=lambda u: {"user_id": u},
        available_image_ids=ids,
    )
    assert out_b["claim_status"] == "contradicted", out_b
    assert out_b["_routing"]["reinspected"] is True, out_b["_routing"]
    assert out_b["_routing"]["label_flipped"] is True, out_b["_routing"]

    # Scenario C: model never calls a tool -> nudge -> fallback.
    turns_c = [
        _FakeMessage(content="I think it's fine."),
        _FakeMessage(content="Still just talking."),
    ]
    out_c = run_claim(
        row, client=_ScriptedClient(turns_c), synth_model="fake",
        inspect_fn=lambda i: {}, evidence_fn=lambda o, f: {}, history_fn=lambda u: {},
        available_image_ids=ids,
    )
    assert out_c["claim_status"] == "not_enough_information", out_c
    assert out_c["risk_flags"] == "manual_review_required", out_c
    assert out_c["supporting_image_ids"] == "none"
    assert out_c["severity"] == "unknown"


if __name__ == "__main__":
    _self_test()
    print("loop self-test passed")
