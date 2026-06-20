"""Append a captured Codex session to the shared AGENTS.md transcript log."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"(?i)(AZURE_OPENAI_API_KEY\s*=\s*)([^\s]+)"),
    re.compile(r"(?i)\b(api[_-]?key\s*[:=]\s*)([^\s]+)"),
    re.compile(r"(?i)\b(apikey\s*[:=]\s*)([^\s]+)"),
    re.compile(r"(?i)\b(authorization\s*[:=]\s*bearer\s+)([A-Za-z0-9._~+/=-]+)"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{12,})\b"),
]


def default_log_path() -> Path:
    return Path.home() / "hackerrank_orchestrate" / "log.txt"


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda match: match.group(1) + "[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _section_body(session_text: str, label: str, timestamp: str | None = None) -> str:
    stamp = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    body = _normalize_newlines(redact_secrets(session_text)).rstrip("\n")
    return f"## [{stamp}] CODEX SESSION — {label}\n{body}\n\n"


def append_codex_section(
    session_text: str,
    *,
    label: str,
    log_path: str | Path | None = None,
) -> str:
    """Append a labelled Codex session block unless the same body already exists."""
    path = Path(log_path) if log_path is not None else default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    redacted_body = _normalize_newlines(redact_secrets(session_text)).rstrip("\n")
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = f"] CODEX SESSION — {label}\n{redacted_body}\n"
    if marker in existing:
        return "already merged"

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_section_body(redacted_body, label))
    return "merged"


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "log.txt"
        session = (
            "ran tests\n"
            "AZURE_OPENAI_API_KEY=abc123\n"
            "Authorization: Bearer token-value\n"
            "sk-abcdefghijklmnopqrstuvwxyz\n"
        )
        result = append_codex_section(session, label="Codex P5-CX", log_path=log_path)
        assert result == "merged", result
        text = log_path.read_text(encoding="utf-8")
        assert "CODEX SESSION — Codex P5-CX" in text
        assert "abc123" not in text
        assert "token-value" not in text
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in text
        assert "[REDACTED]" in text

        result = append_codex_section(session, label="Codex P5-CX", log_path=log_path)
        assert result == "already merged", result
        assert log_path.read_text(encoding="utf-8") == text


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a captured Codex session into the shared transcript log."
    )
    parser.add_argument("--label", required=False, default="Codex session")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("session_file", nargs="?")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        print("merge_codex_log self-test passed")
        return 0
    if not args.session_file:
        print("session_file is required unless --self-test is used", file=sys.stderr)
        return 2
    session_path = Path(args.session_file)
    session_text = session_path.read_text(encoding="utf-8")
    result = append_codex_section(
        session_text,
        label=args.label,
        log_path=args.log_path,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
