from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
IMAGES_ROOT = DATASET_ROOT / "images"
OUTPUT_CSV = REPO_ROOT / "output.csv"

CLAIMS_CSV = DATASET_ROOT / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_ROOT / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_ROOT / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_ROOT / "evidence_requirements.csv"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION",
    "2025-01-01-preview",
)

# Pinned from probe_azure.py (code/azure_probe_report.json): this resource
# exposes gpt-5.4 (reasoner) and gpt-4.1 (vision), both text+vision capable.
# Cheap vision handles the high-volume per-image inspection; the stronger
# reasoner runs the single synthesis/decision loop (decision #6).
INSPECTION_MODEL = os.environ.get("INSPECTION_MODEL", "gpt-4.1")
SYNTHESIS_MODEL = os.environ.get("SYNTHESIS_MODEL", "gpt-5.4")

MAX_LOOP_ITERS = 8
MAX_RETRIES = 4


def azure_client():
    from openai import AzureOpenAI

    # Read at call time, not import time: callers (e.g. main.py) may load a
    # .env into os.environ after this module is first imported.
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION)

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing Azure OpenAI environment variable(s): {missing_list}"
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
