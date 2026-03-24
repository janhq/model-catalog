"""
Fetch a single Jan model from HuggingFace and output it in model_catalog_v2.json format.

Usage:
  python fetch_newest_jan_model.py
  python fetch_newest_jan_model.py janhq/Jan-v3-4B-base-instruct-gguf

If a repo ID is passed as an argument, it overrides LATEST_MODEL_NAME below.
"""

import json
import sys

import requests

from catalog_helpers import (
    HF_BASE_API_URL,
    HEADERS,
    REQUEST_TIMEOUT,
    detect_library_name,
    process_gguf_model,
    process_mlx_model,
)

# --- Configuration ---
LATEST_MODEL_NAME = "janhq/Jan-v3-4B-base-instruct-gguf"
DISPLAY_NAME = "Jan V3"
OUTPUT_FILE = "latest_jan_model.json"


def fetch_single_model(repo_id: str):
    print(f"Fetching model: {repo_id}")

    try:
        r = requests.get(
            f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        r.raise_for_status()
        detail = r.json()
    except Exception as e:
        print(f"Failed to fetch model {repo_id}: {e}")
        sys.exit(1)

    library = detect_library_name(detail)
    print(f"Detected library: {library}")

    if library == "gguf":
        entry = process_gguf_model(repo_id, detail)
    elif library == "mlx":
        entry = process_mlx_model(repo_id, detail)
    else:
        print(f"Unsupported library type: {library}")
        sys.exit(1)

    if not entry:
        print("No valid model data found (no quants/safetensors files).")
        sys.exit(1)

    entry["display_name"] = DISPLAY_NAME

    catalog = [entry]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nWritten to: {OUTPUT_FILE}")
    print(f"Model: {entry['model_name']} ({entry['library_name']})")
    if library == "gguf":
        print(f"Quants: {entry['num_quants']}, mmproj: {entry['num_mmproj']}")
    elif library == "mlx":
        print(f"Safetensors files: {entry['num_safetensors']}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else LATEST_MODEL_NAME
    fetch_single_model(model_name)
