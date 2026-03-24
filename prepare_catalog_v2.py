"""
Model Catalog V2 Generator (Fully Independent)

This script generates model_catalog_v2.json which includes:
- All GGUF models (library_name: "gguf")
- MLX models (library_name: "mlx")

This is completely independent from model_catalog.json (v1).
"""

import json
import os
import time
from collections import defaultdict

import requests

from catalog_helpers import (
    BLACKLISTED_DEVELOPERS,
    HEADERS,
    HF_BASE_API_URL,
    REQUEST_TIMEOUT,
    process_gguf_model,
    process_mlx_model,
)

# --- Configuration ---
OUTPUT_FILE_V2 = "model_catalog_v2.json"
MODELS_PER_PAGE = 100000
MAX_PAGES = 1
REQUEST_DELAY = 0.1

priority_devs = ["Menlo", "janhq", "cortexso", "mlx-community"]

# Tags to look for
DESIRED_TAGS_GGUF = {"text-generation", "conversational", "llama", "image-text-to-text"}
DESIRED_TAGS_MLX = {"text-generation", "conversational", "mlx"}

PINNED_GGUF_MODELS = [
    "janhq/Jan-v3.5-4B-gguf",
    "janhq/Jan-code-4b-gguf",
    "janhq/Jan-v2-VL-max-gguf",
    "janhq/Jan-v3-4B-base-instruct-gguf",
    "janhq/Jan-v2-VL-med-gguf",
    "janhq/Jan-v2-VL-high-gguf",
    "unsloth/Qwen3.5-35B-A3B-GGUF",
    "unsloth/Olmo-3-7B-Instruct-GGUF",
    "unsloth/Olmo-3-7B-Think-GGUF",
    "unsloth/Olmo-3-32B-Think-GGUF",
    "Menlo/Jan-nano-128k-gguf"
]

PINNED_MLX_MODELS = [
    "mlx-community/Jan-v3-4B-base-instruct-4bit",
    "janhq/Jan-v2-VL-high-4bit-mlx",
    "janhq/Jan-v2-VL-high-8bit-mlx",
    "janhq/Jan-v2-VL-high-bf16-mlx",
    "janhq/Jan-v2-VL-med-4bit-mlx",
    "janhq/Jan-v2-VL-med-8bit-mlx",
    "janhq/Jan-v2-VL-med-bf16-mlx",
    "janhq/Jan-v2-VL-low-4bit-mlx",
    "janhq/Jan-v2-VL-low-8bit-mlx",
    "janhq/Jan-v2-VL-low-bf16-mlx"
]

ALL_PINNED_MODELS = PINNED_GGUF_MODELS + PINNED_MLX_MODELS


def load_existing_v2_catalog() -> dict:
    """Load existing v2 catalog as a map for incremental updates."""
    existing_map = {}
    if os.path.exists(OUTPUT_FILE_V2):
        with open(OUTPUT_FILE_V2, "r", encoding="utf-8") as f:
            catalog = json.load(f)
            for entry in catalog:
                key = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
                if key != "/":
                    existing_map[key] = entry
        print(f"Loaded {len(existing_map)} existing entries from {OUTPUT_FILE_V2}")
    return existing_map


def fetch_gguf_models(existing_map: dict) -> list:
    """Fetch GGUF models from HuggingFace."""
    gguf_models = []
    processed_models = set()

    print("\n=== Fetching GGUF models from HuggingFace API ===")

    page = 0
    while page < MAX_PAGES:
        params = {"limit": MODELS_PER_PAGE, "skip": page * MODELS_PER_PAGE}
        print(f"Fetching page {page + 1}...")

        try:
            resp = requests.get(
                f"{HF_BASE_API_URL}/models",
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            resp.raise_for_status()
            summaries = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed for page {page}: {e}")
            break
        except ValueError:
            print(f"Failed to parse JSON response for page {page}")
            break

        if not summaries:
            print("No more models found, stopping pagination.")
            break

        page_processed = 0
        for summary in summaries:
            repo_id = summary.get("id")
            if not repo_id or repo_id in processed_models:
                continue

            processed_models.add(repo_id)
            developer = repo_id.split("/")[0]
            model_name = repo_id.split("/")[-1]

            if developer in BLACKLISTED_DEVELOPERS:
                continue

            # Only consider repos whose summary tags include any DESIRED_TAGS_GGUF
            summary_tags = set(summary.get("tags", []))
            if not summary_tags.intersection(DESIRED_TAGS_GGUF):
                continue

            entry_key = f"{developer}/{model_name}"
            existing_entry = existing_map.get(entry_key)

            # Skip if existing entry is up to date (same downloads)
            downloads = summary.get("downloads", 0)
            if existing_entry and existing_entry.get("library_name") == "gguf":
                if existing_entry.get("downloads") == downloads and existing_entry.get("description"):
                    gguf_models.append(existing_entry)
                    continue

            print(f"Processing GGUF: {repo_id}")

            time.sleep(REQUEST_DELAY)
            try:
                r = requests.get(
                    f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
                    timeout=REQUEST_TIMEOUT,
                    headers=HEADERS,
                )
                r.raise_for_status()
                detail = r.json()

                entry = process_gguf_model(repo_id, detail, existing_entry)
                if entry:
                    gguf_models.append(entry)
                    page_processed += 1
                    print(f"  -> Added with {entry['num_quants']} quants")
            except Exception as e:
                print(f"  -> Failed: {e}")

        print(f"Processed {page_processed} GGUF models from page {page + 1}")
        page += 1
        time.sleep(REQUEST_DELAY)

        if len(summaries) < MODELS_PER_PAGE:
            print("Reached end of available models.")
            break

    # Process pinned GGUF models
    print("\n=== Processing pinned GGUF models ===")
    for pinned_repo_id in PINNED_GGUF_MODELS:
        if any(f"{m.get('developer')}/{m.get('model_name')}" == pinned_repo_id for m in gguf_models):
            print(f"Pinned model {pinned_repo_id} already in list")
            continue

        print(f"Processing pinned GGUF: {pinned_repo_id}")
        existing_entry = existing_map.get(pinned_repo_id)

        time.sleep(REQUEST_DELAY)
        try:
            r = requests.get(
                f"{HF_BASE_API_URL}/models/{pinned_repo_id}?blobs=true",
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            r.raise_for_status()
            detail = r.json()

            entry = process_gguf_model(pinned_repo_id, detail, existing_entry)
            if entry:
                gguf_models.append(entry)
                print(f"  -> Added pinned GGUF model")
        except Exception as e:
            print(f"  -> Failed: {e}")

    return gguf_models


def fetch_mlx_models(existing_map: dict) -> list:
    """Fetch MLX models from HuggingFace."""
    mlx_models = []
    processed_repos = set()

    print("\n=== Fetching MLX models from HuggingFace API ===")

    # Fetch from mlx-community author
    mlx_authors = ["mlx-community"]

    for author in mlx_authors:
        print(f"Fetching models from author: {author}")
        params = {
            "author": author,
            "limit": 100,
            "sort": "downloads",
            "direction": -1,
        }

        try:
            resp = requests.get(
                f"{HF_BASE_API_URL}/models",
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            resp.raise_for_status()
            summaries = resp.json()

            print(f"Found {len(summaries)} models from {author}")

            for summary in summaries:
                repo_id = summary.get("id")
                if not repo_id or repo_id in processed_repos:
                    continue

                processed_repos.add(repo_id)
                developer = repo_id.split("/")[0]
                model_name = repo_id.split("/")[-1]

                if developer in BLACKLISTED_DEVELOPERS:
                    continue

                entry_key = f"{developer}/{model_name}"
                existing_entry = existing_map.get(entry_key)

                # Skip if existing entry is up to date
                downloads = summary.get("downloads", 0)
                if existing_entry and existing_entry.get("library_name") == "mlx":
                    if existing_entry.get("downloads") == downloads and existing_entry.get("description"):
                        mlx_models.append(existing_entry)
                        continue

                print(f"Processing MLX: {repo_id}")

                time.sleep(REQUEST_DELAY)
                try:
                    r = requests.get(
                        f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
                        timeout=REQUEST_TIMEOUT,
                        headers=HEADERS,
                    )
                    r.raise_for_status()
                    detail = r.json()

                    entry = process_mlx_model(repo_id, detail, existing_entry)
                    if entry:
                        mlx_models.append(entry)
                        print(f"  -> Added with {entry['num_safetensors']} safetensors files")
                except Exception as e:
                    print(f"  -> Failed: {e}")

        except Exception as e:
            print(f"Failed to fetch MLX models from {author}: {e}")

    # Process pinned MLX models
    print("\n=== Processing pinned MLX models ===")
    for pinned_repo_id in PINNED_MLX_MODELS:
        if any(f"{m.get('developer')}/{m.get('model_name')}" == pinned_repo_id for m in mlx_models):
            print(f"Pinned model {pinned_repo_id} already in list")
            continue

        print(f"Processing pinned MLX: {pinned_repo_id}")
        existing_entry = existing_map.get(pinned_repo_id)

        time.sleep(REQUEST_DELAY)
        try:
            r = requests.get(
                f"{HF_BASE_API_URL}/models/{pinned_repo_id}?blobs=true",
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            r.raise_for_status()
            detail = r.json()

            entry = process_mlx_model(pinned_repo_id, detail, existing_entry)
            if entry:
                mlx_models.append(entry)
                print(f"  -> Added pinned MLX model")
        except Exception as e:
            print(f"  -> Failed: {e}")

    return mlx_models


def remove_duplicates(catalog: list) -> list:
    """Remove duplicate models, keeping the one with higher downloads."""
    model_groups = defaultdict(list)

    for entry in catalog:
        model_name = entry.get("model_name", "")
        if model_name:
            model_groups[model_name].append(entry)

    final_entries = []

    for model_name, entries in model_groups.items():
        if len(entries) > 1:
            # Group by library_name to keep both GGUF and MLX versions
            by_library = defaultdict(list)
            for e in entries:
                by_library[e.get("library_name", "unknown")].append(e)

            for library, lib_entries in by_library.items():
                if len(lib_entries) > 1:
                    # Keep pinned/priority, or highest downloads
                    kept = None
                    for e in lib_entries:
                        repo_id = f"{e.get('developer', '')}/{e.get('model_name', '')}"
                        if repo_id in ALL_PINNED_MODELS:
                            kept = e
                            break
                        if e.get("developer") in priority_devs:
                            kept = e
                            break

                    if not kept:
                        lib_entries.sort(key=lambda x: x.get("downloads", 0), reverse=True)
                        kept = lib_entries[0]

                    final_entries.append(kept)
                else:
                    final_entries.append(lib_entries[0])
        else:
            final_entries.append(entries[0])

    return final_entries


def generate_v2_catalog():
    """Generate model_catalog_v2.json with GGUF + MLX models."""

    # Load existing v2 catalog for incremental updates
    existing_map = load_existing_v2_catalog()

    # Fetch GGUF models
    gguf_models = fetch_gguf_models(existing_map)
    print(f"\nTotal GGUF models: {len(gguf_models)}")

    # Fetch MLX models
    mlx_models = fetch_mlx_models(existing_map)
    print(f"Total MLX models: {len(mlx_models)}")

    # Combine
    all_models = gguf_models + mlx_models

    # Remove duplicates
    print("\n=== Removing duplicates ===")
    all_models = remove_duplicates(all_models)
    print(f"After dedup: {len(all_models)} models")

    # Sort
    def sort_key(entry):
        repo_id = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
        is_pinned = repo_id in ALL_PINNED_MODELS
        is_priority_dev = entry.get("developer", "") in priority_devs
        return (
            not is_pinned,
            not is_priority_dev,
            -entry.get("downloads", 0),
            entry.get("model_name", "").lower(),
        )

    all_models.sort(key=sort_key)

    # Write v2 catalog
    with open(OUTPUT_FILE_V2, "w", encoding="utf-8") as f:
        json.dump(all_models, f, indent=2, ensure_ascii=False)

    # Print summary
    gguf_count = sum(1 for m in all_models if m.get("library_name") == "gguf")
    mlx_count = sum(1 for m in all_models if m.get("library_name") == "mlx")

    print(f"\n=== V2 Catalog Summary ===")
    print(f"Total models: {len(all_models)}")
    print(f"  - GGUF models: {gguf_count}")
    print(f"  - MLX models: {mlx_count}")
    print(f"Written to: {OUTPUT_FILE_V2}")

    # Final line for CI to parse
    print(f"\nUpdated catalog; total now {len(all_models)} (GGUF: {gguf_count}, MLX: {mlx_count})")


if __name__ == "__main__":
    generate_v2_catalog()
