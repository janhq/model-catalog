import json
import math
import os
import re
import time
from collections import defaultdict

import openai
import requests

# --- Configuration ---
HF_BASE_API_URL = "https://huggingface.co/api"
OUTPUT_FILE = "model_catalog.json"
MODELS_PER_PAGE = 100000  # Reasonable page size
MAX_PAGES = 1  # Set a reasonable limit to prevent infinite loops
REQUEST_DELAY = 0.1
REQUEST_TIMEOUT = 10  # seconds
priority_devs = ["Menlo", "janhq", "cortexso"]

# Tags to look for in the summary metadata
DESIRED_TAGS = {"text-generation", "conversational", "llama", "image-text-to-text"}

BLACKLISTED_DEVELOPERS = {
    "TheBloke",
    "Mungert",  # Temporarily blacklist Diffucoder
    "UmeAiRT",
    "ReadyArt"
}

PINNED_MODELS = ["janhq/Jan-v3-4B-base-instruct-gguf", "janhq/Jan-v2-VL-med-gguf", "janhq/Jan-v2-VL-high-gguf", "unsloth/Olmo-3-7B-Instruct-GGUF", "unsloth/Olmo-3-7B-Think-GGUF", "unsloth/Olmo-3-32B-Think-GGUF"]

client = openai.OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def summarize_readme_one_liner(text: str) -> str:
    resp = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M2.1",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes Hugging Face README.md files in one shortest sentence as possible.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=30000,
    )
    return resp.choices[0].message.content.strip()


def convert_bytes_to_human_readable(size_bytes):
    if size_bytes is None:
        return "N/A"
    try:
        b = int(size_bytes)
    except (ValueError, TypeError):
        return "N/A"
    if b == 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(b, 1024)))
    val = round(b / (1024**i), 1)
    return f"{val} {units[i]}"


def is_multipart_gguf(filename):
    """
    Check if a GGUF file is a multi-part file.
    Pattern: <model name><precision>-xxxxxxx-of-xxxxx.gguf
    """
    # Pattern to match multi-part GGUF files
    multipart_pattern = r".*-\d+-of-\d+\.gguf$"
    return bool(re.match(multipart_pattern, filename, re.IGNORECASE))


def is_mmproj_file(filename):
    """
    Check if a file is a multimodal projection model file.
    These are typically named like mmproj-model-f16.gguf or similar.
    """
    name = filename.lower()
    return name.startswith("mmproj") and name.endswith(".gguf")


def process_model_details(repo_id, detail=None, existing_entry=None):
    """
    Process model details from HF API response and return structured entry.
    Returns None if model should be skipped.
    """
    if not detail:
        return None

    developer = repo_id.split("/")[0]
    model_name = repo_id.split("/")[-1]

    # Apply filtering
    if developer in BLACKLISTED_DEVELOPERS:
        print(f"Filtering out blacklisted developer: {developer}/{model_name}")
        return None

    downloads = detail.get("downloads", 0)
    createdAt = detail.get("createdAt")

    # Check for tool support in chat template
    supports_tools = False
    gguf_data = detail.get("gguf")
    if gguf_data and isinstance(gguf_data, dict):
        chat_template = gguf_data.get("chat_template")
        if (
            chat_template
            and isinstance(chat_template, str)
            and "for tool" in chat_template
        ):
            supports_tools = True
    print(f"  -> Tool support: {supports_tools}")

    # Normalize and collect GGUF files + README, separating text models from mmproj
    quants = []
    mmproj_models = []
    readme_url = None
    readme_text = None
    has_multipart_gguf = False

    # First pass: check for any multi-part GGUF files
    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()

        if name.endswith(".gguf"):
            # Check if ANY GGUF file is multi-part (regardless of type)
            if is_multipart_gguf(raw):
                print(f"  -> Found multi-part GGUF: {raw}")
                has_multipart_gguf = True

    # If ANY multi-part GGUF file is found, skip this entire repository
    if has_multipart_gguf:
        print(f"  -> Repository contains multi-part GGUF files, skipping entire repo")
        return None

    # Second pass: collect valid GGUF files, separating text models from mmproj
    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()
        url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

        if name.endswith(".gguf"):
            # Check if it's an mmproj model
            if is_mmproj_file(name):
                mmproj_models.append(
                    {
                        "model_id": raw.rsplit(".gguf", 1)[0],
                        "path": url,
                        "file_size": convert_bytes_to_human_readable(sib.get("size")),
                    }
                )
            # Check if it's a regular text generation model (not embedding/ocr/speech/reranker)
            elif all(x not in name for x in ("embedding", "ocr", "speech", "reranker", "encoder", "clip")):
                quants.append(
                    {
                        "model_id": raw.rsplit(".gguf", 1)[0],
                        "path": url,
                        "file_size": convert_bytes_to_human_readable(sib.get("size")),
                    }
                )
        elif name == "readme.md":
            readme_url = url
            try:
                d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
                d.raise_for_status()
                readme_text = d.text
            except Exception as e:
                print(f"  -> Failed to fetch README: {e}")
                readme_text = None

    # Only keep repos that actually have valid GGUF files (either text models or mmproj)
    if not quants and not mmproj_models:
        print(f"  -> No valid model GGUF files found, skipping")
        return None

    print(
        f"  -> Found {len(quants)} text models and {len(mmproj_models)} mmproj models"
    )

    # Summarize the README if present and if it's a new entry or missing description
    description = ""
    if existing_entry:
        description = existing_entry.get("description", "")

    if readme_text and (not existing_entry or not existing_entry.get("description")):
        try:
            description = summarize_readme_one_liner(readme_text.strip())
            print(f"  -> Generated new description")
        except Exception as e:
            print(f"  -> Failed to summarize README: {e}")
            description = (
                existing_entry.get("description", "") if existing_entry else ""
            )

    # Create entry
    entry = {
        "model_name": model_name,
        "developer": developer,
        "downloads": downloads,
        "createdAt": createdAt,
        "tools": supports_tools,
        "num_quants": len(quants),
        "quants": quants,
        "num_mmproj": len(mmproj_models),
        "mmproj_models": mmproj_models,
        "readme": readme_url,
        "description": description,
    }

    # If updating an existing entry, preserve values that don't need to change
    if existing_entry:
        # Keep existing description if we didn't generate a new one
        if not readme_text or (existing_entry.get("description") and not description):
            entry["description"] = existing_entry.get("description", "")
        # Keep existing createdAt if current one is missing
        if not createdAt and existing_entry.get("createdAt"):
            entry["createdAt"] = existing_entry.get("createdAt")

    return entry


def remove_duplicates_and_multipart(catalog_data):
    """
    Remove duplicate models (keeping the one with higher downloads)
    and exclude repositories that contain ANY multi-part GGUF files.
    """
    # First, remove repositories that contain ANY multi-part GGUF files
    filtered_catalog = []
    removed_multipart_repos = 0

    for entry in catalog_data:
        quants = entry.get("quants", [])
        mmproj_models = entry.get("mmproj_models", [])
        developer = entry.get("developer", "unknown")
        model_name = entry.get("model_name", "unknown")

        if not quants and not mmproj_models:
            # No quants or mmproj at all, remove
            print(f"Removing repository with no GGUF files: {developer}/{model_name}")
            removed_multipart_repos += 1
            continue

        # Check if ANY quant is a multi-part file
        has_multipart = False
        multipart_files = []

        for quant in quants:
            quant_path = quant.get("path", "")
            filename = quant_path.split("/")[-1] if quant_path else ""

            if is_multipart_gguf(filename):
                has_multipart = True
                multipart_files.append(filename)

        # Also check mmproj models for multipart (though less common)
        for mmproj in mmproj_models:
            mmproj_path = mmproj.get("path", "")
            filename = mmproj_path.split("/")[-1] if mmproj_path else ""

            if is_multipart_gguf(filename):
                has_multipart = True
                multipart_files.append(filename)

        if has_multipart:
            # ANY multi-part file found, remove entire repository
            print(
                f"Removing repository with multi-part GGUF files: {developer}/{model_name}"
            )
            print(f"  -> Multi-part files found: {', '.join(multipart_files)}")
            removed_multipart_repos += 1
            continue

        # Repository is clean, keep it
        filtered_catalog.append(entry)

    # Now group models by model name (regardless of developer) for duplicate removal
    model_groups = defaultdict(list)

    for entry in filtered_catalog:
        model_name = entry.get("model_name", "")
        if model_name:
            model_groups[model_name].append(entry)

    # Process each group to keep only the best version
    final_entries = []
    removed_duplicates = 0

    for model_name, entries in model_groups.items():
        if len(entries) > 1:
            kept_entries = []
            removed_entries = []

            for entry in entries:
                repo_id = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
                if (
                    repo_id in PINNED_MODELS
                    or entry.get("developer", "") in priority_devs
                ):
                    kept_entries.append(entry)
                else:
                    removed_entries.append(entry)

            if not kept_entries:
                # If no pinned/priority entries, keep the one with highest downloads
                removed_entries.sort(key=lambda x: x.get("downloads", 0), reverse=True)
                kept_entries.append(removed_entries.pop(0))  # Keep best
                removed_duplicates += len(removed_entries)
            else:
                removed_duplicates += len(removed_entries)

            final_entries.extend(kept_entries)

            print(f"Duplicate model '{model_name}' found in {len(entries)} repos:")
            for entry in entries:
                repo_id = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
                downloads = entry.get("downloads", 0)
                status = "KEPT" if entry in kept_entries else "REMOVED"
                print(f"  - {repo_id} ({downloads} downloads) [{status}]")
        else:
            final_entries.append(entries[0])

    print(f"\nCleanup summary:")
    print(
        f"  - Removed {removed_multipart_repos} repositories containing multi-part GGUF files"
    )
    print(f"  - Removed {removed_duplicates} duplicate models")
    print(f"  - Final catalog size: {len(final_entries)} models")

    return final_entries


def get_gguf_model_catalog():
    # Load existing catalog into a map so we can update downloads and other fields in-place
    existing_map = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_catalog = json.load(f)
            print("=== FILTERING BLACKLISTED DEVELOPERS FROM EXISTING CATALOG ===")
            pre_filter_count = len(existing_catalog)
            existing_catalog = [
                entry
                for entry in existing_catalog
                if entry["developer"] not in BLACKLISTED_DEVELOPERS
            ]
            removed_blacklisted = pre_filter_count - len(existing_catalog)
            if removed_blacklisted > 0:
                print(
                    f"  -> Removed {removed_blacklisted} model(s) from blacklisted developers"
                )
            # Separate mmproj models from existing catalog entries
            print("=== SEPARATING MMPROJ FROM EXISTING CATALOG ===")
            for entry in existing_catalog:
                # Separate mmproj files from regular quants in existing entries
                existing_quants = entry.get("quants", [])
                existing_mmproj = entry.get("mmproj_models", [])

                separated_quants = []
                separated_mmproj = existing_mmproj.copy()  # Keep existing mmproj

                for quant in existing_quants:
                    quant_path = quant.get("path", "")
                    filename = quant_path.split("/")[-1] if quant_path else ""

                    # Check if this quant is actually an mmproj file
                    if is_mmproj_file(filename):
                        separated_mmproj.append(quant)
                        print(f"  -> Moved {filename} from quants to mmproj_models")
                    else:
                        separated_quants.append(quant)

                # Update the entry with separated files
                entry["quants"] = separated_quants
                entry["mmproj_models"] = separated_mmproj
                entry["num_quants"] = len(separated_quants)
                entry["num_mmproj"] = len(separated_mmproj)

            # Then, remove duplicates and multipart files from existing catalog
            print("=== CLEANING EXISTING CATALOG ===")
            cleaned_catalog = remove_duplicates_and_multipart(existing_catalog)
            for entry in cleaned_catalog:
                key = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
                if key != "/":
                    existing_map[key] = entry

    page = 0
    added_or_updated = 0
    processed_models = set()  # Track processed models to avoid duplicates
    api_processed_models = set()  # Track models found in API

    # First pass: Process models from the API
    print("\n=== PASS 1: Processing models from Hugging Face API ===")
    while page < MAX_PAGES:
        # Add pagination parameters
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
            model_name = repo_id.split("/")[-1]
            developer = repo_id.split("/")[0]

            # Apply filtering
            if developer in BLACKLISTED_DEVELOPERS:
                print(f"Filtering out blacklisted developer: {developer}/{model_name}")
                continue

            print(f"Processing {repo_id}")

            downloads = summary.get("downloads", 0)
            createdAt = summary.get("createdAt")

            # Create the correct key format
            entry_key = f"{developer}/{model_name}"
            api_processed_models.add(entry_key)  # Track this model was found in API

            # Only consider repos whose summary tags include any DESIRED_TAGS
            summary_tags = set(summary.get("tags", []))
            if not summary_tags.intersection(DESIRED_TAGS):
                continue

            downloads = summary.get("downloads", 0)

            # Check if we need to process this model
            existing_entry = existing_map.get(entry_key)
            needs_processing = False
            missing_keys = []
            # Define expected keys that every entry should have
            expected_keys = [
                "model_name",
                "developer",
                "downloads",
                "createdAt",
                "tools",
                "num_quants",
                "quants",
                "readme",
                "description",
                "mmproj_models",  # Added mmproj_models to expected keys
                "num_mmproj",  # Added num_mmproj to expected keys
            ]
            if existing_entry is None:
                # New entry
                needs_processing = True
                print(f"  -> New model found")
            else:
                # Check for missing keys
                for key in expected_keys:
                    if key not in existing_entry:
                        missing_keys.append(key)

                if missing_keys:
                    needs_processing = True
                    print(f"  -> Missing keys: {missing_keys}")
                elif existing_entry.get("downloads", 0) != downloads:
                    # Downloads changed
                    needs_processing = True
                    print(
                        f"  -> Downloads changed: {existing_entry.get('downloads', 0)} -> {downloads}"
                    )
                else:
                    print(f"  -> No update needed")

            if not needs_processing:
                continue

            # Fetch full metadata
            time.sleep(REQUEST_DELAY)
            try:
                r = requests.get(
                    f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
                    timeout=REQUEST_TIMEOUT,
                    headers=HEADERS,
                )
                r.raise_for_status()
                detail = r.json()
            except requests.exceptions.RequestException as e:
                print(f"  -> Request failed for {repo_id}: {e}")
                continue
            except ValueError:
                print(f"  -> Failed to parse JSON response for {repo_id}")
                continue

            # Process the model details
            entry = process_model_details(repo_id, detail, existing_entry)
            if entry is None:
                continue

            if entry_key not in existing_map:
                print(f"  -> Added new entry")
                added_or_updated += 1
            else:
                print(f"  -> Updated existing entry")
                added_or_updated += 1

            existing_map[entry_key] = entry
            page_processed += 1

        print(f"Processed {page_processed} models from page {page + 1}")
        page += 1
        time.sleep(REQUEST_DELAY)

        # If we processed fewer models than the page size, we're likely at the end
        if len(summaries) < MODELS_PER_PAGE:
            print("Reached end of available models.")
            break

    # Second pass: Process existing models in catalog that weren't found in API
    print("\n=== PASS 2: Processing existing models not found in API ===")
    existing_not_in_api = set(existing_map.keys()) - api_processed_models

    for entry_key in existing_not_in_api:
        existing_entry = existing_map[entry_key]
        developer = existing_entry.get("developer", "")
        model_name = existing_entry.get("model_name", "")
        repo_id = f"{developer}/{model_name}"

        # Apply filtering to existing entries too
        if developer in BLACKLISTED_DEVELOPERS:
            print(f"Removing blacklisted developer from existing catalog: {repo_id}")
            del existing_map[entry_key]
            continue

        print(f"Processing existing model not in API: {repo_id}")

        # Check if this existing entry has missing keys
        expected_keys = [
            "model_name",
            "developer",
            "downloads",
            "createdAt",
            "tools",
            "num_quants",
            "quants",
            "readme",
            "description",
            "mmproj_models",
            "num_mmproj",
        ]
        missing_keys = []

        for key in expected_keys:
            if key not in existing_entry:
                missing_keys.append(key)

        if not missing_keys:
            print(f"  -> No missing keys, skipping")
            continue

        print(f"  -> Missing keys: {missing_keys}")

        # Try to fetch the model details directly
        time.sleep(REQUEST_DELAY)
        try:
            r = requests.get(
                f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            r.raise_for_status()
            detail = r.json()

            # Extract basic info
            downloads = detail.get("downloads", existing_entry.get("downloads", 0))
            createdAt = detail.get("createdAt", existing_entry.get("createdAt"))

            print(f"  -> Successfully fetched model details")

        except requests.exceptions.RequestException as e:
            print(
                f"  -> Model not accessible via API ({e}), using existing data where possible"
            )
            detail = {}
            downloads = existing_entry.get("downloads", 0)
            createdAt = existing_entry.get("createdAt")

        # Check for tool support in chat template
        supports_tools = False
        gguf_data = detail.get("gguf")
        if gguf_data and isinstance(gguf_data, dict):
            chat_template = gguf_data.get("chat_template")
            if (
                chat_template
                and isinstance(chat_template, str)
                and "for tool" in chat_template
            ):
                supports_tools = True
        elif "tools" in existing_entry:
            # Keep existing tools value if we can't fetch new data
            supports_tools = existing_entry["tools"]

        print(f"  -> Tool support: {supports_tools}")

        # Collect GGUF files from API or keep existing (but check for multipart and separate mmproj)
        quants = []
        mmproj_models = []
        readme_url = existing_entry.get("readme")

        if detail.get("siblings"):
            has_multipart_gguf = False

            # First pass: check for ANY multi-part GGUF files
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()

                if name.endswith(".gguf"):
                    # Check if ANY GGUF file is multi-part (regardless of type)
                    if is_multipart_gguf(raw):
                        print(f"  -> Found multi-part GGUF: {raw}")
                        has_multipart_gguf = True
                        break

            # If ANY multi-part GGUF file is found, remove this repository
            if has_multipart_gguf:
                print(
                    f"  -> Repository contains multi-part GGUF files, removing from catalog"
                )
                if entry_key in existing_map:
                    del existing_map[entry_key]
                continue

            # Second pass: collect valid GGUF files, separating text models from mmproj
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()
                url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf"):
                    # Check if it's an mmproj model
                    if is_mmproj_file(name):
                        mmproj_models.append(
                            {
                                "model_id": raw.rsplit(".gguf", 1)[0],
                                "path": url,
                                "file_size": convert_bytes_to_human_readable(
                                    sib.get("size")
                                ),
                            }
                        )
                    # Check if it's a regular text generation model
                    elif all(
                        x not in name
                        for x in ("embedding", "ocr", "speech", "reranker")
                    ):
                        quants.append(
                            {
                                "model_id": raw.rsplit(".gguf", 1)[0],
                                "path": url,
                                "file_size": convert_bytes_to_human_readable(
                                    sib.get("size")
                                ),
                            }
                        )
                elif name == "readme.md":
                    readme_url = url
        else:
            # Check existing quants and mmproj for multi-part files and separate them properly
            existing_quants = existing_entry.get("quants", [])
            existing_mmproj = existing_entry.get("mmproj_models", [])
            has_multipart_gguf = False

            # Check existing quants for multipart
            for quant in existing_quants:
                quant_path = quant.get("path", "")
                filename = quant_path.split("/")[-1] if quant_path else ""
                if is_multipart_gguf(filename):
                    has_multipart_gguf = True
                    print(f"  -> Found multi-part GGUF in existing quants: {filename}")
                    break

            # Check existing mmproj for multipart
            if not has_multipart_gguf:
                for mmproj in existing_mmproj:
                    mmproj_path = mmproj.get("path", "")
                    filename = mmproj_path.split("/")[-1] if mmproj_path else ""
                    if is_multipart_gguf(filename):
                        has_multipart_gguf = True
                        print(
                            f"  -> Found multi-part GGUF in existing mmproj: {filename}"
                        )
                        break

            # If any existing file is multipart, remove this repository
            if has_multipart_gguf:
                print(
                    f"  -> Existing repository contains multi-part GGUF files, removing from catalog"
                )
                if entry_key in existing_map:
                    del existing_map[entry_key]
                continue
            else:
                # Keep existing files since none are multipart
                quants = existing_quants
                mmproj_models = existing_mmproj

        # Use existing description
        description = existing_entry.get("description", "")

        # Update the entry with any missing fields
        updated_entry = existing_entry.copy()  # Start with existing data

        # Only update fields that were missing or need updating
        updated_entry.update(
            {
                "model_name": existing_entry.get("model_name", repo_id.split("/")[-1]),
                "developer": existing_entry.get("developer", repo_id.split("/")[0]),
                "downloads": downloads,
                "createdAt": createdAt,
                "tools": supports_tools,
                "num_quants": (
                    len(quants) if quants else existing_entry.get("num_quants", 0)
                ),
                "quants": quants if quants else existing_entry.get("quants", []),
                "num_mmproj": (
                    len(mmproj_models)
                    if mmproj_models
                    else existing_entry.get("num_mmproj", 0)
                ),
                "mmproj_models": (
                    mmproj_models
                    if mmproj_models
                    else existing_entry.get("mmproj_models", [])
                ),
                "readme": readme_url,
                "description": description,
            }
        )

        existing_map[entry_key] = updated_entry
        added_or_updated += 1
        print(f"  -> Updated existing entry with missing keys")

    # Third pass: Process pinned models that weren't found in API or existing catalog
    print("\n=== PASS 3: Processing pinned models not found elsewhere ===")
    all_processed_models = set(existing_map.keys())

    for pinned_repo_id in PINNED_MODELS:
        if pinned_repo_id not in all_processed_models:
            print(f"Processing missing pinned model: {pinned_repo_id}")

            time.sleep(REQUEST_DELAY)
            try:
                r = requests.get(
                    f"{HF_BASE_API_URL}/models/{pinned_repo_id}?blobs=true",
                    timeout=REQUEST_TIMEOUT,
                    headers=HEADERS,
                )
                r.raise_for_status()
                detail = r.json()

                print(f"  -> Successfully fetched pinned model details")

                # Process the pinned model details
                entry = process_model_details(pinned_repo_id, detail)
                if entry is not None:
                    existing_map[pinned_repo_id] = entry
                    added_or_updated += 1
                    print(f"  -> Added pinned model to catalog")
                else:
                    print(f"  -> Pinned model failed validation, skipping")

            except requests.exceptions.RequestException as e:
                print(f"  -> Failed to fetch pinned model {pinned_repo_id}: {e}")
            except ValueError:
                print(
                    f"  -> Failed to parse JSON response for pinned model {pinned_repo_id}"
                )
        else:
            print(f"Pinned model {pinned_repo_id} already in catalog")

    # Convert to list and apply final duplicate removal and multipart filtering
    print("\n=== FINAL CLEANUP: Removing duplicates and multipart files ===")
    preliminary_catalog = list(existing_map.values())
    final_catalog = remove_duplicates_and_multipart(preliminary_catalog)

    def sort_key(entry):
        repo_id = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
        is_pinned = repo_id in PINNED_MODELS
        is_priority_dev = entry.get("developer", "") in priority_devs
        return (
            not is_pinned,  # Pinned models come first (False < True)
            not is_priority_dev,  # Then priority devs
            -entry.get("downloads", 0),  # Then by downloads descending
            entry.get("model_name", "").lower(),  # Then alphabetically
        )

    final_catalog.sort(key=sort_key)

    # Write out catalog
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_catalog, f, indent=2, ensure_ascii=False)

    print(
        f"\nUpdated or Added {added_or_updated} models; catalog total now {len(final_catalog)}"
    )


if __name__ == "__main__":
    get_gguf_model_catalog()
