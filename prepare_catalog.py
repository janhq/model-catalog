import json
import math
import os
import re
import time
from collections import defaultdict

import requests

# --- Configuration ---
HF_BASE_API_URL = "https://huggingface.co/api"
OUTPUT_FILE = "model_catalog.json"
MODELS_PER_PAGE = 100000  # Reasonable page size
MAX_PAGES = 1  # Set a reasonable limit to prevent infinite loops
REQUEST_DELAY = 0.1
REQUEST_TIMEOUT = 10  # seconds
priority_devs = ["Menlo", "cortexso"]

# Tags to look for in the summary metadata
DESIRED_TAGS = {"text-generation", "conversational", "llama"}

BLACKLISTED_DEVELOPERS = {
    "TheBloke",
}


HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def summarize_readme_one_liner(text: str) -> str:
    resp = client.chat.completions.create(
        model="Jan-nano-128k",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes Hugging Face README.md files in one shortest sentence as possible.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=60,
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
        developer = entry.get("developer", "unknown")
        model_name = entry.get("model_name", "unknown")

        if not quants:
            # No quants at all, remove
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
            # Sort by downloads (descending) to keep the one with most downloads
            entries.sort(key=lambda x: x.get("downloads", 0), reverse=True)
            best_entry = entries[0]
            removed_duplicates += len(entries) - 1

            print(f"Duplicate model '{model_name}' found in {len(entries)} repos:")
            for i, entry in enumerate(entries):
                developer = entry.get("developer", "unknown")
                downloads = entry.get("downloads", 0)
                status = "KEPT" if i == 0 else "REMOVED"
                print(
                    f"  - {developer}/{model_name} ({downloads} downloads) [{status}]"
                )
        else:
            best_entry = entries[0]

        final_entries.append(best_entry)

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
            # First, remove duplicates and multipart files from existing catalog
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

            # Normalize and collect GGUF files + README
            quants = []
            readme_url = None
            readme_text = None
            has_multipart_gguf = False
            total_gguf_files = 0

            # First pass: check for any multi-part GGUF files
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()

                if name.endswith(".gguf"):
                    total_gguf_files += 1
                    # Check if ANY GGUF file is multi-part (regardless of type)
                    if is_multipart_gguf(raw):
                        print(f"  -> Found multi-part GGUF: {raw}")
                        has_multipart_gguf = True

            # If ANY multi-part GGUF file is found, skip this entire repository
            if has_multipart_gguf:
                print(
                    f"  -> Repository contains multi-part GGUF files, skipping entire repo"
                )
                continue

            # Second pass: collect valid GGUF files (only if no multi-part files were found)
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()
                url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf") and all(
                    x not in name for x in ("embedding", "ocr", "speech", "reranker")
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
                    try:
                        d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
                        d.raise_for_status()
                        readme_text = d.text
                    except Exception as e:
                        print(f"  -> Failed to fetch README: {e}")
                        readme_text = None

            # Only keep repos that actually have valid GGUF files
            if not quants:
                print(f"  -> No valid model GGUF files found, skipping")
                continue

            # Summarize the README if present and if it's a new entry or missing description
            description = ""
            if existing_entry:
                description = existing_entry.get("description", "")

            if readme_text and (
                not existing_entry or not existing_entry.get("description")
            ):
                try:
                    description = summarize_readme_one_liner(readme_text.strip())
                    print(f"  -> Generated new description")
                except Exception as e:
                    print(f"  -> Failed to summarize README: {e}")
                    description = (
                        existing_entry.get("description", "") if existing_entry else ""
                    )

            # Create or update entry, preserving existing values where appropriate
            entry = {
                "model_name": model_name,
                "developer": developer,
                "downloads": downloads,  # Always update downloads
                "createdAt": createdAt,
                "tools": supports_tools,
                "num_quants": len(quants),
                "quants": quants,
                "readme": readme_url,
                "description": description,
            }

            # If updating an existing entry, preserve values that don't need to change
            if existing_entry:
                # Keep existing description if we didn't generate a new one
                if not readme_text or (
                    existing_entry.get("description") and not description
                ):
                    entry["description"] = existing_entry.get("description", "")
                # Keep existing createdAt if current one is missing
                if not createdAt and existing_entry.get("createdAt"):
                    entry["createdAt"] = existing_entry.get("createdAt")

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

        # Collect GGUF files from API or keep existing (but check for multipart)
        quants = []
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

            # Second pass: collect valid GGUF files (only if no multi-part files were found)
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()
                url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf") and all(
                    x not in name for x in ("embedding", "ocr", "speech", "reranker")
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
            # Check existing quants for multi-part files
            existing_quants = existing_entry.get("quants", [])
            has_multipart_gguf = False

            for quant in existing_quants:
                quant_path = quant.get("path", "")
                filename = quant_path.split("/")[-1] if quant_path else ""
                if is_multipart_gguf(filename):
                    has_multipart_gguf = True
                    print(f"  -> Found multi-part GGUF in existing quants: {filename}")
                    break

            # If any existing quant is multipart, remove this repository
            if has_multipart_gguf:
                print(
                    f"  -> Existing repository contains multi-part GGUF files, removing from catalog"
                )
                if entry_key in existing_map:
                    del existing_map[entry_key]
                continue
            else:
                # Keep all existing quants since none are multipart
                quants = existing_quants

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
                "readme": readme_url,
                "description": description,
            }
        )

        existing_map[entry_key] = updated_entry
        added_or_updated += 1
        print(f"  -> Updated existing entry with missing keys")

    # Convert to list and apply final duplicate removal and multipart filtering
    print("\n=== FINAL CLEANUP: Removing duplicates and multipart files ===")
    preliminary_catalog = list(existing_map.values())
    final_catalog = remove_duplicates_and_multipart(preliminary_catalog)

    # Sort: priority devs first, then alphabetically by model_name
    final_catalog.sort(
        key=lambda e: (
            e.get("developer", "") not in priority_devs,
            e.get("model_name", "").lower(),
        )
    )

    # Write out catalog
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_catalog, f, indent=2, ensure_ascii=False)

    print(
        f"\nUpdated or Added {added_or_updated} models; catalog total now {len(final_catalog)}"
    )


if __name__ == "__main__":
    get_gguf_model_catalog()
