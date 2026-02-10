"""
Model Catalog V2 Generator (Fully Independent)

This script generates model_catalog_v2.json which includes:
- All GGUF models (library_name: "gguf") 
- MLX models (library_name: "mlx")

This is completely independent from model_catalog.json (v1).
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Literal

import openai
import requests

# --- Configuration ---
HF_BASE_API_URL = "https://huggingface.co/api"
OUTPUT_FILE_V2 = "model_catalog_v2.json"
MODELS_PER_PAGE = 100000
MAX_PAGES = 1
REQUEST_DELAY = 0.1
REQUEST_TIMEOUT = 10

priority_devs = ["Menlo", "janhq", "cortexso", "mlx-community"]

# Tags to look for
DESIRED_TAGS_GGUF = {"text-generation", "conversational", "llama", "image-text-to-text"}
DESIRED_TAGS_MLX = {"text-generation", "conversational", "mlx"}

BLACKLISTED_DEVELOPERS = {
    "TheBloke",
    "Mungert",
    "UmeAiRT",
    "ReadyArt"
}

PINNED_GGUF_MODELS = [
    "janhq/Jan-v3-4B-base-instruct-gguf",
    "janhq/Jan-v2-VL-med-gguf",
    "janhq/Jan-v2-VL-high-gguf",
    "unsloth/Olmo-3-7B-Instruct-GGUF",
    "unsloth/Olmo-3-7B-Think-GGUF",
    "unsloth/Olmo-3-32B-Think-GGUF",
]

PINNED_MLX_MODELS = [
    "mlx-community/Jan-v3-4B-base-instruct-4bit"
]

ALL_PINNED_MODELS = PINNED_GGUF_MODELS + PINNED_MLX_MODELS

client = openai.OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def summarize_readme_one_liner(text: str) -> str:
    resp = client.chat.completions.create(
        model="jan/jan-v2-30b-instruct",
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
    """Check if a GGUF file is a multi-part file."""
    multipart_pattern = r".*-\d+-of-\d+\.gguf$"
    return bool(re.match(multipart_pattern, filename, re.IGNORECASE))


def is_mmproj_file(filename):
    """Check if a file is a multimodal projection model file."""
    name = filename.lower()
    return name.startswith("mmproj") and name.endswith(".gguf")


def detect_library_name(detail: dict) -> Literal["gguf", "mlx", "transformers", "unknown"]:
    """Detect the library/format type from HuggingFace model details."""
    library_name = detail.get("library_name", "").lower()
    if library_name == "mlx":
        return "mlx"
    if library_name == "gguf":
        return "gguf"
    
    tags = set(tag.lower() for tag in detail.get("tags", []))
    if "mlx" in tags:
        return "mlx"
    if "gguf" in tags:
        return "gguf"
    
    siblings = detail.get("siblings", [])
    has_gguf = any(s.get("rfilename", "").lower().endswith(".gguf") for s in siblings)
    has_safetensors = any(s.get("rfilename", "").lower().endswith(".safetensors") for s in siblings)
    has_mlx_config = any("mlx" in s.get("rfilename", "").lower() for s in siblings)
    
    if has_gguf:
        return "gguf"
    if has_safetensors and has_mlx_config:
        return "mlx"
    if library_name == "transformers":
        return "transformers"
    
    return "unknown"


def process_gguf_model(repo_id: str, detail: dict, existing_entry: dict = None) -> dict | None:
    """Process GGUF model details. Returns None if model should be skipped."""
    if not detail:
        return None

    developer = repo_id.split("/")[0]
    model_name = repo_id.split("/")[-1]

    if developer in BLACKLISTED_DEVELOPERS:
        print(f"Filtering out blacklisted developer: {developer}/{model_name}")
        return None

    downloads = detail.get("downloads", 0)
    createdAt = detail.get("createdAt")

    # Check for tool support
    supports_tools = False
    gguf_data = detail.get("gguf")
    if gguf_data and isinstance(gguf_data, dict):
        chat_template = gguf_data.get("chat_template")
        if chat_template and isinstance(chat_template, str) and "for tool" in chat_template:
            supports_tools = True

    # Collect GGUF files
    quants = []
    mmproj_models = []
    readme_url = None
    readme_text = None
    has_multipart_gguf = False

    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()

        if name.endswith(".gguf") and is_multipart_gguf(raw):
            has_multipart_gguf = True

    if has_multipart_gguf:
        print(f"  -> Repository contains multi-part GGUF files, skipping")
        return None

    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()
        url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

        if name.endswith(".gguf"):
            if is_mmproj_file(name):
                mmproj_models.append({
                    "model_id": raw.rsplit(".gguf", 1)[0],
                    "path": url,
                    "file_size": convert_bytes_to_human_readable(sib.get("size")),
                })
            elif all(x not in name for x in ("embedding", "ocr", "speech", "reranker", "encoder", "clip")):
                quants.append({
                    "model_id": raw.rsplit(".gguf", 1)[0],
                    "path": url,
                    "file_size": convert_bytes_to_human_readable(sib.get("size")),
                })
        elif name == "readme.md":
            readme_url = url
            try:
                d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
                d.raise_for_status()
                readme_text = d.text
            except Exception:
                readme_text = None

    if not quants and not mmproj_models:
        return None

    # Summarize README
    description = ""
    if existing_entry:
        description = existing_entry.get("description", "")

    if readme_text and (not existing_entry or not existing_entry.get("description")):
        try:
            description = summarize_readme_one_liner(readme_text.strip())
            print(f"  -> Generated description")
        except Exception as e:
            print(f"  -> Failed to summarize README: {e}")
            description = existing_entry.get("description", "") if existing_entry else ""

    return {
        "model_name": model_name,
        "developer": developer,
        "downloads": downloads,
        "createdAt": createdAt,
        "library_name": "gguf",
        "tools": supports_tools,
        "num_quants": len(quants),
        "quants": quants,
        "num_mmproj": len(mmproj_models),
        "mmproj_models": mmproj_models,
        "readme": readme_url,
        "description": description,
    }


def process_mlx_model(repo_id: str, detail: dict, existing_entry: dict = None) -> dict | None:
    """Process MLX model details. Returns None if model should be skipped."""
    if not detail:
        return None

    developer = repo_id.split("/")[0]
    model_name = repo_id.split("/")[-1]

    if developer in BLACKLISTED_DEVELOPERS:
        print(f"Filtering out blacklisted developer: {developer}/{model_name}")
        return None

    downloads = detail.get("downloads", 0)
    createdAt = detail.get("createdAt")

    # Collect safetensors files for MLX
    safetensors_files = []
    readme_url = None
    readme_text = None
    config_url = None

    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()
        url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

        if name.endswith(".safetensors"):
            safetensors_files.append({
                "model_id": raw.rsplit(".safetensors", 1)[0],
                "path": url,
                "file_size": convert_bytes_to_human_readable(sib.get("size")),
            })
        elif name == "readme.md":
            readme_url = url
            try:
                d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
                d.raise_for_status()
                readme_text = d.text
            except Exception:
                readme_text = None
        elif name == "config.json":
            config_url = url

    if not safetensors_files:
        return None

    # Summarize README
    description = ""
    if existing_entry:
        description = existing_entry.get("description", "")

    if readme_text and (not existing_entry or not existing_entry.get("description")):
        try:
            description = summarize_readme_one_liner(readme_text.strip())
            print(f"  -> Generated description")
        except Exception as e:
            print(f"  -> Failed to summarize README: {e}")
            description = existing_entry.get("description", "") if existing_entry else ""

    return {
        "model_name": model_name,
        "developer": developer,
        "downloads": downloads,
        "createdAt": createdAt,
        "library_name": "mlx",
        "tools": False,  # MLX tool support TBD
        "num_safetensors": len(safetensors_files),
        "safetensors_files": safetensors_files,
        "config": config_url,
        "readme": readme_url,
        "description": description,
    }


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
    
    print("\n=== Fetching MLX models from HuggingFace API ===")
    
    params = {
        "library": "mlx",
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
        
        print(f"Found {len(summaries)} MLX models")
        
        for summary in summaries:
            repo_id = summary.get("id")
            if not repo_id:
                continue
            
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
        print(f"Failed to fetch MLX models: {e}")
    
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
