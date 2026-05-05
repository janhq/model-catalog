"""
Shared helpers for model catalog scripts.

Used by prepare_catalog_v2.py and fetch_newest_jan_model.py.
"""

import math
import os
import re
from typing import Literal

import openai
import requests

# --- Shared Configuration ---
HF_BASE_API_URL = "https://huggingface.co/api"
REQUEST_TIMEOUT = 10

BLACKLISTED_DEVELOPERS = {
    "TheBloke",
    "Mungert",
    "UmeAiRT",
    "ReadyArt",
}

EXCLUDED_GGUF_KEYWORDS = ("embedding", "ocr", "speech", "reranker", "encoder", "clip")

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
        )
    return _client


def summarize_readme_one_liner(text: str) -> str:
    resp = _get_client().chat.completions.create(
        model="minimaxai/minimax-m2.7",
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


def _fetch_readme_text(url: str) -> str | None:
    """Fetch README text from a URL, returning None on failure."""
    try:
        d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        d.raise_for_status()
        return d.text
    except Exception:
        return None


def _resolve_description(readme_text: str | None, existing_entry: dict | None) -> str:
    """Resolve description from README text or existing entry."""
    if existing_entry:
        description = existing_entry.get("description", "")
    else:
        description = ""

    if readme_text and (not existing_entry or not existing_entry.get("description")):
        try:
            description = summarize_readme_one_liner(readme_text.strip())
            print(f"  -> Generated description")
        except Exception as e:
            print(f"  -> Failed to summarize README: {e}")
            description = existing_entry.get("description", "") if existing_entry else ""

    return description


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

    supports_tools = False
    gguf_data = detail.get("gguf")
    if gguf_data and isinstance(gguf_data, dict):
        chat_template = gguf_data.get("chat_template")
        if chat_template and isinstance(chat_template, str) and "for tool" in chat_template:
            supports_tools = True

    quants = []
    mmproj_models = []
    readme_url = None
    readme_text = None

    for sib in detail.get("siblings", []):
        raw = sib.get("rfilename")
        if not raw:
            continue
        name = raw.lower()
        url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

        if name.endswith(".gguf"):
            if is_multipart_gguf(raw):
                print(f"  -> Skipping multi-part file: {raw}")
                continue

            if is_mmproj_file(name):
                mmproj_models.append({
                    "model_id": raw.rsplit(".gguf", 1)[0],
                    "path": url,
                    "file_size": convert_bytes_to_human_readable(sib.get("size")),
                })
            elif all(x not in name for x in EXCLUDED_GGUF_KEYWORDS):
                quants.append({
                    "model_id": raw.rsplit(".gguf", 1)[0],
                    "path": url,
                    "file_size": convert_bytes_to_human_readable(sib.get("size")),
                })
        elif name == "readme.md":
            readme_url = url
            readme_text = _fetch_readme_text(url)

    if not quants and not mmproj_models:
        return None

    description = _resolve_description(readme_text, existing_entry)

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
            readme_text = _fetch_readme_text(url)
        elif name == "config.json":
            config_url = url

    if not safetensors_files:
        return None

    description = _resolve_description(readme_text, existing_entry)

    return {
        "model_name": model_name,
        "developer": developer,
        "downloads": downloads,
        "createdAt": createdAt,
        "library_name": "mlx",
        "tools": False,
        "num_safetensors": len(safetensors_files),
        "safetensors_files": safetensors_files,
        "config": config_url,
        "readme": readme_url,
        "description": description,
    }
