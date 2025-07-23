import json
import math
import os
import time
import openai
import requests

# --- Configuration ---
HF_BASE_API_URL   = "https://huggingface.co/api"
OUTPUT_FILE       = "model_catalog.json"
MODELS_PER_PAGE   = 100000  # Reasonable page size
MAX_PAGES         = 1   # Set a reasonable limit to prevent infinite loops
REQUEST_DELAY     = 0.1
REQUEST_TIMEOUT   = 10         # seconds
priority_devs = ["Menlo", "cortexso"]

# Tags to look for in the summary metadata
DESIRED_TAGS = {"text-generation", "conversational", "llama"}

client = openai.OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def summarize_readme_one_liner(text: str) -> str:
    resp = client.chat.completions.create(
        model="Jan-nano-128k",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes Hugging Face README.md files in one shortest sentence as possible."
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=60
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
    val = round(b / (1024 ** i), 1)
    return f"{val} {units[i]}"

def get_gguf_model_catalog():
    # Load existing catalog into a map so we can update downloads and other fields in-place
    existing_map = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                key = f"{entry.get('developer', '')}/{entry.get('model_name', '')}"
                if key != '/':
                    existing_map[key] = entry

    page = 0
    added_or_updated = 0
    processed_models = set()  # Track processed models to avoid duplicates
    api_processed_models = set()  # Track models found in API

    # First pass: Process models from the API
    print("=== PASS 1: Processing models from Hugging Face API ===")
    while page < MAX_PAGES:
        # Add pagination parameters
        params = {
            'limit': MODELS_PER_PAGE,
            'skip': page * MODELS_PER_PAGE
        }

        print(f"Fetching page {page + 1}...")

        try:
            resp = requests.get(
                f"{HF_BASE_API_URL}/models",
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS
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
            print(f"Processing {repo_id}")

            model_name = repo_id.split("/")[-1]
            developer = repo_id.split("/")[0]

            # Create the correct key format
            entry_key = f"{developer}/{model_name}"
            api_processed_models.add(entry_key)  # Track this model was found in API

            # Only consider repos whose summary tags include any DESIRED_TAGS
            summary_tags = set(summary.get("tags", []))
            if not summary_tags.intersection(DESIRED_TAGS):
                continue

            downloads = summary.get("downloads", 0)
            createdAt = summary.get("createdAt")

            # Check if we need to process this model
            existing_entry = existing_map.get(entry_key)
            needs_processing = False
            missing_keys = []
            # Define expected keys that every entry should have
            expected_keys = ["model_name", "developer", "downloads", "createdAt", "tools", "num_quants", "quants", "readme", "description"]
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
                    print(f"  -> Downloads changed: {existing_entry.get('downloads', 0)} -> {downloads}")
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
                    headers=HEADERS
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
                if chat_template and isinstance(chat_template, str) and "for tool" in chat_template:
                    supports_tools = True
            print(f"  -> Tool support: {supports_tools}")

            # Normalize and collect GGUF files + README
            quants = []
            readme_url = None
            readme_text = None

            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()
                url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf") and all(x not in name for x in ("embedding", "ocr", "speech", "reranker")):
                    quants.append({
                        "model_id": raw.rsplit(".gguf", 1)[0],
                        "path": url,
                        "file_size": convert_bytes_to_human_readable(sib.get("size"))
                    })
                elif name == "readme.md":
                    readme_url = url
                    try:
                        d = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
                        d.raise_for_status()
                        readme_text = d.text
                    except Exception as e:
                        print(f"  -> Failed to fetch README: {e}")
                        readme_text = None

            # Only keep repos that actually have GGUF files
            if not quants:
                print(f"  -> No GGUF files found, skipping")
                continue

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
                    description = existing_entry.get("description", "") if existing_entry else ""

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
                "description": description
            }

            # If updating an existing entry, preserve values that don't need to change
            if existing_entry:
                # Keep existing description if we didn't generate a new one
                if not readme_text or (existing_entry.get("description") and not description):
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
        repo_id = f"{existing_entry.get('developer', '')}/{existing_entry.get('model_name', '')}"

        print(f"Processing existing model not in API: {repo_id}")

        # Check if this existing entry has missing keys
        expected_keys = ["model_name", "developer", "downloads", "createdAt", "tools", "num_quants", "quants", "readme", "description"]
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
                headers=HEADERS
            )
            r.raise_for_status()
            detail = r.json()

            # Extract basic info
            downloads = detail.get("downloads", existing_entry.get("downloads", 0))
            createdAt = detail.get("createdAt", existing_entry.get("createdAt"))

            print(f"  -> Successfully fetched model details")

        except requests.exceptions.RequestException as e:
            print(f"  -> Model not accessible via API ({e}), using existing data where possible")
            detail = {}
            downloads = existing_entry.get("downloads", 0)
            createdAt = existing_entry.get("createdAt")

        # Check for tool support in chat template
        supports_tools = False
        gguf_data = detail.get("gguf")
        if gguf_data and isinstance(gguf_data, dict):
            chat_template = gguf_data.get("chat_template")
            if chat_template and isinstance(chat_template, str) and "for tool" in chat_template:
                supports_tools = True
        elif "tools" in existing_entry:
            # Keep existing tools value if we can't fetch new data
            supports_tools = existing_entry["tools"]

        print(f"  -> Tool support: {supports_tools}")

        # Collect GGUF files from API or keep existing
        quants = []
        readme_url = existing_entry.get("readme")

        if detail.get("siblings"):
            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename")
                if not raw:
                    continue
                name = raw.lower()
                url = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf") and all(x not in name for x in ("embedding", "ocr", "speech", "reranker")):
                    quants.append({
                        "model_id": raw.rsplit(".gguf", 1)[0],
                        "path": url,
                        "file_size": convert_bytes_to_human_readable(sib.get("size"))
                    })
                elif name == "readme.md":
                    readme_url = url
        else:
            # Keep existing quants if we can't fetch new data
            quants = existing_entry.get("quants", [])

        # Use existing description
        description = existing_entry.get("description", "")

        # Update the entry with any missing fields
        updated_entry = existing_entry.copy()  # Start with existing data

        # Only update fields that were missing or need updating
        updated_entry.update({
            "model_name": existing_entry.get("model_name", repo_id.split("/")[-1]),
            "developer": existing_entry.get("developer", repo_id.split("/")[0]),
            "downloads": downloads,
            "createdAt": createdAt,
            "tools": supports_tools,
            "num_quants": len(quants) if quants else existing_entry.get("num_quants", 0),
            "quants": quants if quants else existing_entry.get("quants", []),
            "readme": readme_url,
            "description": description
        })

        existing_map[entry_key] = updated_entry
        added_or_updated += 1
        print(f"  -> Updated existing entry with missing keys")

    final_catalog = list(existing_map.values())

    # Sort: priority devs first, then alphabetically by model_name
    final_catalog.sort(key=lambda e: (
        e.get("developer", "") not in priority_devs, 
        e.get("model_name", "").lower()
    ))

    # Write out catalog
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_catalog, f, indent=2, ensure_ascii=False)

    print(f"Updated or Added {added_or_updated} models; catalog total now {len(final_catalog)}")

if __name__ == "__main__":
    get_gguf_model_catalog()
