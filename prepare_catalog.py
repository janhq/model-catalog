import json
import math
import os
import time
import openai
import requests

# --- Configuration ---
HF_BASE_API_URL   = "https://huggingface.co/api"
OUTPUT_FILE       = "model_catalog.json"
MODELS_PER_PAGE   = 1_000_000
MAX_PAGES         = None       # None = fetch until exhausted
REQUEST_DELAY     = 0.1
REQUEST_TIMEOUT   = 10         # seconds

# Tags to look for in the summary metadata
DESIRED_TAGS = {"text-generation", "conversational", "llama"}

client = openai.OpenAI(
    base_url="https://nano.jan.ai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

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
    # Load existing catalog into a map so we can update downloads in-place
    existing_map = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                key = f"{entry['developer']}/{entry['model_name']}"
                existing_map[key] = entry

    final_catalog = []
    seen = set()
    offset = 0
    added = 0
    page = 0

    while page < 2:
        resp = requests.get(
            f"{HF_BASE_API_URL}/models",
            params={"limit": MODELS_PER_PAGE, "offset": offset},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        summaries = resp.json()
        if not summaries:
            break

        for summary in summaries:
            repo_id   = summary.get("id")
            print(f"processing {repo_id}")
            downloads = summary.get("downloads", 0)
            if not repo_id:
                continue

            # update downloads if already seen
            if repo_id in existing_map:
                existing_map[repo_id]["downloads"] = downloads
                seen.add(repo_id)
                continue
            if repo_id in seen:
                continue

            # only consider repos whose summary tags include any DESIRED_TAGS
            summary_tags = set(summary.get("tags", []))
            if not summary_tags.intersection(DESIRED_TAGS):
                seen.add(repo_id)
                continue

            model_name = repo_id.split("/")[-1]
            developer  = repo_id.split("/")[0]
            createdAt  = summary.get("createdAt")

            # fetch full metadata
            time.sleep(REQUEST_DELAY)
            detail = requests.get(
                f"{HF_BASE_API_URL}/models/{repo_id}?blobs=true",
                timeout=REQUEST_TIMEOUT
            ).json()

            # normalize and collect GGUF files + README
            quants      = []
            readme_url  = None
            readme_text = None

            for sib in detail.get("siblings", []):
                raw = sib.get("rfilename") or sib.get("filename")
                if not raw:
                    continue
                name = raw.lower()
                print(name)
                url  = f"https://huggingface.co/{repo_id}/resolve/main/{raw}"

                if name.endswith(".gguf") and all(x not in name for x in ("embedding","ocr","speech")):
                    quants.append({
                        "model_id": raw.rsplit(".gguf", 1)[0],
                        "path": url,
                        "file_size": convert_bytes_to_human_readable(sib.get("size"))
                    })
                elif name == "readme.md":
                    readme_url = url
                    try:
                        d = requests.get(url, timeout=REQUEST_TIMEOUT)
                        d.raise_for_status()
                        readme_text = d.text
                    except Exception:
                        readme_text = None

            # only keep repos that actually have GGUF files
            if not quants:
                seen.add(repo_id)
                continue

            # summarize the README if present
            description = ""
            if readme_text:
                try:
                    description = summarize_readme_one_liner(readme_text.strip())
                except Exception:
                    description = ""

            entry = {
                "model_name": model_name,
                "developer": developer,
                "downloads": downloads,
                "createdAt": createdAt,
                "num_quants": len(quants),
                "quants": quants,
                "readme": readme_url,
                "description": description
            }

            final_catalog.append(entry)
            existing_map[repo_id] = entry
            seen.add(repo_id)
            added += 1

        offset += MODELS_PER_PAGE
        page  += 1
        time.sleep(REQUEST_DELAY)

    # merge updated existing entries + new ones
    for entry in existing_map.values():
        final_catalog.append(entry)

    final_catalog = list(existing_map.values())

    # sort: menlo first, then alphabetically by model_name
    final_catalog.sort(key=lambda e: (e["developer"] != "Menlo", e["model_name"].lower()))

    # write out catalog
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_catalog, f, indent=2, ensure_ascii=False)

    print(f"Added {added} new models; catalog total now {len(final_catalog)}")

if __name__ == "__main__":
    get_gguf_model_catalog()

