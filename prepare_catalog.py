import requests
import json
import time
import math
import os # Import os for file path checking

# --- Configuration ---
HF_BASE_API_URL = "https://huggingface.co/api"
OUTPUT_FILE = "model_catalog.json"
MODELS_PER_PAGE = 500  # Max models per page for the /api/models endpoint
MAX_PAGES = None       # Set to `None` for unlimited pages (fetch all).
                       # Set to a number (e.g., 5) to limit pages for testing/development.
REQUEST_DELAY = 0.1    # Seconds between API requests to avoid hitting rate limits.
REQUEST_TIMEOUT = 10   # Timeout for HTTP requests in seconds.

# --- Helper Functions ---
def convert_bytes_to_human_readable(size_bytes):
    """Convert bytes to a human-readable string (KB, MB, GB, etc.)."""
    if size_bytes is None:
        return "N/A"
    try:
        size_bytes = int(size_bytes)
    except (ValueError, TypeError):
        return "N/A"

    if size_bytes == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(float(size_bytes), 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return f"{s} {size_name[i]}"

# --- Main Script ---
def get_gguf_model_catalog():
    final_catalog_data = []
    processed_hf_repo_ids = set() 

    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing catalog from {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_catalog = json.load(f)
                for entry in existing_catalog:
                    # Reconstruct the Hugging Face repo ID from the existing entry
                    hf_repo_id_from_existing = f"{entry.get('developer', 'unknown')}/{entry['model_name']}"
                    processed_hf_repo_ids.add(hf_repo_id_from_existing)
                final_catalog_data.extend(existing_catalog)
                print(f"Loaded {len(existing_catalog)} existing models into catalog ({len(processed_hf_repo_ids)} unique HF repo IDs).")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error loading existing catalog: {e}. Starting with an empty catalog.")
            final_catalog_data = []
            processed_hf_repo_ids = set() # Reset in case of error

    offset = 0
    newly_added_models_count = 0
    page_num = 0
    list_url = f"{HF_BASE_API_URL}/models" # Initialize list_url here

    print("Starting to fetch new GGUF models from Hugging Face Hub...")

    while page_num < 1:
        try:
            params = {
                'tags': 'gguf',
                'limit': MODELS_PER_PAGE,
                'offset': offset
            }
            print(f"\n  Requesting page {page_num + 1} (offset: {offset})...")
            response = requests.get(list_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            current_page_models_summaries = response.json()
            if not current_page_models_summaries:
                print("  No more GGUF models found on new pages or end of pages reached.")
                break # Exit loop if no models are returned

            print(f"  Found {len(current_page_models_summaries)} models on this page.")

            for model_summary in current_page_models_summaries:
                hf_repo_id = model_summary.get('id') 
                if not hf_repo_id:
                    print(f"    Skipping a model with missing 'id' in summary: {model_summary}")
                    continue

                if hf_repo_id in processed_hf_repo_ids:
                    # print(f"    Skipping '{hf_repo_id}' as it's already in the catalog.") # Can uncomment for verbose debug
                    continue # Skip to the next model summary

                model_name = hf_repo_id.split('/')[-1]
                developer = model_summary.get('author', hf_repo_id.split('/')[0])
                downloads = model_summary.get('downloads', 0)

                detail_url = f"{HF_BASE_API_URL}/models/{hf_repo_id}?blobs=true"
                time.sleep(REQUEST_DELAY) # Pause before making next request
                try:
                    detail_response = requests.get(detail_url, timeout=REQUEST_TIMEOUT)
                    detail_response.raise_for_status()
                    model_detail = detail_response.json()

                    current_model_quants = []
                    description = "No detailed description available." # Placeholder as explained

                    if 'siblings' in model_detail:
                        for sibling in model_detail['siblings']:
                            rfilename = sibling.get('rfilename')
                            if rfilename and rfilename.lower().endswith('.gguf'):
                                file_size = sibling.get('size')
                                quant_path = f"https://huggingface.co/{hf_repo_id}/resolve/main/{rfilename}"
                                current_model_quants.append({
                                    "model_id": rfilename, # This is the GGUF filename (e.g., "llama-3.1-8b-instruct-q2_k.gguf")
                                    "path": quant_path,
                                    "file_size": convert_bytes_to_human_readable(file_size)
                                })
                    if current_model_quants:
                        catalog_entry = {
                            "model_name": model_name,
                            "description": description,
                            "developer": developer,
                            "downloads": downloads,
                            "num_quants": len(current_model_quants),
                            "quants": current_model_quants
                        }
                        final_catalog_data.append(catalog_entry)
                        processed_hf_repo_ids.add(hf_repo_id) # Mark this HF repo ID as processed
                        newly_added_models_count += 1
                        print(f"    Added NEW '{model_name}' (from '{hf_repo_id}') with {len(current_model_quants)} GGUF quants.")
                    else:
                        processed_hf_repo_ids.add(hf_repo_id) 
                        # print(f"    Skipping '{hf_repo_id}': GGUF tag but no GGUF files found in detail.")


                except requests.exceptions.Timeout:
                    print(f"      Timeout fetching details for {hf_repo_id}. Skipping this model.")
                    processed_hf_repo_ids.add(hf_repo_id) # Avoid re-trying this specific repo in this run
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"      Error fetching details for {hf_repo_id}: {e}. Skipping this model.")
                    processed_hf_repo_ids.add(hf_repo_id) # Avoid re-trying this specific repo in this run
                    continue
                except json.JSONDecodeError:
                    print(f"      Error decoding JSON for {hf_repo_id} details. Skipping this model.")
                    processed_hf_repo_ids.add(hf_repo_id) # Avoid re-trying this specific repo in this run
                    continue
            offset += MODELS_PER_PAGE # Move to the next page
            page_num += 1
            if MAX_PAGES is not None and page_num >= MAX_PAGES:
                print(f"  Reached maximum number of pages ({MAX_PAGES}). Stopping.")
                break

            time.sleep(REQUEST_DELAY) # Pause before fetching the next page

        except requests.exceptions.Timeout:
            print(f"Error: Timeout fetching models list from {list_url}. Exiting.")
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models list from {list_url}: {e}. Exiting.")
            break
        except json.JSONDecodeError:
            print(f"Error decoding JSON for models list from {list_url}. Exiting.")
            break

    print(f"\nFinished fetching. Added {newly_added_models_count} new GGUF models this run.")
    print(f"Total models in catalog: {len(final_catalog_data)}")

    final_catalog_data_sorted = sorted(final_catalog_data, key=lambda x: x['model_name'].lower())

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_catalog_data_sorted, f, indent=2, ensure_ascii=False)
        print(f"Catalog successfully written to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing to file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    get_gguf_model_catalog()
