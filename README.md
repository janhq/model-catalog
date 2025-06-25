# model-catalog

**Purpose:** A curated catalog of Large Language Models (LLMs) sourced from the Hugging Face Hub, specifically designed for integration with the Jan application's model hub.  This catalog provides a database for popular HF repositories and the models hosted by them.

## Introduction

This repository contains a self updating using a workflow of a structured data format (primarily JSON) that serves as a centralized inventory of LLMs available on some of the popular HF repositories tracked by us. It's built to streamline the process of selecting and deploying LLMs within Jan, reducing friction and improving discoverability.

## Tracked repos

[TBD]


## Data Structure

The core of the catalog is a JSON file named `models.json`. This file contains a list of LLM objects, each with the following key properties:

```json
[
{
  "model_name": "Jan-Nano-Gguf",
  "description": "Jan Nano is a fine-tuned language model built on top of the Qwen3 architecture. Developed as part of the Jan ecosystem, it balances compact size and extended context length, making it ide...",
  "developer": "Menlo",
  "downloads": 19148,
  "num_quants": 14,
  "quants": [
    {
        "model_id": "jan-nano-4b-iQ4_XS.gguf",
        "path": "https://huggingface.co/Menlo/Jan-nano-gguf/resolve/main/jan-nano-4b-iQ4_XS.gguf?download=true"
    }
  ]
}
]
```
