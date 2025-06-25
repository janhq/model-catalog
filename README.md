# model-catalog

**Repository:** model-catalog
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
    "repo_id": "Menlo/Jan-nano-gguf",
    "file_name": "jan-nano-4b-iQ4_XS.gguf",
    "url": "https://huggingface.co/Menlo/Jan-nano-gguf/resolve/main/jan-nano-4b-iQ4_XS.gguf?download=true"
  }
]
```
