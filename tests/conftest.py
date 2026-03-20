"""
Shared pytest configuration.

Patches external clients (OpenAI, HF token) at import time so that
importing the catalog scripts never triggers real API calls.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path so we can import the scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock environment variables before any script imports
os.environ.setdefault("BASE_URL", "http://fake-llm-api")
os.environ.setdefault("API_KEY", "fake-key")
os.environ.setdefault("HF_TOKEN", "fake-hf-token")

# Patch the OpenAI client creation so imports don't fail
_mock_client = MagicMock()
_mock_choice = MagicMock()
_mock_choice.message.content = "A test model description."
_mock_client.chat.completions.create.return_value = MagicMock(
    choices=[_mock_choice]
)

# Patch at the openai module level before scripts are imported
with patch("openai.OpenAI", return_value=_mock_client):
    pass
