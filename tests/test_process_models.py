"""
Tests for model processing functions across all catalog scripts.

These tests mock external API calls (HuggingFace, OpenAI) and verify
the model processing logic produces correct output structure.

After refactoring, process_gguf_model and process_mlx_model live in
catalog_helpers and are imported by both v2 and fetch. We test them
through catalog_helpers directly.
"""

import pytest
from unittest.mock import patch, MagicMock

import prepare_catalog as v1
import catalog_helpers as helpers
import prepare_catalog_v2 as v2


# ---------------------------------------------------------------------------
# Fixtures: realistic HuggingFace API responses
# ---------------------------------------------------------------------------
@pytest.fixture
def gguf_detail():
    """A realistic HF API detail response for a GGUF model."""
    return {
        "id": "janhq/test-model-gguf",
        "downloads": 50000,
        "createdAt": "2026-01-15T10:00:00.000Z",
        "library_name": "gguf",
        "tags": ["text-generation", "gguf"],
        "gguf": {
            "chat_template": "some template for tool calling"
        },
        "siblings": [
            {"rfilename": "test-model-Q4_K_M.gguf", "size": 2500000000},
            {"rfilename": "test-model-Q5_K_M.gguf", "size": 3000000000},
            {"rfilename": "test-model-Q8_0.gguf", "size": 4400000000},
            {"rfilename": "README.md", "size": 5000},
        ],
    }


@pytest.fixture
def gguf_detail_with_mmproj():
    """A GGUF model with mmproj files (vision model)."""
    return {
        "id": "janhq/test-vl-model-gguf",
        "downloads": 30000,
        "createdAt": "2026-02-01T10:00:00.000Z",
        "library_name": "gguf",
        "tags": ["image-text-to-text", "gguf"],
        "gguf": {},
        "siblings": [
            {"rfilename": "test-vl-Q4_K_M.gguf", "size": 2500000000},
            {"rfilename": "mmproj-test-vl-f16.gguf", "size": 600000000},
            {"rfilename": "README.md", "size": 3000},
        ],
    }


@pytest.fixture
def gguf_detail_with_multipart():
    """A GGUF model with multipart files that should be skipped."""
    return {
        "id": "someone/big-model-gguf",
        "downloads": 10000,
        "createdAt": "2026-01-10T10:00:00.000Z",
        "library_name": "gguf",
        "tags": ["text-generation"],
        "gguf": {},
        "siblings": [
            {"rfilename": "big-model-Q4_K_M.gguf", "size": 2000000000},
            {"rfilename": "big-model-Q8_0-00001-of-00003.gguf", "size": 5000000000},
            {"rfilename": "big-model-Q8_0-00002-of-00003.gguf", "size": 5000000000},
            {"rfilename": "big-model-Q8_0-00003-of-00003.gguf", "size": 5000000000},
            {"rfilename": "README.md", "size": 2000},
        ],
    }


@pytest.fixture
def gguf_detail_only_excluded():
    """A GGUF model with only embedding/ocr/etc files (should be skipped)."""
    return {
        "id": "someone/embedding-model-gguf",
        "downloads": 5000,
        "createdAt": "2026-01-05T10:00:00.000Z",
        "library_name": "gguf",
        "tags": ["text-generation"],
        "gguf": {},
        "siblings": [
            {"rfilename": "embedding-model-f16.gguf", "size": 1000000000},
            {"rfilename": "reranker-model-f16.gguf", "size": 500000000},
        ],
    }


@pytest.fixture
def mlx_detail():
    """A realistic HF API detail response for an MLX model."""
    return {
        "id": "mlx-community/test-model-4bit",
        "downloads": 20000,
        "createdAt": "2026-01-20T10:00:00.000Z",
        "library_name": "mlx",
        "tags": ["mlx", "text-generation"],
        "siblings": [
            {"rfilename": "model.safetensors", "size": 2000000000},
            {"rfilename": "model-00001-of-00002.safetensors", "size": 1000000000},
            {"rfilename": "config.json", "size": 500},
            {"rfilename": "README.md", "size": 4000},
        ],
    }


@pytest.fixture
def gguf_detail_no_tools():
    """A GGUF model without tool support."""
    return {
        "id": "janhq/simple-model-gguf",
        "downloads": 15000,
        "createdAt": "2026-01-12T10:00:00.000Z",
        "library_name": "gguf",
        "tags": ["text-generation"],
        "gguf": {
            "chat_template": "basic chat template without tools"
        },
        "siblings": [
            {"rfilename": "simple-model-Q4_K_M.gguf", "size": 2000000000},
            {"rfilename": "README.md", "size": 2000},
        ],
    }


# ---------------------------------------------------------------------------
# process_gguf_model (catalog_helpers — used by both v2 and fetch)
# ---------------------------------------------------------------------------
class TestProcessGgufModel:
    """Tests for helpers.process_gguf_model (shared by v2 and fetch)."""

    @patch("catalog_helpers.requests.get")
    def test_basic_gguf_model(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Test Model\nA test model for testing."
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/test-model-gguf", gguf_detail)
        assert result is not None
        assert result["model_name"] == "test-model-gguf"
        assert result["developer"] == "janhq"
        assert result["downloads"] == 50000
        assert result["library_name"] == "gguf"
        assert result["tools"] is True  # "for tool" in chat_template
        assert result["num_quants"] == 3
        assert len(result["quants"]) == 3
        assert result["num_mmproj"] == 0
        assert result["readme"] is not None

    @patch("catalog_helpers.requests.get")
    def test_gguf_with_mmproj(self, mock_get, gguf_detail_with_mmproj):
        mock_resp = MagicMock()
        mock_resp.text = "# VL Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/test-vl-model-gguf", gguf_detail_with_mmproj)
        assert result is not None
        assert result["num_quants"] == 1
        assert result["num_mmproj"] == 1
        assert result["mmproj_models"][0]["model_id"] == "mmproj-test-vl-f16"

    @patch("catalog_helpers.requests.get")
    def test_multipart_files_skipped(self, mock_get, gguf_detail_with_multipart):
        mock_resp = MagicMock()
        mock_resp.text = "# Big Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("someone/big-model-gguf", gguf_detail_with_multipart)
        assert result is not None
        assert result["num_quants"] == 1
        assert result["quants"][0]["model_id"] == "big-model-Q4_K_M"

    @patch("catalog_helpers.requests.get")
    def test_excluded_files_skipped(self, mock_get, gguf_detail_only_excluded):
        mock_resp = MagicMock()
        mock_resp.text = "# Embedding Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("someone/embedding-model-gguf", gguf_detail_only_excluded)
        assert result is None

    @patch("catalog_helpers.requests.get")
    def test_no_tool_support(self, mock_get, gguf_detail_no_tools):
        mock_resp = MagicMock()
        mock_resp.text = "# Simple Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/simple-model-gguf", gguf_detail_no_tools)
        assert result is not None
        assert result["tools"] is False

    def test_blacklisted_developer(self, gguf_detail):
        result = helpers.process_gguf_model("TheBloke/some-model-gguf", gguf_detail)
        assert result is None

    def test_empty_detail_returns_none(self):
        assert helpers.process_gguf_model("x/y", {}) is None

    @patch("catalog_helpers.requests.get")
    def test_with_existing_entry_preserves_description(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        existing = {"description": "Existing desc", "createdAt": "2025-01-01T00:00:00.000Z"}
        result = helpers.process_gguf_model("janhq/test-model-gguf", gguf_detail, existing)
        assert result is not None
        assert result["description"] == "Existing desc"

    @patch("catalog_helpers.requests.get")
    def test_quant_file_size_formatting(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/test-model-gguf", gguf_detail)
        assert result is not None
        for quant in result["quants"]:
            assert "GB" in quant["file_size"] or "MB" in quant["file_size"]

    @patch("catalog_helpers.requests.get")
    def test_quant_path_format(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/test-model-gguf", gguf_detail)
        for quant in result["quants"]:
            assert quant["path"].startswith("https://huggingface.co/janhq/test-model-gguf/resolve/main/")
            assert quant["path"].endswith(".gguf")


# ---------------------------------------------------------------------------
# process_mlx_model (catalog_helpers — used by both v2 and fetch)
# ---------------------------------------------------------------------------
class TestProcessMlxModel:
    """Tests for helpers.process_mlx_model (shared by v2 and fetch)."""

    @patch("catalog_helpers.requests.get")
    def test_basic_mlx_model(self, mock_get, mlx_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# MLX Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_mlx_model("mlx-community/test-model-4bit", mlx_detail)
        assert result is not None
        assert result["model_name"] == "test-model-4bit"
        assert result["developer"] == "mlx-community"
        assert result["library_name"] == "mlx"
        assert result["tools"] is False
        assert result["num_safetensors"] == 2
        assert result["config"] is not None
        assert "config.json" in result["config"]

    @patch("catalog_helpers.requests.get")
    def test_mlx_no_safetensors_returns_none(self, mock_get):
        detail = {
            "id": "someone/model",
            "downloads": 100,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "siblings": [
                {"rfilename": "README.md", "size": 1000},
                {"rfilename": "config.json", "size": 500},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_mlx_model("someone/model", detail)
        assert result is None

    @patch("catalog_helpers.requests.get")
    def test_mlx_with_existing_entry_preserves_description(self, mock_get, mlx_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        existing = {"description": "Existing MLX desc"}
        result = helpers.process_mlx_model("mlx-community/test-model-4bit", mlx_detail, existing)
        assert result is not None
        assert result["description"] == "Existing MLX desc"


# ---------------------------------------------------------------------------
# process_model_details (v1 only — still standalone)
# ---------------------------------------------------------------------------
class TestProcessModelDetailsV1:
    """Tests for process_model_details in prepare_catalog.py (v1)."""

    @patch("requests.get")
    def test_basic_model(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Test Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = v1.process_model_details("janhq/test-model-gguf", gguf_detail)
        assert result is not None
        assert result["model_name"] == "test-model-gguf"
        assert result["developer"] == "janhq"
        assert result["downloads"] == 50000
        assert result["tools"] is True
        assert result["num_quants"] == 3
        assert result["num_mmproj"] == 0

    @patch("requests.get")
    def test_with_mmproj(self, mock_get, gguf_detail_with_mmproj):
        mock_resp = MagicMock()
        mock_resp.text = "# VL Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = v1.process_model_details("janhq/test-vl-model-gguf", gguf_detail_with_mmproj)
        assert result is not None
        assert result["num_quants"] == 1
        assert result["num_mmproj"] == 1

    def test_blacklisted_developer(self, gguf_detail):
        result = v1.process_model_details("TheBloke/some-model", gguf_detail)
        assert result is None

    def test_none_detail(self):
        assert v1.process_model_details("x/y", None) is None

    def test_empty_detail(self):
        assert v1.process_model_details("x/y", {}) is None

    @patch("requests.get")
    def test_existing_entry_preserves_description(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Test Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        existing = {"description": "Existing description", "createdAt": "2025-01-01T00:00:00.000Z"}
        result = v1.process_model_details("janhq/test-model-gguf", gguf_detail, existing)
        assert result is not None
        assert result["description"] == "Existing description"


# ---------------------------------------------------------------------------
# remove_duplicates (v2)
# ---------------------------------------------------------------------------
class TestRemoveDuplicatesV2:

    def test_no_duplicates(self):
        catalog = [
            {"model_name": "model-a", "developer": "dev1", "downloads": 100, "library_name": "gguf"},
            {"model_name": "model-b", "developer": "dev2", "downloads": 200, "library_name": "gguf"},
        ]
        result = v2.remove_duplicates(catalog)
        assert len(result) == 2

    def test_duplicate_keeps_pinned(self):
        catalog = [
            {"model_name": "Jan-v3-4B-base-instruct-gguf", "developer": "janhq", "downloads": 100, "library_name": "gguf"},
            {"model_name": "Jan-v3-4B-base-instruct-gguf", "developer": "someone", "downloads": 99999, "library_name": "gguf"},
        ]
        result = v2.remove_duplicates(catalog)
        assert len(result) == 1
        assert result[0]["developer"] == "janhq"

    def test_duplicate_keeps_higher_downloads(self):
        catalog = [
            {"model_name": "some-model", "developer": "dev1", "downloads": 100, "library_name": "gguf"},
            {"model_name": "some-model", "developer": "dev2", "downloads": 500, "library_name": "gguf"},
        ]
        result = v2.remove_duplicates(catalog)
        assert len(result) == 1
        assert result[0]["downloads"] == 500

    def test_keeps_both_gguf_and_mlx(self):
        catalog = [
            {"model_name": "some-model", "developer": "dev1", "downloads": 100, "library_name": "gguf"},
            {"model_name": "some-model", "developer": "dev2", "downloads": 200, "library_name": "mlx"},
        ]
        result = v2.remove_duplicates(catalog)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# remove_duplicates_and_multipart (v1)
# ---------------------------------------------------------------------------
class TestRemoveDuplicatesAndMultipartV1:

    def test_removes_empty_entries(self):
        catalog = [
            {"model_name": "empty", "developer": "dev", "quants": [], "mmproj_models": []},
        ]
        result = v1.remove_duplicates_and_multipart(catalog)
        assert len(result) == 0

    def test_keeps_valid_entries(self):
        catalog = [
            {
                "model_name": "valid",
                "developer": "dev",
                "downloads": 100,
                "quants": [{"path": "model-Q4.gguf", "model_id": "model-Q4", "file_size": "2 GB"}],
                "mmproj_models": [],
            },
        ]
        result = v1.remove_duplicates_and_multipart(catalog)
        assert len(result) == 1

    def test_filters_multipart_from_quants(self):
        catalog = [
            {
                "model_name": "model",
                "developer": "dev",
                "downloads": 100,
                "quants": [
                    {"path": "https://example.com/model-Q4.gguf", "model_id": "model-Q4", "file_size": "2 GB"},
                    {"path": "https://example.com/model-Q8-00001-of-00002.gguf", "model_id": "model-Q8-part1", "file_size": "5 GB"},
                ],
                "mmproj_models": [],
            },
        ]
        result = v1.remove_duplicates_and_multipart(catalog)
        assert len(result) == 1
        assert result[0]["num_quants"] == 1
        assert result[0]["quants"][0]["model_id"] == "model-Q4"

    def test_duplicate_keeps_pinned(self):
        catalog = [
            {
                "model_name": "Jan-v3-4B-base-instruct-gguf",
                "developer": "janhq",
                "downloads": 100,
                "quants": [{"path": "q.gguf", "model_id": "q", "file_size": "1 GB"}],
                "mmproj_models": [],
            },
            {
                "model_name": "Jan-v3-4B-base-instruct-gguf",
                "developer": "someone",
                "downloads": 99999,
                "quants": [{"path": "q.gguf", "model_id": "q", "file_size": "1 GB"}],
                "mmproj_models": [],
            },
        ]
        result = v1.remove_duplicates_and_multipart(catalog)
        assert len(result) == 1
        assert result[0]["developer"] == "janhq"


# ---------------------------------------------------------------------------
# Output schema validation
# ---------------------------------------------------------------------------
class TestOutputSchema:
    """Verify that processed model entries have all required keys."""

    GGUF_REQUIRED_KEYS = {
        "model_name", "developer", "downloads", "createdAt",
        "library_name", "tools", "num_quants", "quants",
        "num_mmproj", "mmproj_models", "readme", "description",
    }

    MLX_REQUIRED_KEYS = {
        "model_name", "developer", "downloads", "createdAt",
        "library_name", "tools", "num_safetensors", "safetensors_files",
        "config", "readme", "description",
    }

    V1_REQUIRED_KEYS = {
        "model_name", "developer", "downloads", "createdAt",
        "tools", "num_quants", "quants",
        "num_mmproj", "mmproj_models", "readme", "description",
    }

    @patch("catalog_helpers.requests.get")
    def test_gguf_output_has_all_keys(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_gguf_model("janhq/test-model-gguf", gguf_detail)
        assert result is not None
        missing = self.GGUF_REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in GGUF output: {missing}"

    @patch("catalog_helpers.requests.get")
    def test_mlx_output_has_all_keys(self, mock_get, mlx_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = helpers.process_mlx_model("mlx-community/test-model-4bit", mlx_detail)
        assert result is not None
        missing = self.MLX_REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in MLX output: {missing}"

    @patch("requests.get")
    def test_v1_output_has_all_keys(self, mock_get, gguf_detail):
        mock_resp = MagicMock()
        mock_resp.text = "# Model"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = v1.process_model_details("janhq/test-model-gguf", gguf_detail)
        assert result is not None
        missing = self.V1_REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in v1 output: {missing}"
