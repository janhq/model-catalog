"""
Tests for shared helper functions across all catalog scripts.

Tests cover:
- catalog_helpers.py (the shared module used by v2 and fetch)
- prepare_catalog.py (v1, still standalone)

When v2/fetch import from catalog_helpers, we verify them through
catalog_helpers directly AND through the re-exported references.
"""

import pytest

import prepare_catalog as v1
import catalog_helpers as helpers


# ---------------------------------------------------------------------------
# convert_bytes_to_human_readable
# ---------------------------------------------------------------------------
class TestConvertBytesToHumanReadable:

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_none_returns_na(self, mod):
        assert mod.convert_bytes_to_human_readable(None) == "N/A"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_invalid_string_returns_na(self, mod):
        assert mod.convert_bytes_to_human_readable("not_a_number") == "N/A"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_zero_returns_zero_b(self, mod):
        assert mod.convert_bytes_to_human_readable(0) == "0 B"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_bytes(self, mod):
        assert mod.convert_bytes_to_human_readable(500) == "500.0 B"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_kilobytes(self, mod):
        assert mod.convert_bytes_to_human_readable(1024) == "1.0 KB"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_megabytes(self, mod):
        assert mod.convert_bytes_to_human_readable(1048576) == "1.0 MB"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_gigabytes(self, mod):
        result = mod.convert_bytes_to_human_readable(2_365_587_456)
        assert "GB" in result
        assert result == "2.2 GB"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_terabytes(self, mod):
        result = mod.convert_bytes_to_human_readable(1099511627776)
        assert result == "1.0 TB"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_string_number(self, mod):
        assert mod.convert_bytes_to_human_readable("1024") == "1.0 KB"

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_float_input(self, mod):
        assert mod.convert_bytes_to_human_readable(1024.9) == "1.0 KB"


# ---------------------------------------------------------------------------
# is_multipart_gguf
# ---------------------------------------------------------------------------
class TestIsMultipartGguf:

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_multipart_pattern(self, mod):
        assert mod.is_multipart_gguf("model-Q4_0-00001-of-00003.gguf") is True

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_multipart_two_digit(self, mod):
        assert mod.is_multipart_gguf("llama-3-70b-Q5_K_M-02-of-05.gguf") is True

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_regular_gguf_not_multipart(self, mod):
        assert mod.is_multipart_gguf("model-Q4_K_M.gguf") is False

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_non_gguf_not_multipart(self, mod):
        assert mod.is_multipart_gguf("model.safetensors") is False

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_case_insensitive(self, mod):
        assert mod.is_multipart_gguf("Model-Q4-1-of-2.GGUF") is True

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_empty_string(self, mod):
        assert mod.is_multipart_gguf("") is False


# ---------------------------------------------------------------------------
# is_mmproj_file
# ---------------------------------------------------------------------------
class TestIsMmprojFile:

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_mmproj_detected(self, mod):
        assert mod.is_mmproj_file("mmproj-model-f16.gguf") is True

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_mmproj_uppercase(self, mod):
        assert mod.is_mmproj_file("MMPROJ-model.gguf") is True

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_regular_model_not_mmproj(self, mod):
        assert mod.is_mmproj_file("model-Q4_K_M.gguf") is False

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_mmproj_without_gguf_extension(self, mod):
        assert mod.is_mmproj_file("mmproj-model.bin") is False

    @pytest.mark.parametrize("mod", [v1, helpers])
    def test_empty_string(self, mod):
        assert mod.is_mmproj_file("") is False


# ---------------------------------------------------------------------------
# detect_library_name (helpers only, v1 doesn't have it)
# ---------------------------------------------------------------------------
class TestDetectLibraryName:

    def test_gguf_from_library_name(self):
        assert helpers.detect_library_name({"library_name": "gguf"}) == "gguf"

    def test_mlx_from_library_name(self):
        assert helpers.detect_library_name({"library_name": "mlx"}) == "mlx"

    def test_gguf_from_tags(self):
        detail = {"library_name": "", "tags": ["gguf", "text-generation"]}
        assert helpers.detect_library_name(detail) == "gguf"

    def test_mlx_from_tags(self):
        detail = {"library_name": "", "tags": ["mlx", "conversational"]}
        assert helpers.detect_library_name(detail) == "mlx"

    def test_gguf_from_siblings(self):
        detail = {
            "library_name": "",
            "tags": [],
            "siblings": [{"rfilename": "model-Q4.gguf"}],
        }
        assert helpers.detect_library_name(detail) == "gguf"

    def test_mlx_from_siblings(self):
        detail = {
            "library_name": "",
            "tags": [],
            "siblings": [
                {"rfilename": "model.safetensors"},
                {"rfilename": "mlx_config.json"},
            ],
        }
        assert helpers.detect_library_name(detail) == "mlx"

    def test_unknown_fallback(self):
        detail = {"library_name": "", "tags": [], "siblings": []}
        assert helpers.detect_library_name(detail) == "unknown"

    def test_empty_detail(self):
        assert helpers.detect_library_name({}) == "unknown"

    def test_case_insensitive_library_name(self):
        assert helpers.detect_library_name({"library_name": "GGUF"}) == "gguf"

    def test_case_insensitive_tags(self):
        detail = {"library_name": "", "tags": ["MLX"]}
        assert helpers.detect_library_name(detail) == "mlx"

    def test_transformers_fallback(self):
        detail = {"library_name": "transformers", "tags": [], "siblings": []}
        assert helpers.detect_library_name(detail) == "transformers"


# ---------------------------------------------------------------------------
# Verify v2 and fetch use the same functions from catalog_helpers
# ---------------------------------------------------------------------------
class TestSharedImports:
    """Verify that v2 and fetch actually use the catalog_helpers functions."""

    def test_fetch_uses_helpers_process_gguf(self):
        import fetch_newest_jan_model as fetch
        assert fetch.process_gguf_model is helpers.process_gguf_model

    def test_fetch_uses_helpers_process_mlx(self):
        import fetch_newest_jan_model as fetch
        assert fetch.process_mlx_model is helpers.process_mlx_model

    def test_fetch_uses_helpers_detect_library(self):
        import fetch_newest_jan_model as fetch
        assert fetch.detect_library_name is helpers.detect_library_name

    def test_v2_uses_helpers_process_gguf(self):
        import prepare_catalog_v2 as v2
        assert v2.process_gguf_model is helpers.process_gguf_model

    def test_v2_uses_helpers_process_mlx(self):
        import prepare_catalog_v2 as v2
        assert v2.process_mlx_model is helpers.process_mlx_model
