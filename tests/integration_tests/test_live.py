"""Live tests against api.piloterr.com. They run only when PILOTERR_API_KEY is set."""

import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_piloterr import PiloterrGoogleSearch, PiloterrLoader, PiloterrWebsiteScraper

pytestmark = pytest.mark.skipif(not os.environ.get("PILOTERR_API_KEY"), reason="PILOTERR_API_KEY is not set")


class TestPiloterrGoogleSearchLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[PiloterrGoogleSearch]:
        return PiloterrGoogleSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "langchain"}


def test_scraper_reads_a_page() -> None:
    out = PiloterrWebsiteScraper().invoke({"url": "https://example.com"})
    assert "Example Domain" in out


def test_loader_loads_a_page() -> None:
    docs = PiloterrLoader(["https://example.com"]).load()
    assert docs and "Example Domain" in docs[0].page_content
