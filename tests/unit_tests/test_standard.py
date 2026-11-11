"""LangChain standard unit tests for both tools."""

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_piloterr import PiloterrGoogleSearch, PiloterrWebsiteScraper

KEY = "test-piloterr-key-placeholder"  # pragma: allowlist secret


class TestPiloterrGoogleSearchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[PiloterrGoogleSearch]:
        return PiloterrGoogleSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": KEY}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "web scraping api"}


class TestPiloterrWebsiteScraperStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[PiloterrWebsiteScraper]:
        return PiloterrWebsiteScraper

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": KEY}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"url": "https://example.com"}
