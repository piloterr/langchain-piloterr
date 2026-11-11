"""Piloterr tools for LangChain agents: Google search and website scraping."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SecretStr

from langchain_piloterr import _client


class _PiloterrTool(BaseTool):
    handle_tool_error: bool = True
    api_key: SecretStr | None = Field(default=None, description="Piloterr API key. Defaults to PILOTERR_API_KEY.")
    base_url: str = _client.DEFAULT_BASE_URL
    timeout: float = _client.DEFAULT_TIMEOUT

    def _key(self) -> str:
        return _client.resolve_api_key(self.api_key.get_secret_value() if self.api_key else None)


class PiloterrGoogleSearchInput(BaseModel):
    """Arguments the model can set."""

    query: str = Field(description="The Google search query. Operators such as site: are supported.")


class PiloterrGoogleSearch(_PiloterrTool):
    """Search Google through Piloterr and return the organic results.

    Setup:
        .. code-block:: bash

            pip install -U langchain-piloterr
            export PILOTERR_API_KEY="your-api-key"

    Invocation:
        .. code-block:: python

            from langchain_piloterr import PiloterrGoogleSearch

            PiloterrGoogleSearch(max_results=5).invoke({"query": "langchain"})
    """

    name: str = "piloterr_google_search"
    description: str = (
        "Search Google through Piloterr and get the top organic results with title, url and snippet. "
        "Input is a search query."
    )
    args_schema: type[BaseModel] = PiloterrGoogleSearchInput
    timeout: float = 60.0

    max_results: int = Field(default=10, ge=1, le=100)
    """Number of results requested from Google (``num``)."""
    country: str | None = None
    """Two-letter country code to search from, such as us."""
    language: str | None = None
    """Language code of the results, optionally with a region, such as en or en-gb."""
    location: str | None = None
    """City or region to search from, such as "Paris, France"."""

    def _params(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ToolException("The search query is empty.")
        params: dict[str, Any] = {"query": query, "num": self.max_results}
        if self.country:
            params["gl"] = self.country.strip().lower()
        if self.language:
            params["hl"] = self.language.strip()
        if self.location:
            params["location"] = self.location.strip()
        return params

    def _output(self, query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"query": query, "results": [_client.to_result(row) for row in rows[: self.max_results]]}

    def _run(self, query: str, run_manager: CallbackManagerForToolRun | None = None) -> dict[str, Any]:
        params = self._params(query)
        try:
            response = _client.get(
                "/v2/google/search", params, api_key=self._key(), base_url=self.base_url, timeout=self.timeout
            )
            return self._output(params["query"], _client.organic_rows(response))
        except _client.PiloterrAPIError as exc:
            raise ToolException(str(exc)) from exc

    async def _arun(self, query: str, run_manager: AsyncCallbackManagerForToolRun | None = None) -> dict[str, Any]:
        params = self._params(query)
        try:
            response = await _client.aget(
                "/v2/google/search", params, api_key=self._key(), base_url=self.base_url, timeout=self.timeout
            )
            return self._output(params["query"], _client.organic_rows(response))
        except _client.PiloterrAPIError as exc:
            raise ToolException(str(exc)) from exc


class PiloterrWebsiteScraperInput(BaseModel):
    """Arguments the model can set."""

    url: str = Field(description="The absolute http(s) URL of the page to read.")


class PiloterrWebsiteScraper(_PiloterrTool):
    """Fetch a web page through Piloterr and return its readable text.

    Piloterr downloads the page on its own infrastructure: with a plain crawler
    (``engine="crawler"``), a headless browser for JavaScript pages
    (``engine="rendering"``), or a web unlocker for pages behind anti-bot
    protection (``engine="webunlocker"``).

    Invocation:
        .. code-block:: python

            from langchain_piloterr import PiloterrWebsiteScraper

            PiloterrWebsiteScraper(engine="rendering").invoke({"url": "https://example.com"})
    """

    name: str = "piloterr_website_scraper"
    description: str = (
        "Read a web page through Piloterr and get its title and readable text. Input is the absolute URL of the page."
    )
    args_schema: type[BaseModel] = PiloterrWebsiteScraperInput

    engine: _client.Engine = "crawler"
    wait_in_seconds: int | None = Field(default=None, ge=0, le=30)
    """Rendering only: extra seconds to wait after the page loads."""
    max_characters: int | None = Field(default=20_000, ge=1)
    """Truncate the returned text to this many characters; ``None`` keeps it all."""

    def _output(self, url: str, html: str) -> str:
        title, text = _client.html_to_text(html)
        if self.max_characters is not None:
            text = text[: self.max_characters]
        return f"# {title or url}\n\n{text}" if text else f"# {title or url}\n\n(The page has no readable text.)"

    def _run(self, url: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            path, params = _client.website_params(url, self.engine, self.wait_in_seconds)
            response = _client.get(path, params, api_key=self._key(), base_url=self.base_url, timeout=self.timeout)
            return self._output(params["query"], _client.page_html(response))
        except _client.PiloterrAPIError as exc:
            raise ToolException(str(exc)) from exc

    async def _arun(self, url: str, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        try:
            path, params = _client.website_params(url, self.engine, self.wait_in_seconds)
            response = await _client.aget(
                path, params, api_key=self._key(), base_url=self.base_url, timeout=self.timeout
            )
            return self._output(params["query"], _client.page_html(response))
        except _client.PiloterrAPIError as exc:
            raise ToolException(str(exc)) from exc
