"""Load web pages through Piloterr as LangChain documents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_piloterr import _client


class PiloterrLoader(BaseLoader):
    """Fetch web pages through Piloterr and load them as documents.

    Each document's ``page_content`` is the readable text of the page; its
    metadata holds ``source`` (the URL), ``title`` and ``engine``. Set
    ``include_html=True`` to also keep the raw HTML in ``metadata["html"]``.

    Example:
        .. code-block:: python

            from langchain_piloterr import PiloterrLoader

            loader = PiloterrLoader(["https://example.com"], engine="rendering")
            docs = loader.load()
    """

    def __init__(
        self,
        urls: str | Sequence[str],
        *,
        api_key: str | None = None,
        engine: _client.Engine = "crawler",
        wait_in_seconds: int | None = None,
        include_html: bool = False,
        continue_on_failure: bool = False,
        base_url: str = _client.DEFAULT_BASE_URL,
        timeout: float = _client.DEFAULT_TIMEOUT,
    ) -> None:
        """
        Args:
            urls: One URL or a list of URLs to load.
            api_key: Piloterr API key. Defaults to ``PILOTERR_API_KEY``.
            engine: ``crawler``, ``rendering`` (JavaScript pages) or ``webunlocker`` (protected pages).
            wait_in_seconds: Rendering only: extra seconds to wait after the page loads, up to 30.
            include_html: Keep the raw HTML in ``metadata["html"]``.
            continue_on_failure: Skip pages that fail instead of raising.
            base_url: API base URL.
            timeout: Seconds to wait for each page.
        """
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.api_key = _client.resolve_api_key(api_key)
        self.engine = engine
        self.wait_in_seconds = wait_in_seconds
        self.include_html = include_html
        self.continue_on_failure = continue_on_failure
        self.base_url = base_url
        self.timeout = timeout

    def _document(self, url: str, html: str) -> Document:
        title, text = _client.html_to_text(html)
        metadata = {"source": url, "title": title, "engine": self.engine}
        if self.include_html:
            metadata["html"] = html
        return Document(page_content=text, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        for url in self.urls:
            try:
                path, params = _client.website_params(url, self.engine, self.wait_in_seconds)
                response = _client.get(path, params, api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
                yield self._document(params["query"], _client.page_html(response))
            except _client.PiloterrAPIError:
                if not self.continue_on_failure:
                    raise

    async def alazy_load(self) -> AsyncIterator[Document]:
        for url in self.urls:
            try:
                path, params = _client.website_params(url, self.engine, self.wait_in_seconds)
                response = await _client.aget(
                    path, params, api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
                )
                yield self._document(params["query"], _client.page_html(response))
            except _client.PiloterrAPIError:
                if not self.continue_on_failure:
                    raise
