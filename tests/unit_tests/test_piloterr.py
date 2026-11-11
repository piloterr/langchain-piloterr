"""Offline tests: every request goes through an httpx.MockTransport."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from langchain_piloterr import (
    PiloterrAPIError,
    PiloterrGoogleSearch,
    PiloterrLoader,
    PiloterrWebsiteScraper,
    _client,
)

API_KEY = "test-piloterr-key-placeholder"  # pragma: allowlist secret
ROWS = [
    {"position": 1, "title": "LangChain", "link": "https://www.langchain.com/", "snippet": "Build agents."},
    {"position": 2, "title": "Only a title", "link": "https://example.com/"},
    "not an object",
]
PAGE = (
    "<!doctype html><html><head><title>Example  Domain</title><style>p{color:red}</style></head>"
    "<body><script>var x = 1;</script><h1>Example Domain</h1><p>For use in examples.</p>"
    "<ul><li>One</li><li>Two &amp; three</li></ul></body></html>"
)
PAGE_TEXT = "Example Domain\nFor use in examples.\nOne\nTwo & three"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[httpx.Request]]:
    def install(handler: Handler) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        transport = httpx.MockTransport(record)
        real_client, real_async = httpx.Client, httpx.AsyncClient
        monkeypatch.setattr(_client.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
        monkeypatch.setattr(_client.httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))
        return seen

    return install


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PILOTERR_API_KEY", raising=False)


def html(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=PAGE)


class TestGoogleSearch:
    def test_invoke(self, mock_http) -> None:
        seen = mock_http(lambda request: httpx.Response(200, json={"organic_results": ROWS}))
        out = PiloterrGoogleSearch(api_key=API_KEY, max_results=5, country="US", language="en-gb").invoke(
            {"query": " q "}
        )

        request = seen[0]
        assert request.url.path == "/v2/google/search"
        assert dict(request.url.params) == {"query": "q", "num": "5", "gl": "us", "hl": "en-gb"}
        assert request.headers["x-api-key"] == API_KEY
        assert out["results"] == [
            {"title": "LangChain", "url": "https://www.langchain.com/", "content": "Build agents.", "position": 1},
            {"title": "Only a title", "url": "https://example.com/", "content": "", "position": 2},
        ]

    def test_location(self, mock_http) -> None:
        seen = mock_http(lambda request: httpx.Response(200, json={}))
        out = PiloterrGoogleSearch(api_key=API_KEY, location="Paris, France").invoke({"query": "q"})

        assert seen[0].url.params["location"] == "Paris, France"
        assert out["results"] == []

    def test_empty_query(self, mock_http) -> None:
        seen = mock_http(html)
        assert PiloterrGoogleSearch(api_key=API_KEY).invoke({"query": " "}) == "The search query is empty."
        assert seen == []

    def test_missing_key(self, mock_http) -> None:
        seen = mock_http(html)
        assert "PILOTERR_API_KEY" in PiloterrGoogleSearch().invoke({"query": "q"})
        assert seen == []

    def test_env_key(self, mock_http, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PILOTERR_API_KEY", f"{API_KEY}\n")
        seen = mock_http(lambda request: httpx.Response(200, json={}))
        PiloterrGoogleSearch().invoke({"query": "q"})
        assert seen[0].headers["x-api-key"] == API_KEY

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (401, {"error": "Invalid API key"}, "Piloterr rejected the API key (HTTP 401): Invalid API key"),
            (
                401,
                {"error": {"code": "x", "message": "Rate limit exceeded"}},
                "Piloterr rate limit exceeded (HTTP 401): Rate limit exceeded",
            ),
            (402, {"error": "No credits"}, "Piloterr account is out of credits (HTTP 402): No credits"),
            (500, {"message": f"boom {API_KEY}"}, "Piloterr API error (HTTP 500): boom ***"),
        ],
    )
    def test_errors(self, mock_http, status, body, expected) -> None:
        mock_http(lambda request: httpx.Response(status, json=body))
        assert PiloterrGoogleSearch(api_key=API_KEY).invoke({"query": "q"}) == expected

    def test_not_json(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(200, text="<html>"))
        assert "not JSON" in PiloterrGoogleSearch(api_key=API_KEY).invoke({"query": "q"})

    async def test_ainvoke(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(200, json={"organic_results": ROWS}))
        out = await PiloterrGoogleSearch(api_key=API_KEY).ainvoke({"query": "q"})
        assert len(out["results"]) == 2


class TestWebsiteScraper:
    def test_crawler_returns_title_and_text(self, mock_http) -> None:
        seen = mock_http(html)
        out = PiloterrWebsiteScraper(api_key=API_KEY).invoke({"url": " https://example.com/ "})

        assert seen[0].url.path == "/v2/website/crawler"
        assert dict(seen[0].url.params) == {"query": "https://example.com/"}
        assert out == f"# Example Domain\n\n{PAGE_TEXT}"

    def test_raw_html_body(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(200, text=PAGE))
        assert (
            PiloterrWebsiteScraper(api_key=API_KEY)
            .invoke({"url": "https://example.com"})
            .startswith("# Example Domain")
        )

    def test_rendering_with_wait(self, mock_http) -> None:
        seen = mock_http(html)
        PiloterrWebsiteScraper(api_key=API_KEY, engine="rendering", wait_in_seconds=5).invoke(
            {"url": "https://example.com"}
        )
        assert seen[0].url.path == "/v2/website/rendering"
        assert seen[0].url.params["wait_in_seconds"] == "5"

    def test_wait_only_for_rendering(self, mock_http) -> None:
        seen = mock_http(html)
        PiloterrWebsiteScraper(api_key=API_KEY, engine="webunlocker", wait_in_seconds=5).invoke(
            {"url": "https://example.com"}
        )
        assert seen[0].url.path == "/v2/website/webunlocker"
        assert "wait_in_seconds" not in seen[0].url.params

    def test_truncation(self, mock_http) -> None:
        mock_http(html)
        out = PiloterrWebsiteScraper(api_key=API_KEY, max_characters=7).invoke({"url": "https://example.com"})
        assert out == "# Example Domain\n\nExample"

    def test_page_without_text(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(200, json="<html><body><script>x</script></body></html>"))
        out = PiloterrWebsiteScraper(api_key=API_KEY).invoke({"url": "https://example.com"})
        assert out == "# https://example.com\n\n(The page has no readable text.)"

    @pytest.mark.parametrize("url", ["", "example.com", "file:///etc/passwd", "ftp://example.com/x"])
    def test_rejects_non_http_urls(self, mock_http, url: str) -> None:
        seen = mock_http(html)
        assert "absolute http(s) URL" in PiloterrWebsiteScraper(api_key=API_KEY).invoke({"url": url})
        assert seen == []

    @pytest.mark.parametrize("body", [{"html": "x"}, ["x"], 3])
    def test_non_string_json(self, mock_http, body) -> None:
        mock_http(lambda request: httpx.Response(200, json=body))
        assert "expected the page HTML" in PiloterrWebsiteScraper(api_key=API_KEY).invoke({"url": "https://a.b"})

    def test_timeout(self, mock_http) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        mock_http(slow)
        out = PiloterrWebsiteScraper(api_key=API_KEY, timeout=3).invoke({"url": "https://a.b"})
        assert out == "Piloterr did not answer within 3 seconds."

    def test_network_error_without_the_key(self, mock_http) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"refused {API_KEY}", request=request)

        mock_http(down)
        out = PiloterrWebsiteScraper(api_key=API_KEY).invoke({"url": "https://a.b"})
        assert out.startswith("Could not reach Piloterr") and API_KEY not in out

    async def test_ainvoke(self, mock_http) -> None:
        mock_http(html)
        out = await PiloterrWebsiteScraper(api_key=API_KEY).ainvoke({"url": "https://example.com"})
        assert PAGE_TEXT in out


class TestLoader:
    def test_load(self, mock_http) -> None:
        mock_http(html)
        docs = PiloterrLoader(["https://a.example/", "https://b.example/"], api_key=API_KEY).load()

        assert [d.metadata["source"] for d in docs] == ["https://a.example/", "https://b.example/"]
        assert docs[0].page_content == PAGE_TEXT
        assert docs[0].metadata == {"source": "https://a.example/", "title": "Example Domain", "engine": "crawler"}

    def test_include_html_and_single_url(self, mock_http) -> None:
        mock_http(html)
        docs = PiloterrLoader("https://a.example/", api_key=API_KEY, include_html=True).load()
        assert docs[0].metadata["html"] == PAGE

    def test_failure_raises_or_is_skipped(self, mock_http) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(500, json={"error": "upstream"})
            return httpx.Response(200, json=PAGE)

        mock_http(handler)
        urls = ["https://bad.example/", "https://good.example/"]
        with pytest.raises(PiloterrAPIError, match="HTTP 500"):
            PiloterrLoader(urls, api_key=API_KEY).load()
        docs = PiloterrLoader(urls, api_key=API_KEY, continue_on_failure=True).load()
        assert [d.metadata["source"] for d in docs] == ["https://good.example/"]

    async def test_aload(self, mock_http) -> None:
        mock_http(html)
        docs = await PiloterrLoader(["https://a.example/"], api_key=API_KEY, engine="rendering").aload()
        assert docs[0].metadata["engine"] == "rendering"

    def test_missing_key_raises_at_construction(self) -> None:
        with pytest.raises(PiloterrAPIError, match="PILOTERR_API_KEY"):
            PiloterrLoader(["https://a.example/"])


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        PiloterrWebsiteScraper(engine="browser")
