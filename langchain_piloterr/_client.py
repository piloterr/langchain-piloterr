"""HTTP client for the Piloterr API (https://docs.piloterr.com)."""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from typing import Any, Literal

import httpx

DEFAULT_BASE_URL = "https://api.piloterr.com"
DEFAULT_TIMEOUT = 120.0
USER_AGENT = "langchain-piloterr"

Engine = Literal["crawler", "rendering", "webunlocker"]
ENGINES: tuple[str, ...] = ("crawler", "rendering", "webunlocker")

_ERROR_DETAIL_LIMIT = 300


class PiloterrAPIError(Exception):
    """A Piloterr request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``PILOTERR_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get("PILOTERR_API_KEY") or "").strip()
    if not key:
        raise PiloterrAPIError("Set PILOTERR_API_KEY or pass api_key. Get a key at https://app.piloterr.com")
    return key


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def _error(response: httpx.Response, api_key: str) -> PiloterrAPIError:
    detail = ""
    try:
        body = response.json()
    except ValueError:
        detail = response.text.strip()
    else:
        if isinstance(body, dict):
            error = body.get("error") or body.get("message") or ""
            if isinstance(error, dict):
                error = error.get("message") or ""
            detail = str(error)
    detail = _redact(detail, api_key)[:_ERROR_DETAIL_LIMIT]
    status = response.status_code
    if status == 401 and "rate limit" in detail.lower():
        label = "Piloterr rate limit exceeded"
    elif status == 401:
        label = "Piloterr rejected the API key"
    elif status == 402:
        label = "Piloterr account is out of credits"
    elif status == 429:
        label = "Piloterr rate limit exceeded"
    else:
        label = "Piloterr API error"
    return PiloterrAPIError(f"{label} (HTTP {status})" + (f": {detail}" if detail else ""), status)


def _headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "Accept": "application/json", "User-Agent": USER_AGENT}


def get(
    path: str,
    params: dict[str, Any],
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Response:
    """GET ``path`` and return the successful response."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url.rstrip('/')}{path}", params=params, headers=_headers(api_key))
    except httpx.TimeoutException as exc:
        raise PiloterrAPIError(f"Piloterr did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise PiloterrAPIError(f"Could not reach Piloterr: {_redact(str(exc), api_key)}") from exc
    if not response.is_success:
        raise _error(response, api_key)
    return response


async def aget(
    path: str,
    params: dict[str, Any],
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Response:
    """Async version of :func:`get`."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url.rstrip('/')}{path}", params=params, headers=_headers(api_key))
    except httpx.TimeoutException as exc:
        raise PiloterrAPIError(f"Piloterr did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise PiloterrAPIError(f"Could not reach Piloterr: {_redact(str(exc), api_key)}") from exc
    if not response.is_success:
        raise _error(response, api_key)
    return response


def organic_rows(response: httpx.Response) -> list[dict[str, Any]]:
    """The ``organic_results`` rows of a Google search response (missing means none)."""
    try:
        body = response.json()
    except ValueError as exc:
        raise PiloterrAPIError("Piloterr returned a response that is not JSON.") from exc
    if not isinstance(body, dict):
        raise PiloterrAPIError("Piloterr returned an unexpected response shape.")
    rows = body.get("organic_results")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise PiloterrAPIError("Piloterr returned organic_results that is not a list.")
    return [row for row in rows if isinstance(row, dict)]


def page_html(response: httpx.Response) -> str:
    """The page HTML of a website endpoint: a JSON string, or a raw HTML body."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    if not isinstance(body, str):
        raise PiloterrAPIError("Piloterr returned an unexpected response shape; expected the page HTML.")
    return body


def check_url(url: str) -> str:
    """Return ``url`` stripped, or raise if it is not an absolute http(s) URL."""
    url = (url or "").strip()
    if not re.match(r"^https?://[^/\s]+", url, re.IGNORECASE):
        raise PiloterrAPIError(f"Use an absolute http(s) URL, such as https://example.com; got {url!r}.")
    return url


def website_params(url: str, engine: str, wait_in_seconds: int | None) -> tuple[str, dict[str, Any]]:
    """Path and query parameters for ``/v2/website/{engine}``."""
    if engine not in ENGINES:
        raise PiloterrAPIError(f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    params: dict[str, Any] = {"query": check_url(url)}
    if engine == "rendering" and wait_in_seconds:
        params["wait_in_seconds"] = min(max(int(wait_in_seconds), 0), 30)
    return f"/v2/website/{engine}", params


_SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "td", "th", "tr", "ul",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    """Readable text of a page: drops scripts and styles, keeps one line per block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIPPED_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIPPED_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML page."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return " ".join("".join(parser.title_parts).split()), "\n".join(line for line in lines if line)


def to_result(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one organic row to ``title``/``url``/``content`` plus its position."""
    result: dict[str, Any] = {
        "title": row.get("title") or "",
        "url": row.get("link") or "",
        "content": row.get("snippet") or "",
    }
    if row.get("position") is not None:
        result["position"] = row["position"]
    return result
