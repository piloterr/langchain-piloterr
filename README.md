# langchain-piloterr

LangChain integration for [Piloterr](https://www.piloterr.com), a web data API.

It adds three components:

- `PiloterrGoogleSearch`, a tool that searches Google;
- `PiloterrWebsiteScraper`, a tool that reads a web page and returns its text;
- `PiloterrLoader`, a document loader that turns web pages into LangChain documents.

Piloterr fetches pages on its own infrastructure in one of three ways:
- `crawler`: plain HTTP, the default;
- `rendering`: a headless browser, for JavaScript pages;
- `webunlocker`: for pages behind anti-bot protection.

## Installation

```bash
pip install -U langchain-piloterr
```

Get an API key at [app.piloterr.com](https://app.piloterr.com) and set it in the environment:

```bash
export PILOTERR_API_KEY="your-api-key"
```

You can also pass `api_key=...` to any component.

## Google search tool

```python
from langchain_piloterr import PiloterrGoogleSearch

tool = PiloterrGoogleSearch(max_results=5, country="us")
tool.invoke({"query": "web scraping api"})
```

The tool returns the query and a `results` list. Each result has `title`, `url`, `content` (the snippet) and `position`.

| Option | Default | Description |
| --- | --- | --- |
| `max_results` | `10` | Results requested from Google, 1 to 100 |
| `country` | | Two-letter country code, such as `us` |
| `language` | | Language code, optionally with a region, such as `en-gb` |
| `location` | | City or region to search from, such as `Paris, France` |
| `timeout` | `60` | Seconds to wait for Piloterr |

## Website scraper tool

```python
from langchain_piloterr import PiloterrWebsiteScraper

tool = PiloterrWebsiteScraper(engine="rendering", wait_in_seconds=2)
print(tool.invoke({"url": "https://example.com"}))
```

The tool returns the page title as a heading, followed by the readable text. Scripts, styles and markup are removed. The text is cut at `max_characters` (20,000 by default; `None` keeps everything).

| Option | Default | Description |
| --- | --- | --- |
| `engine` | `crawler` | `crawler`, `rendering` or `webunlocker` |
| `wait_in_seconds` | | Rendering only: extra wait after the page loads, 0 to 30 |
| `max_characters` | `20000` | Truncate the returned text |
| `timeout` | `120` | Seconds to wait for Piloterr |

Both tools return API errors to the model as text instead of raising, and the key never appears in them. Piloterr reports a rate limit as HTTP 401 with "Rate limit exceeded"; the tools say "rate limit exceeded" in that case.

### With an agent

```python
from langchain.agents import create_agent
from langchain_piloterr import PiloterrGoogleSearch, PiloterrWebsiteScraper

agent = create_agent("openai:gpt-4.1-mini", tools=[PiloterrGoogleSearch(), PiloterrWebsiteScraper()])
agent.invoke({"messages": [{"role": "user", "content": "Summarize the home page of piloterr.com"}]})
```

## Document loader

```python
from langchain_piloterr import PiloterrLoader

loader = PiloterrLoader(
    ["https://example.com", "https://www.python.org"],
    engine="crawler",
    continue_on_failure=True,
)
docs = loader.load()
```

Each document's `page_content` is the readable text of the page. Its metadata holds `source`, `title` and `engine`. Set `include_html=True` to also keep the raw HTML in `metadata["html"]`.

All components have async versions (`ainvoke`, `aload`).

## Development

```bash
pip install -e ".[test]"
pytest tests/unit_tests                         # offline
PILOTERR_API_KEY=... pytest tests/integration_tests  # live
```

## License

MIT
