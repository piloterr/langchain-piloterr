"""LangChain integration for Piloterr: Google search and website scraping."""

from importlib import metadata

from langchain_piloterr._client import PiloterrAPIError
from langchain_piloterr.document_loaders import PiloterrLoader
from langchain_piloterr.tools import (
    PiloterrGoogleSearch,
    PiloterrGoogleSearchInput,
    PiloterrWebsiteScraper,
    PiloterrWebsiteScraperInput,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "PiloterrAPIError",
    "PiloterrGoogleSearch",
    "PiloterrGoogleSearchInput",
    "PiloterrLoader",
    "PiloterrWebsiteScraper",
    "PiloterrWebsiteScraperInput",
    "__version__",
]
