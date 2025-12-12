"""Lightweight CoinDesk web search helper used at runtime."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timezone
from email.utils import parsedate_to_datetime
from html import unescape
import logging
import re
from typing import Sequence
import xml.etree.ElementTree as ET

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class WebArticle:
    """Normalized news article payload."""

    title: str
    summary: str
    url: str
    published_at: str | None
    categories: str
    source: str = "CoinDesk"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


class CoinDeskWebSearch:
    """Fetch the public CoinDesk RSS feed and filter it by query/symbols."""

    FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/xml",
    }
    _SYMBOL_KEYWORDS = {
        "BTC": ("btc", "bitcoin"),
        "ETH": ("eth", "ether", "ethereum"),
        "SOL": ("sol", "solana"),
        "XRP": ("xrp", "ripple"),
        "DOGE": ("doge", "dogecoin"),
        "ADA": ("ada", "cardano"),
        "AVAX": ("avax", "avalanche"),
        "DOT": ("dot", "polkadot"),
        "MATIC": ("matic", "polygon"),
        "LINK": ("link", "chainlink"),
    }

    def __init__(self, feed_url: str | None = None, timeout: int = 15) -> None:
        self._feed_url = feed_url or self.FEED_URL
        self._timeout = timeout
        self._session = requests.Session()

    def search(
        self,
        query: str,
        symbols: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, str | None]]:
        """Return a list of the latest CoinDesk articles filtered by the query."""
        feed_text = self._fetch_feed()
        if not feed_text:
            return []
        articles = self._parse_feed(feed_text)
        tokens = self._tokenize(query)
        symbol_tokens = self._symbol_tokens(symbols or ())

        filtered: list[WebArticle] = []
        for article in articles:
            haystack = " ".join(
                filter(
                    None,
                    (article.title, article.summary, article.categories),
                )
            ).lower()
            if tokens and not any(token in haystack for token in tokens):
                continue
            if symbol_tokens and not any(token in haystack for token in symbol_tokens):
                continue
            filtered.append(article)

        if not filtered:
            filtered = articles  # fall back to latest headlines
        trimmed = filtered[:limit]
        return [item.to_dict() for item in trimmed]

    def _fetch_feed(self) -> str | None:
        try:
            response = self._session.get(self._feed_url, headers=self._HEADERS, timeout=self._timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            LOGGER.warning("CoinDesk feed fetch failed: %s", exc)
            return None

    def _parse_feed(self, xml_text: str) -> list[WebArticle]:
        articles: list[WebArticle] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            LOGGER.warning("Failed to parse CoinDesk RSS feed: %s", exc)
            return articles

        for item in root.findall(".//item"):
            title = self._clean_text(item.findtext("title", default="").strip())
            summary = self._clean_text(item.findtext("description", default="").strip())
            url = item.findtext("link", default="").strip()
            pub_date_raw = item.findtext("pubDate")
            published_at = self._normalize_datetime(pub_date_raw)
            categories = ",".join(
                self._clean_text(elem.text or "")
                for elem in item.findall("category")
                if elem is not None and (elem.text or "").strip()
            )
            articles.append(
                WebArticle(
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=published_at,
                    categories=categories,
                )
            )
        return articles

    @staticmethod
    def _clean_text(value: str) -> str:
        if not value:
            return ""
        text = unescape(value)
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _normalize_datetime(value: str | None) -> str | None:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            return value

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        terms = re.findall(r"[a-z0-9]{3,}", text.lower())
        return tuple(dict.fromkeys(terms))

    def _symbol_tokens(self, symbols: Sequence[str]) -> tuple[str, ...]:
        tokens: list[str] = []
        for symbol in symbols:
            keywords = self._SYMBOL_KEYWORDS.get(symbol.upper())
            if keywords:
                tokens.extend(keywords)
            else:
                tokens.append(symbol.lower())
        return tuple(dict.fromkeys(tokens))


__all__ = ["CoinDeskWebSearch", "WebArticle"]
