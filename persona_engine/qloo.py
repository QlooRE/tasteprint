"""Qloo API client — entity search + multi-domain taste insights."""

import requests
from typing import Optional

from .models import DomainResult, EntityResult

QLOO_SEARCH   = "https://api.qloo.com/search"
QLOO_INSIGHTS = "https://api.qloo.com/v2/insights"

DOMAINS: dict[str, dict] = {
    "fnb": {
        "label": "Food & Beverage",
        "filter_type": "urn:entity:place",
    },
    "music": {
        "label": "Music",
        "filter_type": "urn:entity:artist",
    },
    "fashion": {
        "label": "Fashion & Brands",
        "filter_type": "urn:entity:brand",
    },
    "movies": {
        "label": "Movies",
        "filter_type": "urn:entity:movie",
    },
    "tv": {
        "label": "TV",
        "filter_type": "urn:entity:tv_show",
    },
}

AGE_OPTIONS = [
    ("",               "Any age"),
    ("24_and_younger", "Under 25"),
    ("25_to_29",       "25–29"),
    ("30_to_34",       "30–34"),
    ("35_to_44",       "35–44"),
    ("45_to_54",       "45–54"),
    ("55_and_older",   "55+"),
]

# Tag ID fragments worth surfacing per domain type
_MEANINGFUL_TAG_FRAGMENTS = [
    "cuisine:",
    "culinary_style:",
    "ambience:",
    "good_for:",
    "genre:artist",
    "genre:movie",
    "genre:tv_show",
    "genre:brand",
    "genre:music",
    "mood:",
    "style:",
    "decade:",
]

_SKIP_TAG_NAMES = {"Place", "Restaurant", "Bar", "Artist", "Brand", "Movie", "TV Show"}


def _extract_tags(raw_tags: list) -> list[str]:
    out = []
    for t in raw_tags or []:
        if not isinstance(t, dict):
            continue
        tid  = t.get("id", "")
        name = t.get("name", "").strip()
        if not name or name in _SKIP_TAG_NAMES:
            continue
        if any(frag in tid for frag in _MEANINGFUL_TAG_FRAGMENTS):
            out.append(name)
    return out[:5]


class QlooClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers["X-Api-Key"] = api_key

    def search_entity(self, query: str) -> Optional[dict]:
        """Resolve a name to a Qloo entity (any type). Returns raw entity dict or None."""
        r = self._session.get(
            QLOO_SEARCH,
            params={"query": query, "take": 5},
            timeout=10,
        )
        r.raise_for_status()
        entities = r.json().get("results", [])
        if isinstance(entities, dict):
            entities = entities.get("entities", [])
        # Skip locality results — those are for city filtering, not interest signals
        for e in entities:
            if "locality" not in e.get("subtype", ""):
                return e
        return None

    def search_locality(self, query: str) -> Optional[dict]:
        """Resolve a city name to a Qloo locality entity."""
        r = self._session.get(
            QLOO_SEARCH,
            params={"query": query, "types": "urn:entity:locality", "filter.recommendable": "true", "take": 1},
            timeout=10,
        )
        r.raise_for_status()
        entities = r.json().get("results", [])
        if isinstance(entities, dict):
            entities = entities.get("entities", [])
        return entities[0] if entities else None

    def domain_insights(
        self,
        city: str,
        domain: str,
        seed_ids: list[str],
        age: Optional[str],
        gender: Optional[str],
        take: int = 8,
    ) -> DomainResult:
        """Return affinity-ranked results for one city × domain combination."""
        cfg = DOMAINS[domain]
        params: dict = {
            "filter.type":           cfg["filter_type"],
            "filter.location.query": city,
            "signal.location.query": city,
            "take":                  take,
        }
        if seed_ids:
            params["signal.interests.entities"] = ",".join(seed_ids)
        if age:
            params["signal.demographics.age"] = age
        if gender:
            params["signal.demographics.gender"] = gender

        try:
            r = self._session.get(QLOO_INSIGHTS, params=params, timeout=20)
            r.raise_for_status()
            raw_entities = r.json().get("results", {}).get("entities", [])
        except Exception as exc:
            return DomainResult(
                city=city, domain=domain, domain_label=cfg["label"],
                results=[], error=str(exc),
            )

        results = []
        for e in raw_entities:
            props   = e.get("properties") or {}
            results.append(EntityResult(
                name        = e.get("name", ""),
                affinity    = round(e.get("query", {}).get("affinity", 0), 4),
                popularity  = round(e.get("popularity", 0), 4),
                tags        = _extract_tags(e.get("tags")),
                description = (props.get("description") or "")[:120],
                address     = props.get("address", ""),
            ))

        return DomainResult(
            city=city, domain=domain, domain_label=cfg["label"], results=results,
        )
