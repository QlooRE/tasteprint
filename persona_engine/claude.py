"""Claude API — synthesize persona narratives from Qloo taste data."""

import json
import os

from anthropic import Anthropic

_client: Anthropic | None = None

AGE_VALUES = ["24_and_younger", "25_to_29", "30_to_34", "35_to_44", "45_to_54", "55_and_older"]


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def parse_audience(description: str) -> dict:
    """Use Claude (Haiku) to extract structured Qloo signals from a natural language audience description."""
    prompt = f"""Parse this audience description into Qloo API signal parameters.

Description: "{description}"

Return ONLY valid JSON with these fields:
- "age": one of {AGE_VALUES} — infer from any age mention, or null
- "gender": "male", "female", or null
- "entity_names": array of specific named brands, restaurants, artists, movies, TV shows, or places to search Qloo for (e.g. ["Wendy's", "Nike", "Drake"])
- "interest_terms": array of general taste/interest keywords that don't map to a specific entity (e.g. ["streetwear", "hip-hop", "fine dining"])
- "audience_label": a concise 4–6 word label describing this audience (e.g. "30–34 male streetwear enthusiasts")

Example input: "30-34 year old men who love Wendy's and follow street fashion"
Example output: {{"age":"30_to_34","gender":"male","entity_names":["Wendy's"],"interest_terms":["street fashion","streetwear"],"audience_label":"30–34 male streetwear enthusiasts"}}"""

    client = _get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _age_label(age: str) -> str:
    return (
        age.replace("_and_younger", " and younger")
           .replace("_to_", "–")
           .replace("_and_older", "+")
    ) if age else "all ages"


def generate_personas(
    seed_name: str,
    age: str,
    gender: str,
    cities_data: dict,   # {city: {domain_label: [{"name": str, "tags": list[str]}]}}
) -> dict:
    """Call Claude to write persona profiles for every city in cities_data."""

    age_str    = _age_label(age)
    gender_str = gender or "all genders"
    city_list  = list(cities_data.keys())
    n_cities   = len(city_list)

    # Build a compact data summary
    lines = []
    for city, domains in cities_data.items():
        lines.append(f"\n**{city}**")
        for domain_label, results in domains.items():
            if not results:
                continue
            names    = ", ".join(r["name"] for r in results[:6])
            all_tags = []
            for r in results[:4]:
                all_tags.extend(r.get("tags", [])[:2])
            tag_str = f"  [{', '.join(dict.fromkeys(all_tags))[:4]}]" if all_tags else ""
            lines.append(f"  {domain_label}: {names}{tag_str}")

    data_summary   = "\n".join(lines)
    compare_clause = (
        f" Compare and contrast these {n_cities} markets: {', '.join(city_list)}."
        if n_cities > 1 else ""
    )

    prompt = f"""You are a senior analyst at an audience intelligence firm. \
You have Qloo taste-affinity data showing what {age_str} {gender_str}s who \
have affinity for **{seed_name}** gravitate toward across {n_cities} {"city" if n_cities == 1 else "cities"}.{compare_clause}

Taste data — top affinity results by city and category:
{data_summary}

Produce a data-grounded audience profile for each city. Every claim must be \
traceable to the taste signals above — do not invent behaviors or interests \
not supported by the data. Write in a confident, analytical register: clear and \
specific, like a Nielsen or Ipsos deliverable. Avoid marketing copy, aspirational \
language, or generic descriptors.

Each profile must include:

- **archetype**: A precise 3–5 word name with "The" prefix that captures who this \
person is as revealed by the data (e.g. "The West Loop Tastemaker"). \
Earn it from the signals — avoid clichés.

- **summary**: 3–4 sentences synthesizing the full picture: who this audience is, \
what their collective taste profile reveals about their motivations and lifestyle, \
and — if multiple cities — what distinguishes this market's version. \
Lead with the insight, not the demographics.

- **taste_insights**: Array of exactly 4 insight statements, each connecting one or \
more specific entities from the taste data to a behavioral or motivational pattern. \
Format: observation about the data → what it reveals. Be precise and non-obvious. \
(e.g. "Co-indexing on Equinox and Aesop alongside Bon Appétit suggests this audience \
treats premiumization as identity expression rather than status display — quality is \
personal, not performative.")

- **day_in_the_life**: A 4–5 sentence narrative in second person ("you"), grounded \
in this specific city, drawing on actual entities from the taste data. \
Write as an observed portrait, not a fantasy. Restrained, specific, credible. \
No hyperbole — the texture of a real day, not a highlight reel.

- **audience_signals**: Array of exactly 4 bullets summarizing the key brand, media, \
and cultural affinities for this audience and what each cluster signals about their \
consumption patterns, values, or behaviors. Ground each bullet in the data. \
(e.g. "Music: strong indexing toward [artist] and [artist] indicates preference for \
[characteristic] — suggests receptivity to [implication].")

- **city_distinction**: One sharp analytical sentence on what makes this city's version \
of this audience meaningfully different from the others. If only one city, describe \
what the data reveals as culturally specific to this market.

Return ONLY valid JSON — no markdown fences, no commentary — in this exact shape:
{{
  "CityName": {{
    "archetype": "...",
    "summary": "...",
    "taste_insights": ["...", "...", "...", "..."],
    "day_in_the_life": "...",
    "audience_signals": ["...", "...", "...", "..."],
    "city_distinction": "..."
  }}
}}"""

    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    return json.loads(text)
