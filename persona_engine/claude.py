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

    prompt = f"""You are a senior cultural strategist at a leading creative agency. \
You have Qloo taste-affinity data showing what {age_str} {gender_str}s who \
have affinity for **{seed_name}** gravitate toward across {n_cities} {"city" if n_cities == 1 else "cities"}.{compare_clause}

Taste data — top affinity results by city and category:
{data_summary}

Write a rich, consulting-grade persona profile for each city. Each profile must include:

- **archetype**: A vivid 3–5 word name with "The" prefix capturing who this person is \
(e.g. "The West Loop Tastemaker", "The Sunday Brunch Optimizer"). \
Make it feel earned by the data, not generic.

- **core_promises**: Array of exactly 5 short, punchy, quotable lines written in the \
second person ("you") or as brand promises — capturing the emotional pull of this \
lifestyle for this audience. Each line should be poetic, specific, and 8–15 words. \
(e.g. "Your slip is your social circle, dock life meets design.", \
"Luxury without pretension, adventure without effort.")

- **psychographics**: 1–2 sentences describing this person's mindset, values, and \
social identity — what they believe about themselves and how they move through the world. \
Be specific and cultural, not demographic.

- **core_motivation**: One sharp sentence capturing the single deepest driver for this \
persona — what they are ultimately seeking. \
(e.g. "Belonging to a community that's elevated, social, and convenient.")

- **lifestyle_traits**: A comma-separated string of 6–8 vivid descriptors capturing \
how this person lives day-to-day. \
(e.g. "Design-conscious, fitness-minded, experience-driven, socially curated, sustainability-oriented")

- **key_interests**: Array of 7–9 specific behaviors and interests inferred directly \
from the taste data — concrete activities, media, brands, and habits. \
(e.g. "Weekend farmers markets and boutique fitness", "Craft cocktail culture and local dining")

- **what_they_seek**: Array of 5–6 bullet points on what specifically attracts and \
retains this persona in this city — what experiences, amenities, or qualities of place \
resonate most with them here. Ground these in the actual taste signals.

- **day_in_the_life**: An immersive 4–6 sentence narrative written entirely in second \
person ("you") — specific to this city, drawing on the actual taste data entities. \
Paint a vivid sensory scene of a day in this person's life here. \
End with a line that captures emotional resonance, not just activity. \
(Model: "Morning coffee on the deck as the marina hums awake. A casual chat with \
neighbors before setting out... Here, you're not just near the lake — you belong to it.")

- **creative_direction**: Array of exactly 4 actionable cues for a creative team: \
tone of voice, visual/aesthetic language, cultural touchpoints to reference, \
and one "avoid" (what would ring false for this persona).

- **city_distinction**: One sharp sentence on what makes this city's version of this \
persona meaningfully different from the others. If there's only one city, describe \
what makes this persona culturally specific to that market.

Return ONLY valid JSON — no markdown fences, no commentary — in this exact shape:
{{
  "CityName": {{
    "archetype": "...",
    "core_promises": ["...", "...", "...", "...", "..."],
    "psychographics": "...",
    "core_motivation": "...",
    "lifestyle_traits": "...",
    "key_interests": ["...", "...", "...", "...", "...", "...", "..."],
    "what_they_seek": ["...", "...", "...", "...", "..."],
    "day_in_the_life": "...",
    "creative_direction": ["...", "...", "...", "..."],
    "city_distinction": "..."
  }}
}}"""

    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    return json.loads(text)
