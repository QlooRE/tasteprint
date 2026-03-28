"""Agentic Qloo + Claude loop.

Claude orchestrates the research: it decides which entities to search, which
city × domain combos to fetch, and when it has enough data to synthesize
personas. Python executes the Qloo tool calls server-side (API key never
leaves the process). SSE events are yielded as an async generator.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from anthropic import Anthropic

QLOO_BASE = "https://api.qloo.com"

DOMAIN_TYPES: dict[str, dict] = {
    "fnb":     {"label": "Food & Beverage", "filter_type": "urn:entity:place"},
    "music":   {"label": "Music",           "filter_type": "urn:entity:artist"},
    "fashion": {"label": "Fashion & Brands","filter_type": "urn:entity:brand"},
    "movies":  {"label": "Movies",          "filter_type": "urn:entity:movie"},
    "tv":      {"label": "TV",              "filter_type": "urn:entity:tv_show"},
}

TOOLS = [
    {
        "name": "search_entity",
        "description": (
            "Search Qloo for a named entity (brand, artist, restaurant, movie, TV show, etc.) "
            "and return its entity_id, name, and type. Use this to resolve audience interest "
            "signals to Qloo entity IDs before calling get_taste_insights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The entity name to search for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_taste_insights",
        "description": (
            "Fetch Qloo taste affinity insights for a given entity type, city, and audience signals. "
            "Returns a ranked list of entities this audience gravitates toward in that city."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_type": {
                    "type": "string",
                    "description": (
                        "Entity type URN — one of: urn:entity:artist | urn:entity:brand | "
                        "urn:entity:movie | urn:entity:tv_show | urn:entity:place"
                    ),
                },
                "city": {"type": "string", "description": "City name for location filtering"},
                "seed_entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Qloo entity IDs to use as interest signals (from search_entity calls)",
                },
                "age": {
                    "type": "string",
                    "description": "Age range: 24_and_younger | 25_to_29 | 30_to_34 | 35_to_44 | 45_to_54 | 55_and_older",
                },
                "gender": {"type": "string", "description": "male or female"},
                "take": {"type": "integer", "description": "Number of results (default 8)"},
            },
            "required": ["filter_type", "city"],
        },
    },
]


# ─── Qloo tool implementations ─────────────────────────────────────────────────

def _qloo_headers() -> dict:
    return {"X-Api-Key": os.environ["QLOO_API_KEY"]}


def _extract_tags(raw_tags: list) -> list[str]:
    FRAGS = [
        "cuisine:", "culinary_style:", "ambience:", "good_for:",
        "genre:artist", "genre:movie", "genre:tv_show", "genre:brand",
        "genre:music", "mood:", "style:", "decade:",
    ]
    SKIP = {"Place", "Restaurant", "Bar", "Artist", "Brand", "Movie", "TV Show"}
    out = []
    for t in raw_tags or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "").strip()
        if not name or name in SKIP:
            continue
        if any(frag in t.get("id", "") for frag in FRAGS):
            out.append(name)
    return out[:5]


def _tool_search_entity(query: str) -> dict:
    r = requests.get(
        f"{QLOO_BASE}/search",
        params={"query": query, "take": 5},
        headers=_qloo_headers(),
        timeout=10,
    )
    r.raise_for_status()
    entities = r.json().get("results", [])
    if isinstance(entities, dict):
        entities = entities.get("entities", [])
    for e in entities:
        if "locality" not in e.get("subtype", ""):
            return {
                "entity_id": e.get("entity_id"),
                "name": e.get("name"),
                "type": e.get("subtype", e.get("type", "")),
            }
    return {"error": f"No entity found for '{query}'"}


def _tool_get_taste_insights(
    filter_type: str,
    city: str,
    seed_entity_ids: list[str] | None = None,
    age: str | None = None,
    gender: str | None = None,
    take: int = 8,
) -> dict:
    params: dict = {
        "filter.type":           filter_type,
        "filter.location.query": city,
        "signal.location.query": city,
        "take":                  take,
    }
    if seed_entity_ids:
        params["signal.interests.entities"] = ",".join(seed_entity_ids)
    if age:
        params["signal.demographics.age"] = age
    if gender:
        params["signal.demographics.gender"] = gender

    r = requests.get(
        f"{QLOO_BASE}/v2/insights",
        params=params,
        headers=_qloo_headers(),
        timeout=20,
    )
    r.raise_for_status()
    entities = r.json().get("results", {}).get("entities", [])
    results = []
    for e in entities:
        props = e.get("properties") or {}
        results.append({
            "name":        e.get("name", ""),
            "affinity":    round(e.get("query", {}).get("affinity", 0), 4),
            "popularity":  round(e.get("popularity", 0), 4),
            "tags":        _extract_tags(e.get("tags")),
            "description": (props.get("description") or "")[:120],
            "address":     props.get("address", ""),
        })
    return {"results": results}


def _dispatch_tool(name: str, inp: dict) -> dict:
    """Execute a tool call synchronously. Called from a thread executor."""
    if name == "search_entity":
        return _tool_search_entity(inp["query"])
    if name == "get_taste_insights":
        return _tool_get_taste_insights(
            filter_type=inp["filter_type"],
            city=inp["city"],
            seed_entity_ids=inp.get("seed_entity_ids"),
            age=inp.get("age"),
            gender=inp.get("gender"),
            take=inp.get("take", 8),
        )
    return {"error": f"Unknown tool: {name}"}


# ─── Agentic loop ──────────────────────────────────────────────────────────────

def _build_system(audience: str, cities: list[str], domains: list[str]) -> str:
    active = {k: DOMAIN_TYPES[k] for k in domains if k in DOMAIN_TYPES}
    domain_lines = "\n".join(
        f"  {k}: {v['label']} (filter_type: {v['filter_type']})"
        for k, v in active.items()
    )
    city_list = ", ".join(cities)
    return f"""You are a taste intelligence analyst powered by Qloo.

TASK
1. Parse the audience description — extract age range, gender, and named cultural interest signals (specific brands, artists, restaurants, shows, etc.).
2. Resolve each interest signal to a Qloo entity ID via search_entity.
3. For every city × domain combination, call get_taste_insights with the resolved entity IDs as seeds.
4. Once all data is gathered, return ONLY a JSON object with persona profiles.

AUDIENCE: {audience}
CITIES: {city_list}
DOMAINS:
{domain_lines}

PERSONA JSON FORMAT — return this and nothing else when done:
{{
  "CityName": {{
    "archetype": "The [3-5 word label earned from the data, not a cliché]",
    "summary": "3-4 sentences synthesizing the full taste picture. Lead with the insight, not the demographics.",
    "taste_insights": [
      "4 insight statements. Each connects specific entities from the data to a behavioral or motivational pattern. Format: observation → what it reveals. Precise and non-obvious."
    ],
    "day_in_the_life": "4-5 sentences in second person (you), grounded in this city, drawing on actual entities from the data. Restrained and specific.",
    "audience_signals": [
      "4 bullets summarizing key brand, media, and cultural affinities and what each signals about values or consumption patterns. Ground each in the data."
    ],
    "city_distinction": "One sharp analytical sentence on what makes this city's version of this audience meaningfully different from the others."
  }}
}}

RULES
- Every claim must trace to an entity returned by your tool calls. Do not invent.
- Write like a Nielsen or Ipsos analyst — confident, precise, no marketing copy.
- Cover all {len(cities)} {"city" if len(cities) == 1 else "cities"} and all {len(active)} domains before synthesizing."""


async def run_agent(
    audience: str,
    cities: list[str],
    domains: list[str],
    loop,
    executor: ThreadPoolExecutor,
):
    """Async generator — yields SSE event dicts throughout the agentic loop."""
    client = Anthropic()
    system = _build_system(audience, cities, domains)
    messages = [{"role": "user", "content": "Begin the research now."}]

    # Emit audience_parsed immediately so the frontend can show the banner
    yield {"event": "audience_parsed", "data": {
        "label": audience,
        "age": None,
        "gender": None,
        "resolved": [],
        "seed_ids": [],
    }}

    while True:
        # Run the (blocking) Claude API call in the thread pool
        response = await loop.run_in_executor(
            executor,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=6000,
                system=system,
                tools=TOOLS,
                messages=messages,
            ),
        )

        # Append assistant turn to the conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Claude is done — extract the persona JSON from the text block
            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    personas = json.loads(match.group())
                    yield {"event": "persona_generating", "data": {}}
                    for city, data in personas.items():
                        yield {"event": "persona_done", "data": {"city": city, **data}}
                except json.JSONDecodeError as exc:
                    yield {"event": "error", "data": {"message": f"Persona JSON parse error: {exc}"}}
            else:
                yield {"event": "error", "data": {"message": "Agent finished without producing persona JSON."}}
            yield {"event": "done", "data": {}}
            break

        if response.stop_reason == "tool_use":
            tool_result_blocks = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                name = block.name
                inp  = block.input

                # Capture loop variables to avoid closure issues
                _name, _inp = name, inp
                try:
                    result = await loop.run_in_executor(
                        executor, lambda n=_name, i=_inp: _dispatch_tool(n, i)
                    )
                except Exception as exc:
                    result = {"error": str(exc)}

                # Emit domain_done when get_taste_insights completes
                if name == "get_taste_insights":
                    domain_key = next(
                        (k for k, v in DOMAIN_TYPES.items() if v["filter_type"] == inp.get("filter_type")),
                        None,
                    )
                    if domain_key:
                        yield {"event": "domain_done", "data": {
                            "city":         inp.get("city", ""),
                            "domain":       domain_key,
                            "domain_label": DOMAIN_TYPES[domain_key]["label"],
                            "results":      result.get("results", []),
                            "error":        result.get("error"),
                        }}

                tool_result_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_result_blocks})
