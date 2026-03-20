#!/usr/bin/env python3
"""Persona Engine — cross-domain taste personas powered by Qloo + Claude."""

import asyncio
import dataclasses
import json
import os
import secrets
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from persona_engine.qloo import QlooClient, DOMAINS
from persona_engine.claude import generate_personas, parse_audience

STATIC = Path(__file__).parent / "static"

app = FastAPI()
_executor = ThreadPoolExecutor(max_workers=24)
_sessions: dict[str, dict] = {}   # token -> {qloo_key}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _fetch_domain_sync(
    client: QlooClient, city: str, domain: str,
    seed_ids: list[str], age: str, gender: str,
):
    return client.domain_insights(city, domain, seed_ids, age, gender, take=8)


def _search_entity_sync(client: QlooClient, name: str) -> tuple[str, dict | None]:
    try:
        return name, client.search_entity(name)
    except Exception:
        return name, None


def _search_locality_sync(client: QlooClient, query: str) -> dict | None:
    try:
        return client.search_locality(query)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _event_stream(
    request: Request,
    audience: str,
    cities: list[str],
    domains: list[str],
    qloo_key: str,
) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    qloo = QlooClient(qloo_key)

    # ── 1. Parse natural language audience description ───────────────────
    age      = ""
    gender   = ""
    seed_ids: list[str] = []
    audience_label = audience or "General audience"

    if audience.strip():
        try:
            parsed = await loop.run_in_executor(_executor, parse_audience, audience.strip())
            age    = parsed.get("age") or ""
            gender = parsed.get("gender") or ""
            audience_label = parsed.get("audience_label") or audience

            # Resolve entity names to Qloo IDs in parallel
            entity_names    = parsed.get("entity_names", [])
            interest_terms  = parsed.get("interest_terms", [])
            all_search_terms = entity_names + interest_terms

            search_tasks = [
                loop.run_in_executor(_executor, _search_entity_sync, qloo, name)
                for name in all_search_terms
            ]
            resolved = []
            for coro in asyncio.as_completed(search_tasks):
                name, entity = await coro
                if entity and entity.get("entity_id"):
                    seed_ids.append(entity["entity_id"])
                    resolved.append({
                        "queried": name,
                        "matched": entity.get("name", name),
                        "type":    entity.get("subtype", "").split(":")[-1],
                    })

            yield _sse("audience_parsed", {
                "label":    audience_label,
                "age":      age,
                "gender":   gender,
                "resolved": resolved,
                "seed_ids": seed_ids,
            })
        except Exception as exc:
            yield _sse("audience_parsed", {
                "label":   audience,
                "error":   str(exc),
                "seed_ids": [],
            })
    else:
        yield _sse("audience_parsed", {"label": "No audience signal", "seed_ids": []})

    # ── 2. Launch all city × domain calls concurrently ───────────────────
    tasks = [
        loop.run_in_executor(
            _executor, _fetch_domain_sync, qloo, city, domain, seed_ids, age, gender,
        )
        for city   in cities
        for domain in domains
    ]

    # Accumulate for Claude:  city → domain_label → [{name, tags}]
    accumulated: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for coro in asyncio.as_completed(tasks):
        if await request.is_disconnected():
            return

        result = await coro
        serialized = [dataclasses.asdict(r) for r in result.results]

        accumulated[result.city][result.domain_label] = [
            {"name": r["name"], "tags": r["tags"]} for r in serialized
        ]

        yield _sse("domain_done", {
            "city":         result.city,
            "domain":       result.domain,
            "domain_label": result.domain_label,
            "results":      serialized,
            "error":        result.error,
        })

    # ── 3. Generate persona narratives with Claude ───────────────────────
    if accumulated:
        yield _sse("persona_generating", {
            "message": "Synthesizing personas…",
            "cities":  list(accumulated.keys()),
        })
        try:
            personas = await loop.run_in_executor(
                _executor,
                generate_personas,
                audience_label,
                age,
                gender,
                dict(accumulated),
            )
            for city, data in personas.items():
                yield _sse("persona_done", {"city": city, **data})
        except Exception as exc:
            yield _sse("error", {"message": f"Persona synthesis failed: {exc}"})

    yield _sse("done", {"cities": len(cities), "domains": len(domains)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    html = (STATIC / "index.html").read_text()
    prefilled_qloo = os.environ.get("QLOO_API_KEY", "")
    return HTMLResponse(html.replace("__QLOO_KEY__", prefilled_qloo))


@app.post("/session")
async def create_session(request: Request):
    body  = await request.json()
    token = secrets.token_urlsafe(24)
    _sessions[token] = {"qloo_key": body.get("qloo_key", "").strip()}
    return JSONResponse({"token": token})


@app.get("/locality-search")
async def locality_search(q: str = ""):
    """Validate a city name against Qloo's locality index. Uses server env key."""
    qloo_key = os.environ.get("QLOO_API_KEY", "")
    if not q.strip() or not qloo_key:
        return JSONResponse({"found": False})
    loop  = asyncio.get_running_loop()
    qloo  = QlooClient(qloo_key)
    result = await loop.run_in_executor(_executor, _search_locality_sync, qloo, q.strip())
    if result:
        return JSONResponse({
            "found":          True,
            "name":           result.get("name", q),
            "disambiguation": result.get("disambiguation", ""),
        })
    return JSONResponse({"found": False})


@app.get("/stream")
async def stream(
    request: Request,
    audience: str = "",
    cities:  str = "",
    domains: str = "",
    token:   str = "",
):
    session  = _sessions.get(token, {})
    qloo_key = session.get("qloo_key") or os.environ.get("QLOO_API_KEY", "")

    city_list   = [c.strip() for c in cities.split(",")  if c.strip()]
    domain_list = [d.strip() for d in domains.split(",") if d.strip() in DOMAINS]

    if not city_list:
        return JSONResponse({"error": "No cities specified"}, status_code=400)
    if not domain_list:
        domain_list = list(DOMAINS.keys())

    return StreamingResponse(
        _event_stream(request, audience, city_list, domain_list, qloo_key),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
