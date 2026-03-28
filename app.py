#!/usr/bin/env python3
"""Tasteprint — FastAPI app + SSE streaming endpoint."""

import asyncio
import json
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from persona_engine.agent import DOMAIN_TYPES, run_agent

STATIC = Path(__file__).parent / "static"

app = FastAPI()
_executor = ThreadPoolExecutor(max_workers=24)
_sessions: dict[str, dict] = {}


# ─── Routes ────────────────────────────────────────────────────────────────────

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


@app.get("/stream")
async def stream(
    request: Request,
    audience: str = "",
    cities:   str = "",
    domains:  str = "",
    token:    str = "",
):
    session  = _sessions.get(token, {})
    qloo_key = session.get("qloo_key") or os.environ.get("QLOO_API_KEY", "")

    # Override the env key with the session key if the user supplied one
    if qloo_key:
        os.environ["QLOO_API_KEY"] = qloo_key

    city_list   = [c.strip() for c in cities.split(",")  if c.strip()]
    domain_list = [d.strip() for d in domains.split(",") if d.strip() in DOMAIN_TYPES]

    if not city_list:
        return JSONResponse({"error": "No cities specified"}, status_code=400)
    if not domain_list:
        domain_list = list(DOMAIN_TYPES.keys())

    async def event_stream():
        loop = asyncio.get_running_loop()
        try:
            async for evt in run_agent(audience, city_list, domain_list, loop, _executor):
                if await request.is_disconnected():
                    return
                yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
