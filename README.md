# Tasteprint

**Tasteprint** is a cross-domain taste persona engine powered by [Qloo](https://qloo.com) and [Claude](https://claude.ai). Describe an audience in plain English, pick one or more cities, and Tasteprint returns affinity-ranked taste data across Food & Beverage, Music, Fashion, Movies, and TV — then synthesizes it into vivid cultural persona profiles with a one-click PDF export.

## What it does

1. **Parses your audience** — Claude (Haiku) extracts age, gender, and cultural interest signals from a natural language description.
2. **Pulls taste affinities** — Qloo's Insights API returns the top-affinity entities per city × domain combination, ranked for that audience.
3. **Synthesizes personas** — Claude (Sonnet) writes a cultural strategist-quality persona profile for each city: archetype, narrative, creative direction, and market distinction.
4. **Exports a PDF brief** — A consulting-grade report with cover page, research brief, per-city persona profiles, and taste intelligence grids.

## Requirements

You need **two API keys** to run Tasteprint:

| Key | Purpose | Get one at |
|-----|---------|------------|
| `ANTHROPIC_API_KEY` | Claude (audience parsing + persona synthesis) | https://console.anthropic.com |
| `QLOO_API_KEY` | Qloo taste affinity + entity search | https://qloo.com |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/QlooRE/tasteprint.git
cd tasteprint

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export QLOO_API_KEY="your-qloo-key"
```

## Run

```bash
python app.py
```

Then open [http://127.0.0.1:8001](http://127.0.0.1:8001) in your browser.

## Usage

1. Enter an audience description — e.g. *"25–34 year old women who love Pilates and The White Lotus"*
2. Add one or more cities — e.g. `New York, Los Angeles, Chicago`
3. Select the taste domains you want to include
4. Click **Generate** and watch results stream in
5. Click **↓ Export PDF** once the run completes

## Project structure

```
app.py                    FastAPI app + SSE streaming endpoint
persona_engine/
  qloo.py                 Qloo API client (entity search + insights)
  claude.py               Claude API calls (audience parsing + persona synthesis)
  models.py               Pydantic/dataclass models
static/
  index.html              Single-page frontend
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `QLOO_API_KEY` | Yes | Qloo API key for taste data |

## License

MIT
