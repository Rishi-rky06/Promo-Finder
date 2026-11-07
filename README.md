# Promo Finder

Finds and ranks 20-60 second promo clips (hook in the first 3 seconds, cut on a cliffhanger, understandable
to a stranger) in timestamped microdrama screenplays, and explains every recommendation with evidence quoted
from the script. Built for the Kuku AI Generalist assignment (Problem 1).

**Hybrid design.** Code owns everything that must be exact: parsing, timestamps, durations, the 3-second hook
check, "does the clip actually end on the cliffhanger", never cutting a line mid-sentence, evidence quotes,
de-duplication, scoring arithmetic and ranking. A swappable model layer owns the editorial judgment:
which moments are hooks, whether an ending is unresolved, whether a stranger could follow it. The model only
ever refers to screenplay event ids (`L012`); it never writes a timestamp, so it cannot invent one.

```
screenplays ──► parser ──► model / heuristic: propose candidates (event ids + 1-5 rubric scores, JSON)
                              │
                              ▼
                 validator (code): ids → exact times, snap to whole lines, 20-60s, hook ≤3s,
                 cut ≤4s after cliffhanger, overlaps, evidence quoted verbatim
                              │  errors → one repair round with the model → else rejected (logged)
                              ▼
                 scoring (config weights − measured penalties) → dedupe → show-wide ranking
                              │
                 cross-episode montage pass (model) · cold-viewer review pass (model, context-free)
                              ▼
                 results.json · summary.md · report.html · ranking.csv · human_review_sheet.csv · Streamlit UI
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ (uses `tomllib`). Tested on Python 3.12, Windows 11.

## Run it

```bash
# 1. No key needed: deterministic heuristic baseline (clearly labelled, low confidence)
python -m promo_finder run "path/to/screenplays" --provider heuristic --out outputs/my_show_heuristic

# 2. Claude (Anthropic API)
set ANTHROPIC_API_KEY=...        # macOS/Linux: export ANTHROPIC_API_KEY=...
python -m promo_finder check-provider --provider anthropic      # one tiny request to confirm the key/model
python -m promo_finder run "path/to/screenplays" --provider anthropic --out outputs/my_show_claude

# 2b. NVIDIA API catalog (free tier). Put NVIDIA_API_KEY=... in .env (copy .env.example), then:
python -m promo_finder list-models --config config/nvidia.toml          # model ids your key can use
python -m promo_finder check-provider --config config/nvidia.toml
python -m promo_finder run "path/to/screenplays" --config config/nvidia.toml --out outputs/my_show_nvidia

# 3. Any OpenAI-compatible endpoint, e.g. Google Gemini's free tier, Groq, OpenRouter, a local vLLM/Ollama,
#    or an internal Kuku gateway. Set the base_url / api_key_env / model in a config file:
python -m promo_finder run "path/to/screenplays" --config my_gemini.toml --out outputs/my_show_gemini

# 4. No API at all: file exchange. Prompts are written to <out>/exchange/requests/*.md; paste each into any
#    chat model, save its JSON answer as <out>/exchange/responses/<task>.json, and run the same command again.
python -m promo_finder run "path/to/screenplays" --provider file --responder "Gemini web app" --out outputs/my_show_manual

# Editor UI (open a finished run, or process new screenplays, re-weight live)
streamlit run app.py
```

`my_gemini.toml` (only the keys you want to override):

```toml
[provider]
name = "openai_compatible"
[provider.openai_compatible]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env = "GEMINI_API_KEY"
model = "<a current Gemini model id from Google AI Studio>"
```

Input can be one `.txt` file, a folder (searched recursively), or a `.zip`. Files that are not timed
screenplays are listed in the run manifest as skipped, never silently ignored. Try it on the bundled original
sample: `python -m promo_finder run samples/synthetic_show --provider heuristic --out outputs/sample`.

Exit code is 0 when every episode was processed, 2 when the run is incomplete (pending/failed episodes are
listed), 1 on a fatal error (e.g. bad credentials).

## Hosted demo (Streamlit Community Cloud)

The repo is deployable as is: `app.py` is the entry point, `requirements.txt` the dependencies (needs Python 3.11+;
pick 3.12 under "Advanced settings"). On a fresh deployment it opens the committed **demo runs** in `demo_runs/`, which
were produced from the original synthetic sample (no Kuku content): a live-model run (`nvidia/nemotron-3-ultra-550b-a55b`)
and the rule-based baseline. Reviewers can inspect candidates, compare, and re-weight live, and can upload their own
screenplay `.txt`/`.zip` and process it with the keyless heuristic provider. A model provider is offered in the UI only if its
key is configured on the server, so a public deployment never exposes a key. Do **not** add API keys to a public app's
secrets (anyone could spend them), and do not upload confidential scripts to it.

Steps: push the repo to GitHub → share.streamlit.io → New app → pick the repo, branch, `app.py` → Deploy.
Kuku's real results are deliberately not in the repo; they are shared separately as a private folder
(`report.html` in each run folder opens in any browser without installing anything).

## Processing a new show

No code changes are needed. Point `run` at the new folder. If its format differs, the parser records
anomalies per episode in `parse_report.json` (`python -m promo_finder parse <folder>` prints them). The parser
expects the input-pack format: header lines (`EPISODE NUMBER/LOGLINE/CONTINUITY NOTES`), `SCENE NN` blocks and
`**LABEL:**` + `Start: (MM:SS.ff) | End: (MM:SS.ff)` + text; labels `CAMERA`, `ACTION`, `SOUND FX`, anything
else is a speaker. Header keys are case-insensitive.

## Re-ranking without re-running the model

Sub-scores are stored per candidate, so weights and penalties can change after the fact at zero cost:

```bash
python -m promo_finder rerank outputs/my_show_claude/results.json --weights hook=0.35,cliffhanger=0.3 --out outputs/my_show_reweighted
```

The Streamlit sidebar does the same live.

## Scoring (all in `config/default.toml`)

`total = 100 × Σ wᵢ·(sᵢ−1)/4 / Σ wᵢ − penalties`

| criterion | default weight | what 5 means |
|---|---|---|
| hook | 0.25 | something in the first 3s stops a stranger mid-scroll |
| cliffhanger | 0.25 | the cut lands on an open, high-stakes question |
| standalone | 0.20 | a first-time viewer understands who is wronging whom and the stakes |
| emotion_curiosity | 0.15 | intense emotion plus a strong "what happens next" gap |
| visual | 0.08 | striking, phone-readable images (judged from CAM/ACT/SFX lines) |
| spoiler_safety | 0.07 | reveals only premise-level information (what the title already says) |

Penalties are only for things code measured: hook starts after 3.0s (−12), clip runs more than 4s past the
cliffhanger line (−6), dead air at the open (−5), each extra segment (−3), cross-episode montage (−3), boundary
snapped to finish a line (−1), claimed hook is only a camera description (−4), cold-viewer review could not follow the clip (−8), optional distance from
episode 1 (off by default). Confidence starts from the model's own rating and drops one level per measured
problem, repair round, or cold-viewer disagreement; heuristic picks are always "low". Ties break on
hook+cliffhanger, then the cold-viewer score (if reviewed), then fewer segments, then earlier episode.

Scores are editorial recommendations, not predictions of social-media performance: there is no engagement
data to calibrate against.

## Outputs (per run folder)

| file | what |
|---|---|
| `results.json` | everything: every candidate (valid and rejected) with segments, measured metrics, warnings, scores, penalties, evidence, transcript, review |
| `summary.md`, `report.html` | readable shortlist, alternatives with "why lower", duplicates, rejections, per-episode notes |
| `ranking.csv` | one row per ranked candidate |
| `human_review_sheet.csv` | blank Y/N columns for an editor to label the picks (the only route to real accuracy numbers) |
| `parse_report.json` | episodes found, events per type, durations, anomalies |
| `run_manifest.json` | provider, every model task with status/usage, pending/failed tasks, cost estimate, config snapshot |
| `raw_model_io/` | the exact request and unedited response for every model task |
| `exchange/` | file-provider prompts and answers |

## Tests

```bash
python -m pytest -q
```

39 tests: timestamps, parser (including case-insensitive headers and anomaly reporting), every validation
rule, boundary snapping, scoring/penalties/weights, dedupe, both API adapters against fake SDK clients (request
shape, refusal, truncation, missing credentials), file exchange (pending, stale-answer protection), caching,
and end-to-end runs (heuristic, file provider with a repair round, rerank, UI smoke test). The tests use only
the original synthetic sample in `samples/`. No Kuku content and no network.

## Confidentiality

The input pack is Kuku's content. Keep it outside this folder. `.gitignore` excludes `outputs/`, `.cache/`
and any `*.txt` outside `samples/`, because run outputs quote the screenplay. Share outputs privately.

## Layout

```
promo_finder/   parser, timecode, models, prompts (+JSON schemas), providers, heuristic, validate, scoring,
                pipeline, report, cli
config/         default.toml (weights, thresholds, provider settings, pricing for estimates)
app.py          Streamlit editor UI
samples/        original synthetic two-episode screenplay (not Kuku content)
tests/          pytest suite
dev/            helpers used while building (transcript viewer, answer serialiser, run comparison)
docs/           DECISIONS.md: design decisions and assumptions
```
