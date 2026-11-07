# Design decisions and assumptions

A record of the choices made while building Promo Finder, and why. No screenplay content here.

## What the input looks like (measured, not assumed)

- 50 `.txt` files, one per episode. Each has a header (episode number, logline, continuity notes), then
  `SCENE` blocks, then timed blocks `**LABEL:**` / `Start: (MM:SS.ff) | End: (MM:SS.ff)` / text.
- 3,727 timed events across 98 scenes (CAMERA 1,754 · dialogue ~1,500 · ACTION 372 · SOUND FX 114);
  102.7 minutes total; episodes run 1:10 to 3:20 (longer than the brief's "1-2 minutes").
- Timestamps restart at 00:00 in every episode. Events overlap (a camera shot and a line of dialogue share time).
- Irregularities the parser handles and records instead of hiding: episode 6 writes `SCENE HEADING/START/END`
  in upper case; one event in episode 22 ends 0.3s after its scene. No other timing anomalies (end before start,
  out-of-order events, gaps between scenes) were found in the 50 files.
- The same character is credited under different labels (a descriptive label before the name is used).
  The model is told this; output never rewrites labels.
- Episodes pick up exactly where the previous one ended (continuity notes say so). They are not verbatim
  recaps, so cross-episode duplication is about the *same beat*, not the same text.

## Assumptions (the brief was silent)

1. **Times are episode-relative**, so every segment is reported as (episode, start, end) in the screenplay's own
   `MM:SS.ff` format. An editor finds the frame in the episode file.
2. **The screenplay timing matches the final cut.** We only have scripts; if the edit drifts, the timestamps drift.
3. **A promo's "first 3 seconds" means the hook event must *start* within 3.0s of the clip start.** The clip may open
   a beat before the hook (on the shot that sets it up), but not on a slow establishing shot.
4. **"Ends on a cliffhanger" means the clip must end within 4.0s after the cliffhanger line ends.** 4s allows the
   reaction shot or sting; more than that usually means the tension has been released. Configurable.
5. **Never cut dialogue mid-sentence.** Boundaries snap outward to whole lines (recorded as an adjustment).
   Camera shots and sound may be cut; that is normal editing.
6. **The show title is public**, so anything the title itself gives away is premise, not spoiler. Spoiler risk is
   about payoffs (how and when the reveal happens, villains' defeat).
7. **Up to 3 segments** per promo, in story order within an episode; cross-episode montages are allowed but
   penalised slightly because there are no text overlays or voice-over to bridge time jumps.
8. **Platform policy** (graphic injury, sexual content) is flagged in weaknesses, not filtered: that is an editor call.

## Division of labour: code vs model vs human

| Code (deterministic, tested) | Model (judgment, structured JSON) | Human (editor) |
|---|---|---|
| parse, timestamps, durations | is this a hook for a stranger? | final pick and exact trim |
| ids → times, snap to whole lines | is the ending genuinely unresolved? | platform/brand suitability |
| 20-60s, hook ≤3s, cut ≤4s after cliffhanger | can a new viewer follow it? context needed? | labels in human_review_sheet.csv |
| overlap / ordering checks, evidence quoting | spoiler risk, visual potential, why it works | real engagement data later |
| weights, penalties, confidence rules, dedupe, ranking | cross-episode montage proposals; cold-viewer review | |

**Why event ids instead of timestamps from the model:** language models are unreliable with arithmetic and
precise times, and any timestamp they emit is unverifiable. With ids, every time in the output is copied from
the screenplay by code, and a hallucinated id fails validation instead of producing a wrong clip.

**Why the model also states `hook_id` and `cliffhanger_id`:** it turns two editorial claims into checkable facts.
If the claimed hook starts 3.6s in, or the clip keeps running 9s after the claimed cliffhanger, code measures
it, penalises it and lowers confidence. In the delivered live run it caught 10 late hooks and 4 over-long tails among 134 candidates.

**Why 1-5 sub-scores with written anchors:** models are poorly calibrated on 0-100. Anchored 1-5 scores with a
one-line reason each are explainable and can be re-weighted later. Ties are common, so the tie-break order is
explicit: hook+cliffhanger, fewer segments, earlier episode.

**Why these default weights:** hook and cliffhanger define a promo (0.25 each); standalone is the gate for new
viewers (0.20); emotion/curiosity (0.15) is what makes people care; visual (0.08) is judged from text
descriptions only, so it gets less trust; spoiler safety (0.07) is low because the title already reveals the
premise. All configurable, and `rerank` / the UI re-rank without new model calls.

**Why a repair round:** a structural failure (wrong duration, unknown id, hook outside the first segment) is sent
back to the model once with the exact error. If it still fails it is rejected and listed with the reason.

**Confidence is rule-based, not self-reported:** it starts from the model's self-rating and drops one level per
measured problem (late hook, cut after cliffhanger, dead air, non-chronological montage), per repair, and when
the context-free cold-viewer review disagrees. Heuristic picks are always low.

**De-duplication:** two clips are near-duplicates if they share at least 50% of the shorter clip's screen time,
or their dialogue vocabularies overlap (Jaccard ≥0.6). The shortlist also allows only one promo per episode for
variety. Known gap: two different scenes that play the *same beat* in different words (e.g. two near-identical "threat stops
inches away" endings in consecutive episodes) are not caught. Fixing that needs semantic comparison
(embeddings or a model pass over the shortlist).

**Cold-viewer review:** a second, independent model call that sees *only* the clip's own lines (no title,
logline, outline or episode number) and answers as a stranger: what did you understand, what confused you,
would the first 3 seconds stop you, is the ending open, would you watch on. It tests "standalone" far better
than the proposer grading itself. It is AI review, not human review, and is labelled as such.

**Montage pass:** one series-level call sees the top 20 single moments and may propose up to 3 cross-episode
promos (trailer style). They go through the same validation and scoring.

## Providers and what was actually run

- The model layer is one method, `generate_json(task_id, system, user, schema)`. Adapters: Anthropic SDK
  (structured JSON output, server-side refusal fallback, typed error handling), OpenAI-compatible Chat Completions
  (Gemini free tier, Groq, OpenRouter, NVIDIA's catalog, local servers, gateways; `json_mode` falls back from schema to
  object to prompt-only), and a file-exchange provider. `<think>` blocks are stripped before parsing. Identical prompts are
  cached on disk, so re-runs never pay twice. Keys come from environment variables or an ignored `.env` file.
- **Main run (live API):** `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA's OpenAI-compatible endpoint, free-tier key,
  `json_mode = "prompt"`. 50 mining calls, 1 montage call, 4 repair calls, 41 cold-viewer calls: 96 tasks, 0 failures,
  about 45 minutes at 4 workers (about 160 s per mining call; it is a slow reasoning model).
- **Second opinion:** Claude Opus 5.5 answered the exported prompts in the build session (file exchange). Not an API
  call by the tool, and the responder had read the whole show.
- The Anthropic adapter is tested only against fake SDK clients; no Anthropic key was available. The OpenAI-compatible
  adapter is verified live.
- **Changes after inspecting the first live pass** (original preserved in `nvidia_full_first_pass`): wider cold-viewer
  review (12 to 40, and every shortlisted pick); cold-viewer score as a tie-breaker; a "camera-only hook" rule (-4).
  Reason: self-scores saturated (8 clips at exactly 100) and six first-shortlist hooks were camera descriptions.
- Measured on the live run: 505k input and 370k output tokens (including hidden reasoning) per 50-episode show;
  about $9 at Claude Opus 5.5 list prices, for comparison only. Output was roughly 2.8x a visible-text projection.

## Cost and scale

- Mining prompts average ~24k characters (~6k tokens: instructions + series outline + the full timed transcript).
  Work is parallel per episode (`workers`); caching makes re-runs and new episodes cost only what changed.
- For hundreds of shows: run per show as scripts land; a cheap non-reasoning model for mining, a different, stronger
  model as the cold viewer on the shortlist only; editor labels from the review sheet and post performance fed back so
  weights are fitted instead of hand-set; batch APIs for backlogs.

## Known limitations

- Judges scripts, not footage: acting, pacing, music and actual visuals are invisible to it.
- Self-scoring is optimistic: the proposing model rates its own picks (in the live run 85/134 hooks and 103/134
  emotion scores are 5). The cold-viewer pass (same model family, so correlated) and human labels are the counterweights.
- No ground-truth labels, so no accuracy/precision/recall is reported. `human_review_sheet.csv` exists to create them.
- Semantic near-duplicates across episodes are not detected.
- Boundary snapping protects dialogue only; a cut can still clip the tail of a sound effect.
- The heuristic baseline has no story understanding; it over-rewards episode endings and music stings.
