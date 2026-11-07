"""Rule-based fallback miner used when no model is available.

This is a clearly-labelled BASELINE, not an editor. It knows nothing about story,
only surface signals that tend to correlate with promo moments in microdrama scripts:
intense delivery notes, exclamations/questions, physical-action verbs, sound stings,
and the fact that episodes are written to end on a cliffhanger.

It returns candidates in the same JSON shape as the model, so the deterministic
validation, scoring and ranking are shared. All its candidates get confidence "low".
"""
from __future__ import annotations

import re

from .models import Episode, Event

STRONG_WORDS = set("""
slap slaps slapped strike strikes struck hit shove shoves shoved grab grabs grabbed seize seized drag drags dragged
collapse collapses collapsed burst bursts slam slams slammed explode explodes explosion fire flames blaze burning
blood bleeding bleed wound knife blade sword dagger chains chained shackled gallows hang hanged execution execute
poison poisoned poisoning arsenic witch witchcraft arrest arrested guards dungeon prison dead die dying kill killed
murder murderer liar lie lies traitor betrayed betrayal married marry marrying wedding altar ring pregnant divorce
thief theft steal stole stolen kneel kneels scream screams screaming sob sobs sobbing tears slapped whip crop
""".split())
THREAT_WORDS = set("will unless else before never die hang burn kill dead gallows tomorrow tonight dawn".split())
CONTEXT_WORDS = {"remember", "again", "yesterday", "earlier", "promised", "before", "still", "back"}
INTENSE_NOTES = re.compile(r"scream|shout|furious|outrag|sob|cry|crying|desperat|menac|threat|gloat|mock|cruel|"
                           r"horrif|shock|terrif|pani|enrag|venom|seeth|cold|vicious|livid|bellow|roar|defian|"
                           r"broken|devastat|plead|beg", re.I)
STING_SFX = re.compile(r"cliffhanger|sting|swell|drone|ominous|cut to black|crack|slam|smash|explos|thunder|boom", re.I)
CLOSE_SHOT = re.compile(r"close-up|close up|insert|extreme close|push-in|push in", re.I)
_WORD = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _event_hook_value(e: Event) -> float:
    v = 0.0
    words = set(_words(e.text))
    if e.kind == "dialogue":
        v += 0.6
        if "!" in e.text:
            v += 1.0
        if "?" in e.text:
            v += 0.4
        if INTENSE_NOTES.search(e.notes):
            v += 1.0
    elif e.kind == "action":
        v += 0.3
    elif e.kind == "sfx" and STING_SFX.search(e.text):
        v += 0.8
    v += 1.2 * min(2, len(words & STRONG_WORDS))
    return v


def _end_value(e: Event, ep: Episode) -> float:
    v = 0.0
    text = e.text.strip().strip('"').strip()
    if ep.duration - e.end <= 2.5:
        v += 2.0  # episodes are written to end on their cliffhanger
    if e.kind == "sfx" and STING_SFX.search(e.text):
        v += 1.5
    if e.kind == "dialogue":
        if text.endswith("?"):
            v += 1.0
        if text.endswith("...") or text.endswith("—") or text.endswith("-"):
            v += 1.0
        if text.endswith("!"):
            v += 0.5
        if set(_words(text)) & THREAT_WORDS:
            v += 1.0
    return v


def _bucket(x: float, edges: tuple[float, float, float, float]) -> int:
    return 1 + sum(x >= t for t in edges)


def mine_episode(ep: Episode, cfg: dict, series_len: int, n: int | None = None) -> dict:
    p = cfg["promo"]
    n = n or p["candidates_per_episode"]
    lo, hi = max(p["min_seconds"], 25.0), min(p["max_seconds"], 55.0)
    beats = [e for e in ep.events if e.kind != "camera"]
    options = []
    for i, s in enumerate(beats):
        window3 = [e for e in ep.events if s.start - 0.05 <= e.start <= s.start + p["hook_window_seconds"]]
        hook_ev = max(window3, key=_event_hook_value)
        hook_v = sum(_event_hook_value(e) for e in window3)
        opener = _words(s.text)[:1]
        if s.kind == "dialogue" and opener and opener[0] in {"and", "but", "yes", "no", "so", "then"}:
            hook_v -= 0.8  # opens mid-exchange: likely needs context
        for e in beats[i + 1:]:
            dur = e.end - s.start
            if dur < lo:
                continue
            if dur > hi:
                break
            inside = [x for x in ep.events if s.start - 0.05 <= x.start < e.end]
            dlg = [x for x in inside if x.kind == "dialogue"]
            intensity = sum(1 for x in dlg if INTENSE_NOTES.search(x.notes)) / max(1, len(dlg))
            strong = sum(len(set(_words(x.text)) & STRONG_WORDS) for x in inside)
            visual = sum(1 for x in inside if x.kind == "camera" and CLOSE_SHOT.search(x.text)) + \
                sum(1 for x in inside if x.kind == "action" and set(_words(x.text)) & STRONG_WORDS)
            speakers = len({x.speaker for x in dlg})
            ctx = sum(1 for x in dlg if set(_words(x.text)) & CONTEXT_WORDS)
            end_v = _end_value(e, ep)
            standalone = 3.5 - 0.5 * max(0, speakers - 3) - 0.4 * ctx
            total = 1.0 * hook_v + 1.0 * end_v + 2.0 * intensity + 0.15 * strong + 0.08 * visual + 0.4 * standalone
            options.append((total, s, e, hook_ev, hook_v, end_v, intensity, visual / max(1.0, dur / 10), standalone))
    options.sort(key=lambda o: -o[0])

    chosen = []
    for opt in options:
        _, s, e, *_ = opt
        if all(e.end <= c[1].start or s.start >= c[2].end for c in chosen):
            chosen.append(opt)
        if len(chosen) >= n:
            break

    position = ep.number / max(1, series_len)
    candidates = []
    for total, s, e, hook_ev, hook_v, end_v, intensity, vis_rate, standalone in chosen:
        spoiler = 5 if position < 0.3 else 4 if position < 0.6 else 3 if position < 0.8 else 2
        scores = {
            "hook": _bucket(hook_v, (1.0, 2.0, 3.0, 4.0)),
            "cliffhanger": _bucket(end_v, (0.5, 1.5, 2.5, 3.5)),
            "standalone": max(1, min(5, round(standalone))),
            "emotion_curiosity": _bucket(intensity, (0.15, 0.3, 0.5, 0.7)),
            "visual": _bucket(vis_rate, (0.5, 1.0, 1.5, 2.5)),
            "spoiler_safety": spoiler,
        }
        reason = "rule-based estimate (see heuristic.py)"
        candidates.append({
            "title": f"Ep {ep.number} rule-based pick @ {s.id}",
            "segments": [{"episode": ep.number, "start_id": s.id, "end_id": e.id}],
            "hook_id": hook_ev.id,
            "hook": f"Opens on: {hook_ev.text[:140]}",
            "cliffhanger_id": e.id,
            "cliffhanger": f"Cuts on: {e.text[:140]}",
            "why_it_works": (f"Heuristic: opening-signal {hook_v:.1f}, ending-signal {end_v:.1f}, "
                             f"intense delivery on {intensity:.0%} of lines. No story understanding."),
            "stranger_view": "Not assessed (heuristic has no story understanding).",
            "evidence_ids": [hook_ev.id, e.id],
            "scores": {k: {"score": v, "reason": reason} for k, v in scores.items()},
            "context_dependencies": ["not assessed by heuristic"],
            "spoiler_risk": {5: "low", 4: "low", 3: "medium", 2: "high"}[spoiler],
            "spoiler_note": f"estimated from episode position ({ep.number}/{series_len}) only",
            "weaknesses": ["keyword/structure heuristic: cannot judge story clarity or whether the ending is truly unresolved"],
            "confidence": "low",
        })
    return {"episode_note": f"heuristic baseline: {len(options)} windows scored", "candidates": candidates}
