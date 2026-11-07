"""Transparent scoring, confidence, de-duplication and ranking.

score = 100 * sum_i w_i * (s_i - 1) / 4 / sum_i w_i  -  deterministic penalties

* s_i are the 1-5 editorial sub-scores (model or heuristic).
* weights and penalties come from config, so an editor can re-rank without re-running the model.
* penalties are only for things code measured (late hook, cut after the cliffhanger, dead air, extra
  segments, boundary adjustments, cold-viewer confusion), so each point lost has a visible reason.
"""
from __future__ import annotations

import re

from .prompts import SCORE_KEYS

LEVELS = ["low", "medium", "high"]
WARNING_PENALTY = {"late_hook": "late_hook", "cut_after_cliffhanger": "cut_after_cliffhanger",
                   "dead_air_open": "dead_air_open", "boundary_adjusted": "boundary_adjusted",
                   "camera_only_hook": "camera_only_hook"}


def score_candidate(c: dict, cfg: dict) -> dict:
    w = cfg["weights"]
    pen = cfg["penalties"]
    wsum = sum(w.get(k, 0) for k in SCORE_KEYS)
    base = 100 * sum(w.get(k, 0) * (c["scores"][k]["score"] - 1) / 4 for k in SCORE_KEYS) / wsum
    penalties = []
    seen = set()
    for warn in c["warnings"]:
        key = WARNING_PENALTY.get(warn["code"])
        if key and key not in seen and pen.get(key):
            seen.add(key)
            penalties.append({"code": key, "points": pen[key], "reason": warn["message"]})
    extra = max(0, c["metrics"].get("segments", 1) - 1)
    if extra and pen.get("per_extra_segment"):
        penalties.append({"code": "extra_segments", "points": pen["per_extra_segment"] * extra,
                          "reason": f"{extra + 1} segments (continuous clips preferred)"})
    if len(c["episodes"]) > 1 and pen.get("cross_episode"):
        penalties.append({"code": "cross_episode", "points": pen["cross_episode"],
                          "reason": f"joins episodes {c['episodes']}; harder to follow without overlays"})
    cv = c.get("cold_viewer")
    if cv and cv.get("status") == "ok" and not cv["result"].get("could_follow_without_context"):
        penalties.append({"code": "cold_viewer_confused", "points": pen.get("cold_viewer_confused", 0),
                          "reason": "context-free AI review could not follow the clip"})
    depth = pen.get("episode_depth_per_10", 0) or 0
    if depth and c["episodes"] and min(c["episodes"]) > 1:
        penalties.append({"code": "episode_depth", "points": round(depth * (min(c["episodes"]) - 1) / 10, 2),
                          "reason": f"starts in episode {min(c['episodes'])}; new viewers land on episode 1"})
    total = max(0.0, base - sum(p["points"] for p in penalties))
    c["score"] = {"base": round(base, 1), "penalties": penalties, "total": round(total, 1),
                  "weights": {k: w.get(k, 0) for k in SCORE_KEYS}}
    c["confidence"], c["confidence_reasons"] = confidence(c)
    c["review_flags"] = review_flags(c)
    c["needs_human_review"] = bool(c["review_flags"])
    return c


def confidence(c: dict) -> tuple[str, list[str]]:
    """Rule-based: start from the model's own confidence and step down for each measured problem."""
    reasons = []
    if c["origin"] == "heuristic":
        return "low", ["heuristic fallback: keyword/structure rules, no model judgment"]
    level = LEVELS.index(c.get("model_confidence", "low")) if c.get("model_confidence") in LEVELS else 0
    reasons.append(f"model self-rated {c.get('model_confidence')}")
    for warn in c["warnings"]:
        if warn["code"] in ("late_hook", "cut_after_cliffhanger", "dead_air_open", "non_chronological"):
            level -= 1
            reasons.append(f"-1: {warn['code']}")
    if c.get("repaired"):
        level -= 1
        reasons.append("-1: needed a repair round to pass validation")
    cv = c.get("cold_viewer")
    if cv and cv.get("status") == "ok":
        r = cv["result"]
        disagrees = (not r.get("could_follow_without_context") and c["scores"]["standalone"]["score"] >= 4) or \
                    (not r.get("ending_leaves_open_question") and c["scores"]["cliffhanger"]["score"] >= 4) or \
                    (r.get("first_3_seconds_stop_scrolling", 3) <= 2 and c["scores"]["hook"]["score"] >= 4)
        if disagrees:
            level -= 1
            reasons.append("-1: cold-viewer review disagrees with the editorial scores")
        else:
            reasons.append("cold-viewer review agrees")
    level = max(0, min(2, level))
    return LEVELS[level], reasons


def review_flags(c: dict) -> list[str]:
    flags = []
    if c["confidence"] == "low":
        flags.append("low confidence")
    if c.get("spoiler_risk") == "high":
        flags.append("high spoiler risk")
    if c["scores"]["standalone"]["score"] <= 2:
        flags.append("needs context to follow")
    cv = c.get("cold_viewer")
    if cv and cv.get("status") == "ok" and cv["result"].get("confusing_points"):
        if not cv["result"].get("could_follow_without_context"):
            flags.append("cold viewer confused")
    if any(w["code"] == "late_hook" for w in c["warnings"]):
        flags.append("hook arrives late")
    return flags


_WORD = re.compile(r"[a-z']{3,}")


def dialogue_tokens(c: dict) -> set[str]:
    return {w for t in c["transcript"] if t["kind"] == "dialogue" for w in _WORD.findall(t["text"].lower())}


def _overlap_ratio(a: dict, b: dict) -> float:
    """Overlap of shared screen time divided by the shorter clip (1.0 = one clip contains the other)."""
    shared = 0.0
    for x in a["segments"]:
        for y in b["segments"]:
            if x["episode"] == y["episode"]:
                shared += max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    shorter = min(a["metrics"]["total_duration"], b["metrics"]["total_duration"]) or 1
    return shared / shorter


def near_duplicate(a: dict, b: dict, cfg: dict) -> str | None:
    r = cfg["ranking"]
    ov = _overlap_ratio(a, b)
    if ov >= r["same_episode_overlap"]:
        return f"{ov:.0%} of the shorter clip's screen time is shared"
    ta, tb = dialogue_tokens(a), dialogue_tokens(b)
    if ta and tb:
        j = len(ta & tb) / len(ta | tb)
        if j >= r["text_similarity"]:
            return f"dialogue overlap (Jaccard {j:.2f})"
    return None


def sort_key(c: dict) -> tuple:
    """Total score first. Ties (common with 1-5 sub-scores) break on: stronger hook+cliffhanger (the two
    defining promo traits), the independent cold-viewer score (if reviewed), fewer segments (continuous preferred), then earlier episode (a new viewer who
    clicks starts at episode 1, so earlier clips carry less context debt), then id for determinism."""
    cv = c.get("cold_viewer") or {}
    cv_score = 0
    if cv.get("status") == "ok":  # independent stranger-view: first-3s + would-watch-next (both 1-5)
        cv_score = cv["result"].get("first_3_seconds_stop_scrolling", 0) + cv["result"].get("would_watch_next_episode", 0)
    return (-c["score"]["total"],
            -(c["scores"]["hook"]["score"] + c["scores"]["cliffhanger"]["score"]),
            -cv_score,
            c["metrics"].get("segments", 1),
            min(c["episodes"]) if c["episodes"] else 0,
            c["id"])


def rank(cands: list[dict], cfg: dict) -> dict:
    """Score, de-duplicate and rank valid candidates across the whole show."""
    valid = [score_candidate(c, cfg) for c in cands if c["valid"]]
    valid.sort(key=sort_key)
    kept: list[dict] = []
    duplicates: list[dict] = []
    for c in valid:
        dup = next(((k, why) for k in kept if (why := near_duplicate(c, k, cfg))), None)
        if dup:
            c["duplicate_of"] = dup[0]["id"]
            c["duplicate_reason"] = dup[1]
            duplicates.append(c)
        else:
            c["duplicate_of"] = None
            kept.append(c)

    r = cfg["ranking"]
    shortlist, per_ep = [], {}
    for c in kept:
        if len(shortlist) >= r["shortlist_size"]:
            break
        key = tuple(c["episodes"]) if len(c["episodes"]) == 1 else None
        if key is not None and per_ep.get(key, 0) >= r["max_per_episode_in_shortlist"]:
            continue
        shortlist.append(c)
        if key is not None:
            per_ep[key] = per_ep.get(key, 0) + 1
    short_ids = {c["id"] for c in shortlist}
    alternatives = [c for c in kept if c["id"] not in short_ids]
    for i, c in enumerate(kept, start=1):
        c["overall_rank"] = i
    for i, c in enumerate(shortlist, start=1):
        c["shortlist_rank"] = i
    for c in alternatives:
        c["why_lower"] = why_lower(c, shortlist, per_ep, cfg)
    for c in duplicates:
        c["why_lower"] = [f"near-duplicate of {c['duplicate_of']}: {c['duplicate_reason']}"]
    return {"shortlist": shortlist, "alternatives": alternatives, "duplicates": duplicates, "ranked": kept}


def why_lower(c: dict, shortlist: list[dict], per_ep: dict, cfg: dict) -> list[str]:
    reasons = []
    key = tuple(c["episodes"])
    same = next((s for s in shortlist if tuple(s["episodes"]) == key), None)
    if same is not None:
        reasons.append(f"same episode as shortlisted {same['id']} (variety cap of "
                       f"{cfg['ranking']['max_per_episode_in_shortlist']} per episode)")
    if shortlist:
        floor = shortlist[-1]
        weaker = [f"{k} {c['scores'][k]['score']} vs {floor['scores'][k]['score']}"
                  for k in SCORE_KEYS if c["scores"][k]["score"] < floor["scores"][k]["score"]]
        if weaker:
            reasons.append(f"weaker than the last shortlisted pick ({floor['id']}) on: " + ", ".join(weaker))
    for p in c["score"]["penalties"]:
        reasons.append(f"-{p['points']:g} {p['code']}: {p['reason']}")
    return reasons or ["lower total score"]
