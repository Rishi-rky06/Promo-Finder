"""Turn a model (or heuristic) candidate into exact, checked segments.

Everything here is deterministic and measurable:
- event ids -> exact start/end times taken from the screenplay
- boundaries snapped so no dialogue line is cut mid-sentence (recorded as an adjustment)
- total duration, hook timing, cliffhanger placement, dead air, overlaps, ordering
- hook / cliffhanger / evidence text is quoted verbatim from the screenplay, never from the model

``errors`` make a candidate invalid (it goes to repair, then rejection).
``warnings`` keep it valid but feed deterministic penalties and review flags.
"""
from __future__ import annotations

import re

from .models import Episode, Event, Segment
from .prompts import SCORE_KEYS
from .timecode import format_tc

TOL = 0.05
_REF = re.compile(r"^\s*(?:E0*(\d+)\s*[-:_/ ]\s*)?(L\d+)\s*$", re.I)


def parse_ref(ref: str, default_episode: int | None) -> tuple[int | None, str | None]:
    m = _REF.match(str(ref or ""))
    if not m:
        return None, None
    ep = int(m.group(1)) if m.group(1) else default_episode
    num = int(m.group(2)[1:])
    return ep, f"L{num:03d}"


def _events_in(ep: Episode, start: float, end: float) -> list[Event]:
    return [e for e in ep.events if e.start >= start - TOL and e.start < end - TOL]


def _snap(ep: Episode, start: float, end: float) -> tuple[float, float, list[str]]:
    """Never cut a line of dialogue mid-sentence."""
    notes = []
    for _ in range(3):
        moved = False
        for e in ep.events:
            if e.kind != "dialogue":
                continue
            if e.start < start - TOL and e.end > start + 0.15:
                notes.append(f"start moved {format_tc(start)} -> {format_tc(e.start)} to include all of {e.id}")
                start, moved = e.start, True
            if e.start < end - 0.15 and e.end > end + TOL:
                notes.append(f"end moved {format_tc(end)} -> {format_tc(e.end)} to finish {e.id}")
                end, moved = e.end, True
        if not moved:
            break
    return start, end, notes


def resolve(cand: dict, episodes: dict[int, Episode], cfg: dict, *, cid: str, origin: str,
            default_episode: int | None) -> dict:
    p = cfg["promo"]
    errors: list[str] = []
    warnings: list[dict] = []

    def warn(code: str, message: str, value: float | None = None) -> None:
        warnings.append({"code": code, "message": message, "value": value})

    segments: list[Segment] = []
    adjustments: list[str] = []
    raw_segments = cand.get("segments") or []
    if not raw_segments:
        errors.append("no segments given")
    if len(raw_segments) > p["max_segments"]:
        errors.append(f"{len(raw_segments)} segments; at most {p['max_segments']} allowed")

    for i, s in enumerate(raw_segments, start=1):
        seg_ep = s.get("episode") or default_episode
        if default_episode is not None and origin != "montage" and seg_ep != default_episode:
            errors.append(f"segment {i} refers to episode {seg_ep}, but this candidate belongs to episode "
                          f"{default_episode}")
            continue
        ep_a, sid = parse_ref(s.get("start_id"), seg_ep)
        ep_b, eid = parse_ref(s.get("end_id"), seg_ep)
        ep = episodes.get(ep_a) if ep_a is not None else None
        if ep is None or ep_a != ep_b:
            errors.append(f"segment {i}: cannot resolve episode for ids {s.get('start_id')!r}..{s.get('end_id')!r}")
            continue
        a, b = ep.event(sid or ""), ep.event(eid or "")
        if a is None or b is None:
            errors.append(f"segment {i}: unknown event id(s) {s.get('start_id')!r}/{s.get('end_id')!r} in episode {ep.number}")
            continue
        start, end = a.start, b.end
        if end <= start:
            errors.append(f"segment {i}: end {eid} ({format_tc(end)}) is not after start {sid} ({format_tc(start)})")
            continue
        start, end, notes = _snap(ep, start, end)
        adjustments += [f"segment {i}: {n}" for n in notes]
        segments.append(Segment(episode=ep.number, start_id=a.id, end_id=b.id, start=start, end=end))

    # overlaps / order
    for i, x in enumerate(segments):
        for y in segments[i + 1:]:
            if x.episode == y.episode and x.start < y.end - TOL and y.start < x.end - TOL:
                errors.append(f"segments overlap in episode {x.episode} ({format_tc(x.start)}-{format_tc(x.end)} and "
                              f"{format_tc(y.start)}-{format_tc(y.end)})")
    for x, y in zip(segments, segments[1:]):
        if (y.episode, y.start) < (x.episode, x.start):
            warn("non_chronological", "segments are not in story order; check the cut reads correctly")

    total = sum(s.duration for s in segments)
    if segments and not (p["min_seconds"] - TOL <= total <= p["max_seconds"] + TOL):
        errors.append(f"total duration {total:.1f}s is outside {p['min_seconds']:.0f}-{p['max_seconds']:.0f}s")

    # Build the clip transcript on a clip-relative clock.
    transcript, offset = [], 0.0
    seg_offsets = []
    for idx, s in enumerate(segments):
        ep = episodes[s.episode]
        seg_offsets.append(offset)
        for e in _events_in(ep, s.start, s.end):
            transcript.append({"episode": s.episode, "id": e.id, "kind": e.kind, "segment": idx + 1,
                               "clip_time": round(offset + max(0.0, e.start - s.start), 2),
                               "render": e.render(), "speaker": e.speaker, "text": e.text})
        offset += s.duration

    def locate(ref: str, which: str) -> tuple[Event | None, int | None]:
        """Find the referenced event and the index of the segment containing it."""
        fallback_ep = segments[0].episode if which == "hook" and segments else (
            segments[-1].episode if segments else default_episode)
        ep_n, eid = parse_ref(ref, fallback_ep)
        ep = episodes.get(ep_n) if ep_n is not None else None
        ev = ep.event(eid) if ep and eid else None
        if ev is None:
            return None, None
        for idx, s in enumerate(segments):
            if s.episode == ep_n and s.start - TOL <= ev.start < s.end - TOL:
                return ev, idx
        return ev, None

    metrics: dict = {"total_duration": round(total, 2), "segments": len(segments)}
    hook_line = cliff_line = None
    if segments:
        hook_ev, hook_idx = locate(cand.get("hook_id", ""), "hook")
        if hook_ev is None:
            errors.append(f"hook_id {cand.get('hook_id')!r} does not exist")
        elif hook_idx != 0:
            errors.append(f"hook_id {hook_ev.id} is not inside the first segment, so it cannot be the opening hook")
        else:
            hook_line = hook_ev.render()
            if hook_ev.kind == "camera":
                warn("camera_only_hook", "the claimed hook is a camera-framing description, not a line, action or sound "
                     "a scroller would react to; check the opening actually lands")
            off = hook_ev.start - segments[0].start
            metrics["hook_offset"] = round(off, 2)
            if off > p["hook_window_seconds"] + TOL:
                warn("late_hook", f"hook line starts {off:.1f}s into the clip (limit {p['hook_window_seconds']:.1f}s)", off)

        cliff_ev, cliff_idx = locate(cand.get("cliffhanger_id", ""), "cliff")
        if cliff_ev is None:
            errors.append(f"cliffhanger_id {cand.get('cliffhanger_id')!r} does not exist")
        elif cliff_idx != len(segments) - 1:
            errors.append(f"cliffhanger_id {cliff_ev.id} is not inside the last segment, so the clip does not cut on it")
        else:
            cliff_line = cliff_ev.render()
            tail = segments[-1].end - cliff_ev.end
            metrics["cliffhanger_tail"] = round(tail, 2)
            if tail > p["cliffhanger_tail_seconds"] + TOL:
                after = [t for t in transcript if t["segment"] == len(segments) and t["kind"] == "dialogue"
                         and episodes[segments[-1].episode].event(t["id"]).start >= cliff_ev.end - TOL]
                extra = f"; {len(after)} more dialogue line(s) play after it" if after else ""
                warn("cut_after_cliffhanger", f"clip runs {tail:.1f}s past the cliffhanger line{extra}", tail)

        beats = [t["clip_time"] for t in transcript if t["kind"] != "camera"]
        dlg = [t for t in transcript if t["kind"] == "dialogue"]
        metrics["first_beat_offset"] = beats[0] if beats else None
        metrics["first_dialogue_offset"] = dlg[0]["clip_time"] if dlg else None
        if beats and beats[0] > p["dead_air_open_seconds"]:
            warn("dead_air_open", f"nothing but camera movement for the first {beats[0]:.1f}s", beats[0])
        metrics["dialogue_lines"] = len(dlg)
        metrics["speakers"] = len({t["speaker"] for t in dlg})
        # longest stretch with no dialogue starting (picture-only stretches are fine in moderation)
        times = [0.0] + [t["clip_time"] for t in dlg] + [total]
        metrics["longest_gap_without_dialogue"] = round(max(b - a for a, b in zip(times, times[1:])), 2) if total else 0
    if adjustments:
        warn("boundary_adjusted", "; ".join(adjustments))

    # evidence is quoted from the screenplay, never from the model
    evidence = []
    for ref in cand.get("evidence_ids") or []:
        ep_n, eid = parse_ref(ref, segments[0].episode if segments else default_episode)
        ep = episodes.get(ep_n) if ep_n is not None else None
        ev = ep.event(eid) if ep and eid else None
        if ev is None:
            warn("bad_evidence", f"evidence id {ref!r} does not exist (ignored)")
            continue
        inside = any(s.episode == ep_n and s.start - TOL <= ev.start < s.end - TOL for s in segments)
        evidence.append({"episode": ep_n, "id": ev.id, "render": ev.render(), "inside_clip": inside})
        if not inside:
            warn("evidence_outside_clip", f"evidence {ev.id} (ep {ep_n}) is not inside the clip")

    scores = {}
    raw_scores = cand.get("scores") or {}
    for k in SCORE_KEYS:
        item = raw_scores.get(k)
        val = item.get("score") if isinstance(item, dict) else item
        if not isinstance(val, (int, float)):
            errors.append(f"missing score '{k}'")
            continue
        clamped = int(min(5, max(1, round(val))))
        if clamped != val:
            warn("score_clamped", f"score {k}={val} clamped to {clamped}")
        scores[k] = {"score": clamped, "reason": item.get("reason", "") if isinstance(item, dict) else ""}

    return {
        "id": cid,
        "origin": origin,
        "episodes": sorted({s.episode for s in segments}),
        "title": cand.get("title", ""),
        "segments": [s.to_dict() for s in segments],
        "metrics": metrics,
        "hook": cand.get("hook", ""),
        "hook_line": hook_line,
        "cliffhanger": cand.get("cliffhanger", ""),
        "cliffhanger_line": cliff_line,
        "why_it_works": cand.get("why_it_works", ""),
        "stranger_view": cand.get("stranger_view", ""),
        "evidence": evidence,
        "scores": scores,
        "context_dependencies": cand.get("context_dependencies") or [],
        "spoiler_risk": cand.get("spoiler_risk", "medium"),
        "spoiler_note": cand.get("spoiler_note", ""),
        "weaknesses": cand.get("weaknesses") or [],
        "model_confidence": cand.get("confidence", "low"),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
        "transcript": transcript,
        "raw_candidate": cand,
    }
