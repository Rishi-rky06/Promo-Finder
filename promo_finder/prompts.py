"""Prompts and JSON schemas for the model layer.

Design rule: the model never writes timestamps. It refers to screenplay event ids
(e.g. ``L012``); code turns ids into exact times, measures durations and checks
every constraint. The model's job is the editorial judgment code cannot do:
is this a hook, is this ending unresolved, would a stranger follow it.
"""
from __future__ import annotations

from .models import Episode, Event
from .timecode import format_tc

SCORE_KEYS = ["hook", "cliffhanger", "standalone", "emotion_curiosity", "visual", "spoiler_safety"]

RUBRIC = """\
Score each criterion 1-5 (integers). Anchors:
- hook (first 3 seconds of the clip): 5 = a line, action or image that would stop a stranger mid-scroll with zero context
  (public humiliation, slap, shocking accusation, threat, reveal). 3 = intriguing but takes a few seconds to land.
  1 = establishing shot, calm talk, or a line that only matters if you already know the characters.
- cliffhanger (the cut): 5 = cuts on an open, high-stakes question (threat mid-delivery, reveal half-made, decision
  pending, danger arriving). 3 = some tension, but partly resolved or soft. 1 = the conflict resolves or the clip fades out.
- standalone: 5 = a first-time viewer understands who is wronging whom and what is at stake from the clip alone.
  3 = gist is clear but key relationships or stakes are missing. 1 = confusing without earlier episodes.
- emotion_curiosity: 5 = intense emotion AND a strong "what happens next / who is this" gap. 1 = flat or routine.
- visual: judged only from CAM/ACT/SFX descriptions. 5 = striking, readable images on a phone (tight faces, physical
  action, spectacle such as fire, chains, crowds). 1 = static, dark, or talky.
- spoiler_safety: 5 = reveals only premise-level information (the show title is public, so anything the title says is
  not a spoiler). 1 = gives away a major twist or payoff that is the reason to keep watching.
"""

MINING_SYSTEM = f"""\
You are a senior short-form promo editor for a vertical microdrama app. You pick 20-60 second clips from a show's
own footage to post on TikTok/Instagram/YouTube Shorts to pull in NEW viewers who have never seen the show.
You will read one episode's screenplay. Every timed event has an id (L001, L002, ...), a start-end time, a type
(CAM camera shot, ACT action, DIA dialogue, SFX sound) and its text.

A promo must:
1. HOOK: something in the first 3 seconds makes a scrolling stranger stop (slap, reveal, shocking line,
   confrontation, unexpected action, raw emotion). The hook event must START within 3.0 seconds of the clip start,
   so start the clip on, or at most a beat before, the hook. Never open on a slow establishing shot.
2. CLIFFHANGER: it cuts at an unresolved moment. End the clip ON the cliffhanger line/reaction, not after the
   tension has been released.
3. STANDALONE: it makes sense to someone who has never seen the show.
4. LENGTH: 20-60 seconds in total (prefer roughly 25-50). Duration = start time of the start event to end time of
   the end event; work it out from the timestamps shown.
5. Prefer ONE continuous segment. Use 2-3 segments (in story order, same episode) only when cutting out a lull or
   joining a setup to its payoff makes the promo clearly stronger.

Judge the whole sequence as a stranger would see it. Do not pick a clip just because it contains dramatic words.
Character labels in the screenplay can vary for the same person (e.g. a character may be credited by description
before their name is used); a viewer only sees faces, so judge clarity on screen, not labels.

{RUBRIC}
Rules:
- Return up to {{n}} candidates for this episode, best first. They must be DIFFERENT moments, not overlapping variants.
- Return zero candidates if nothing in the episode would work as a promo. Saying so is useful.
- Refer to the screenplay only by event ids. Use only ids that appear in the transcript. Never invent dialogue,
  events or times. hook_id is the event that hooks; cliffhanger_id is the event the clip cuts on.
- evidence_ids: 2-5 ids of the lines that justify your judgment.
- Be candid in weaknesses and context_dependencies: an editor will rely on them.
"""

_SCORE_OBJ = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "episode": {"type": "integer"},
                    "start_id": {"type": "string"},
                    "end_id": {"type": "string"},
                },
                "required": ["episode", "start_id", "end_id"],
                "additionalProperties": False,
            },
        },
        "hook_id": {"type": "string"},
        "hook": {"type": "string"},
        "cliffhanger_id": {"type": "string"},
        "cliffhanger": {"type": "string"},
        "why_it_works": {"type": "string"},
        "stranger_view": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "scores": {
            "type": "object",
            "properties": {k: _SCORE_OBJ for k in SCORE_KEYS},
            "required": SCORE_KEYS,
            "additionalProperties": False,
        },
        "context_dependencies": {"type": "array", "items": {"type": "string"}},
        "spoiler_risk": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "spoiler_note": {"type": "string"},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["title", "segments", "hook_id", "hook", "cliffhanger_id", "cliffhanger", "why_it_works",
                 "stranger_view", "evidence_ids", "scores", "context_dependencies", "spoiler_risk",
                 "spoiler_note", "weaknesses", "confidence"],
    "additionalProperties": False,
}

MINING_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_note": {"type": "string"},
        "candidates": {"type": "array", "items": CANDIDATE_SCHEMA},
    },
    "required": ["episode_note", "candidates"],
    "additionalProperties": False,
}


def render_transcript(events: list[Event], prefix: str = "") -> str:
    return "\n".join(prefix + e.render() for e in events)


def mining_prompt(ep: Episode, show_title: str, series_outline: str, n: int) -> tuple[str, str]:
    system = MINING_SYSTEM.replace("{n}", str(n))
    user = (
        f"SHOW: {show_title}\n\n"
        f"SERIES OUTLINE (episode loglines, for your understanding of who is who and what would be a spoiler; "
        f"a new viewer has NOT seen any of this):\n{series_outline}\n\n"
        f"EPISODE {ep.number} (runtime {format_tc(ep.duration)})\n"
        f"Logline: {ep.logline}\nContinuity: {ep.continuity}\n\n"
        f"TRANSCRIPT (segments[].episode must be {ep.number}):\n{render_transcript(ep.events)}\n"
    )
    return system, user


def repair_prompt(ep_numbers: list[int], transcript: str, failures: list[dict]) -> tuple[str, str]:
    system = MINING_SYSTEM.replace("{n}", str(max(1, len(failures))))
    lines = []
    for f in failures:
        lines.append(f"- Candidate \"{f['candidate'].get('title', '?')}\" failed validation:")
        for err in f["errors"]:
            lines.append(f"    * {err}")
        lines.append(f"  Original candidate JSON: {f['candidate_json']}")
    user = (
        "Some of your promo candidates failed automatic validation. Return corrected versions of ONLY these "
        "candidates (same moment and intent, fixed boundaries). If a candidate cannot be fixed without losing what "
        "made it work, leave it out.\n\n" + "\n".join(lines) +
        f"\n\nTRANSCRIPT (episodes {ep_numbers}):\n{transcript}\n"
    )
    return system, user


MONTAGE_SYSTEM = f"""\
You are a senior short-form promo editor for a vertical microdrama app. Earlier analysis found the strongest
single moments of the show (listed below with their transcripts; ids look like E03-L045 = episode 3, event L045).
Propose up to {{n}} MONTAGE promos that join 2-3 segments from DIFFERENT episodes into one 20-60 second clip, the way
trailer-style promos sell a show's premise. Only propose a montage if it would clearly beat the single clips: each
cut must make sense to a stranger with no text overlays, voice-over or music cues, and the montage must still open on
a hook within 3 seconds and end on an unresolved cliffhanger. Returning zero montages is acceptable.

{RUBRIC}
Rules: use only ids shown below, written exactly as shown (E03-L045). segments[].episode is the episode number and
start_id/end_id are the ids. hook_id and cliffhanger_id and evidence_ids also use the E03-L045 form. Never invent
dialogue, events or times.
"""


def montage_prompt(show_title: str, blocks: list[str], n: int) -> tuple[str, str]:
    system = MONTAGE_SYSTEM.replace("{n}", str(n))
    user = f"SHOW: {show_title}\n\nSTRONGEST SINGLE MOMENTS:\n\n" + "\n\n".join(blocks)
    return system, user


COLD_VIEWER_SYSTEM = """\
You are a typical viewer scrolling short-form video. You have NEVER heard of this show. Below is the complete
screenplay of one short clip (camera, action, dialogue and sound), and separately what happens in its first
3 seconds. Use ONLY what is in the clip. Do not guess plot from outside knowledge. Be honest and specific:
if you would scroll past, say so.
"""

COLD_VIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "what_i_understood": {"type": "string"},
        "confusing_points": {"type": "array", "items": {"type": "string"}},
        "first_3_seconds_stop_scrolling": {"type": "integer"},
        "first_3_seconds_reason": {"type": "string"},
        "ending_leaves_open_question": {"type": "boolean"},
        "open_question": {"type": "string"},
        "could_follow_without_context": {"type": "boolean"},
        "would_watch_next_episode": {"type": "integer"},
    },
    "required": ["what_i_understood", "confusing_points", "first_3_seconds_stop_scrolling",
                 "first_3_seconds_reason", "ending_leaves_open_question", "open_question",
                 "could_follow_without_context", "would_watch_next_episode"],
    "additionalProperties": False,
}


def cold_viewer_prompt(clip_lines: list[str], first3_lines: list[str]) -> tuple[str, str]:
    user = (
        "FIRST 3 SECONDS:\n" + ("\n".join(first3_lines) or "(nothing but picture)") +
        "\n\nFULL CLIP:\n" + "\n".join(clip_lines) +
        "\n\nScores are 1-5 integers (5 = strongest). Answer as the viewer."
    )
    return COLD_VIEWER_SYSTEM, user
