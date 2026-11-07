"""Deterministic screenplay parser.

Expected format (as in the Kuku input pack)::

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. ROOFTOP DINER - NIGHT
    Scene start: 00:00.00
    Scene end: 00:39.00
    **CAMERA:**
    Start: (00:00.00) | End: (00:03.00)
    Wide shot of a rain-soaked diner sign ...
    **VICTOR (Low, Menacing):**
    Start: (00:03.20) | End: (00:05.00)
    "You buried the wrong sister."
    --- SCRIPT END ---

Header keys are matched case-insensitively (episode 6 of the pack uses
upper-case scene keys). Nothing is rewritten: event text, speaker labels and
timestamps are kept exactly as written. Problems are recorded as anomalies,
never silently fixed.
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .models import Episode, Event, Scene
from .timecode import TimecodeError, format_tc, parse_tc

_HEADER = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_SCENE_FIELD = re.compile(r"^Scene (heading|start|end)\s*:\s*(.*)$", re.I)
_BLOCK = re.compile(r"^\*\*(.+?):\*\*\s*$")
_TIMING = re.compile(r"^Start:\s*\(([^)]*)\)\s*\|\s*End:\s*\(([^)]*)\)\s*$", re.I)
_PAREN = re.compile(r"\(([^()]*)\)")
_EP_NUM_FILE = re.compile(r"episode[-_ ]?0*(\d+)", re.I)
_EP_NUM_HEADER = re.compile(r"(\d+)")

TOLERANCE = 0.05  # seconds


def _classify(label: str) -> tuple[str, str, str]:
    """Return (kind, speaker, notes) for a block label such as 'MATRIARCH (O.S.) (Firm)'."""
    upper = label.strip().upper()
    if upper == "CAMERA":
        return "camera", "", ""
    if upper == "ACTION":
        return "action", "", ""
    if upper in ("SOUND FX", "SFX", "SOUND", "MUSIC"):
        return "sfx", "", ""
    notes = [n.strip() for n in _PAREN.findall(label) if n.strip()]
    speaker = _PAREN.sub("", label).strip()
    return "dialogue", speaker, "; ".join(notes)


def parse_text(text: str, source_file: str = "<memory>") -> Episode:
    lines = text.splitlines()
    header: dict[str, str] = {}
    scenes: list[Scene] = []
    events: list[Event] = []
    anomalies: list[str] = []

    pending: dict | None = None  # block being accumulated

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        body = " ".join(t.strip() for t in pending["text"] if t.strip())
        if pending.get("start") is None:
            anomalies.append(f"line {pending['line']}: block '{pending['label']}' has no Start/End timing; skipped")
        else:
            kind, speaker, notes = _classify(pending["label"])
            events.append(Event(
                id=f"L{len(events) + 1:03d}", kind=kind, start=pending["start"], end=pending["end"],
                text=body, speaker=speaker, notes=notes, scene=scenes[-1].number if scenes else 0,
            ))
            if not body:
                anomalies.append(f"line {pending['line']}: block '{pending['label']}' has empty text")
        pending = None

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if m := _HEADER.match(stripped):
            header[m.group(1).upper()] = m.group(2).strip()
            continue
        if stripped.startswith("---") and "SCRIPT" in stripped.upper():
            flush()
            continue
        if m := _SCENE.match(stripped):
            flush()
            scenes.append(Scene(number=int(m.group(1)), heading="", start=None, end=None))
            continue
        if m := _SCENE_FIELD.match(stripped):
            if not scenes:
                scenes.append(Scene(number=1, heading="", start=None, end=None))
                anomalies.append(f"line {i}: scene field before any SCENE marker")
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "heading":
                scenes[-1].heading = value
            else:
                try:
                    setattr(scenes[-1], key, parse_tc(value))
                except TimecodeError as exc:
                    anomalies.append(f"line {i}: {exc}")
            continue
        if stripped.lower().startswith("actions and dialogue"):
            continue
        if m := _BLOCK.match(stripped):
            flush()
            pending = {"label": m.group(1), "line": i, "start": None, "end": None, "text": []}
            continue
        if pending is not None and pending["start"] is None and (m := _TIMING.match(stripped)):
            try:
                pending["start"], pending["end"] = parse_tc(m.group(1)), parse_tc(m.group(2))
            except TimecodeError as exc:
                anomalies.append(f"line {i}: {exc}")
                pending["start"] = None
            continue
        if pending is not None:
            pending["text"].append(stripped)
            continue
        anomalies.append(f"line {i}: unrecognised line outside any block: {stripped[:60]!r}")
    flush()

    number = None
    if "EPISODE NUMBER" in header and (m := _EP_NUM_HEADER.search(header["EPISODE NUMBER"])):
        number = int(m.group(1))
    if number is None and (m := _EP_NUM_FILE.search(Path(source_file).name)):
        number = int(m.group(1))
        anomalies.append("episode number taken from file name (header missing)")
    if number is None:
        number = 0
        anomalies.append("could not determine episode number")
    for key in ("EPISODE LOGLINE", "CONTINUITY NOTES"):
        if key not in header:
            anomalies.append(f"missing header '{key}'")

    _check_timing(scenes, events, anomalies)
    return Episode(
        number=number, title=header.get("EPISODE NUMBER", f"Episode {number}"), source_file=source_file,
        logline=header.get("EPISODE LOGLINE", ""), continuity=header.get("CONTINUITY NOTES", ""),
        scenes=scenes, events=events, anomalies=anomalies,
    )


def _check_timing(scenes: list[Scene], events: list[Event], anomalies: list[str]) -> None:
    for e in events:
        if e.end < e.start - TOLERANCE:
            anomalies.append(f"{e.id}: end {format_tc(e.end)} before start {format_tc(e.start)}")
    by_scene = {s.number: s for s in scenes}
    for e in events:
        s = by_scene.get(e.scene)
        if s and s.start is not None and s.end is not None:
            if e.start < s.start - TOLERANCE or e.end > s.end + TOLERANCE:
                anomalies.append(f"{e.id}: [{format_tc(e.start)}-{format_tc(e.end)}] outside scene {s.number} "
                                 f"[{format_tc(s.start)}-{format_tc(s.end)}]")
    for a, b in zip(scenes, scenes[1:]):
        if a.end is not None and b.start is not None and abs(a.end - b.start) > TOLERANCE:
            anomalies.append(f"scene {a.number} ends {format_tc(a.end)} but scene {b.number} starts {format_tc(b.start)}")
    if scenes and events and scenes[-1].end is not None:
        last = max(e.end for e in events)
        if scenes[-1].end - last > 1.0:
            anomalies.append(f"last timed event ends {format_tc(last)} but final scene end is "
                             f"{format_tc(scenes[-1].end)} ({scenes[-1].end - last:.2f}s untimed tail)")
    prev = None
    for e in events:
        if prev is not None and e.start < prev.start - 1.0:
            anomalies.append(f"{e.id}: starts {format_tc(e.start)}, more than 1s before previous event {prev.id}")
        prev = e
    if not events:
        anomalies.append("no timed events found")


def parse_file(path: Path) -> Episode:
    return parse_text(path.read_text(encoding="utf-8", errors="replace"), source_file=path.name)


@dataclass
class Discovery:
    episodes: list[Episode] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)   # files we saw but could not use, with reason
    warnings: list[str] = field(default_factory=list)


def looks_like_screenplay(text: str) -> bool:
    return bool(_TIMING.search(text) or re.search(r"^Start:\s*\(", text, re.M)) and "**" in text


def discover(input_path: str | Path) -> Discovery:
    """Find and parse every screenplay under a file, folder (recursive) or .zip."""
    path = Path(input_path)
    result = Discovery()
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="promo_finder_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        files = sorted(p for p in tmp.rglob("*") if p.is_file())
    elif path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        raise FileNotFoundError(f"Input not found: {path}")

    for f in files:
        if f.suffix.lower() not in (".txt", ".md", ".fountain", ".text"):
            result.skipped.append({"file": f.name, "reason": f"unsupported extension '{f.suffix}'"})
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if not looks_like_screenplay(text):
            result.skipped.append({"file": f.name, "reason": "no timed screenplay blocks found"})
            continue
        result.episodes.append(parse_text(text, source_file=f.name))

    result.episodes.sort(key=lambda e: e.number)
    seen: dict[int, str] = {}
    for ep in result.episodes:
        if ep.number in seen:
            result.warnings.append(f"duplicate episode number {ep.number}: {seen[ep.number]} and {ep.source_file}")
        seen[ep.number] = ep.source_file
    if result.episodes:
        nums = sorted(seen)
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        if missing:
            result.warnings.append(f"episode numbers missing from sequence: {missing}")
    return result
