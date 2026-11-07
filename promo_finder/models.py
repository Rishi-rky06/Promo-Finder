"""Plain data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .timecode import format_tc


@dataclass
class Event:
    """One timed line of the screenplay (camera shot, action, dialogue or sound cue)."""

    id: str               # stable per-episode id, e.g. "L012" (file order)
    kind: str             # camera | action | dialogue | sfx
    start: float
    end: float
    text: str
    speaker: str = ""      # dialogue only, label exactly as written in the screenplay
    notes: str = ""        # parenthetical delivery notes, e.g. "Cold, Commanding"
    scene: int = 0         # 1-based scene number

    def label(self) -> str:
        if self.kind == "dialogue":
            return f"{self.speaker} ({self.notes})" if self.notes else self.speaker
        return {"camera": "CAMERA", "action": "ACTION", "sfx": "SOUND FX"}.get(self.kind, self.kind.upper())

    def render(self) -> str:
        """Compact one-line form used in prompts and reports."""
        kind = {"camera": "CAM", "action": "ACT", "dialogue": "DIA", "sfx": "SFX"}[self.kind]
        body = f'{self.label()}: {self.text}' if self.kind == "dialogue" else self.text
        return f"{self.id} [{format_tc(self.start)}-{format_tc(self.end)}] {kind} {body}"


@dataclass
class Scene:
    number: int
    heading: str
    start: Optional[float]
    end: Optional[float]


@dataclass
class Episode:
    number: int
    title: str
    source_file: str
    logline: str
    continuity: str
    scenes: list[Scene]
    events: list[Event]
    anomalies: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [e.end for e in self.events] + [s.end for s in self.scenes if s.end is not None]
        return max(ends) if ends else 0.0

    def event(self, event_id: str) -> Optional[Event]:
        return self._index().get(event_id.strip().upper())

    def _index(self) -> dict[str, Event]:
        if not hasattr(self, "_idx"):
            self._idx = {e.id: e for e in self.events}  # type: ignore[attr-defined]
        return self._idx  # type: ignore[attr-defined]

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {
            "episode": self.number,
            "source_file": self.source_file,
            "scenes": len(self.scenes),
            "events": len(self.events),
            "events_by_kind": kinds,
            "duration_seconds": round(self.duration, 2),
            "duration_tc": format_tc(self.duration),
            "anomalies": self.anomalies,
        }


@dataclass
class Segment:
    episode: int
    start_id: str
    end_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_tc"] = format_tc(self.start)
        d["end_tc"] = format_tc(self.end)
        d["duration"] = round(self.duration, 2)
        return d
