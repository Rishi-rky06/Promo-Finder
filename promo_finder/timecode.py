"""Timestamp parsing/formatting. All internal times are float seconds.

Screenplay format observed in the input pack: ``MM:SS.ff`` (e.g. ``02:18.00``),
always relative to the start of the episode. ``H:MM:SS.ff`` is also accepted so
longer episodes don't break the tool.
"""
from __future__ import annotations

import re

_TC = re.compile(r"^\s*(?:(\d+):)?(\d{1,3}):(\d{2}(?:\.\d+)?)\s*$")


class TimecodeError(ValueError):
    pass


def parse_tc(text: str) -> float:
    m = _TC.match(text)
    if not m:
        raise TimecodeError(f"Unrecognised timestamp: {text!r}")
    hours, minutes, seconds = m.groups()
    sec = float(seconds)
    if sec >= 60:
        raise TimecodeError(f"Seconds field >= 60 in timestamp: {text!r}")
    return (int(hours) * 3600 if hours else 0) + int(minutes) * 60 + sec


def format_tc(seconds: float) -> str:
    """Format seconds back into the screenplay's own ``MM:SS.ff`` style."""
    if seconds < 0:
        raise TimecodeError("Negative time")
    total_cs = int(round(seconds * 100))
    minutes, cs = divmod(total_cs, 6000)
    return f"{minutes:02d}:{cs / 100:05.2f}"
