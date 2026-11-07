"""Dev helper: print an episode transcript compactly (camera text abbreviated) for reading."""
import sys
from promo_finder.parser import discover
from promo_finder.timecode import format_tc
d = {e.number: e for e in discover(sys.argv[1]).episodes}
for n in map(int, sys.argv[2:]):
    ep = d[n]
    print(f"=== EP {n} ({format_tc(ep.duration)}) {ep.logline}")
    for e in ep.events:
        t = f"{e.id} {format_tc(e.start)}-{format_tc(e.end)}"
        if e.kind == "camera":
            print(f"{t} cam: {e.text[:70]}")
        elif e.kind == "dialogue":
            print(f"{t} {e.speaker} ({e.notes}): {e.text}")
        else:
            print(f"{t} {e.kind}: {e.text}")
