"""Dev analysis: descriptive comparison of two runs (no ground truth; not an accuracy metric)."""
import json, sys, collections
def load(p): return json.load(open(p, encoding="utf-8"))
ai, he = load(sys.argv[1]), load(sys.argv[2])
def by(r): return {c["id"]: c for c in r["candidates"]}
A, H = by(ai), by(he)
def overlap(a, b):
    sh = sum(max(0, min(x["end"], y["end"]) - max(x["start"], y["start"])) for x in a["segments"] for y in b["segments"] if x["episode"] == y["episode"])
    return sh / min(a["metrics"]["total_duration"], b["metrics"]["total_duration"])
ai_short = [A[i] for i in ai["ranking"]["shortlist"]]; he_short = [H[i] for i in he["ranking"]["shortlist"]]
print("AI shortlist episodes:", [c["episodes"] for c in ai_short])
print("Heuristic shortlist episodes:", [c["episodes"] for c in he_short])
print("Episodes in both shortlists:", sorted({e for c in ai_short for e in c["episodes"]} & {e for c in he_short for e in c["episodes"]}))
print("\nFor each AI shortlisted clip, best-overlapping heuristic candidate anywhere:")
for c in ai_short:
    best = max(H.values(), key=lambda h: overlap(c, h))
    print(f"  {c['id']:7} vs {best['id']:7} overlap {overlap(c, best):.0%}  heuristic overall rank {best.get('overall_rank')}")
print("\nWhere the heuristic shortlist lands in the AI judgment:")
for h in he_short:
    best = max(A.values(), key=lambda a: overlap(h, a)) if A else None
    ov = overlap(h, best) if best else 0
    note = f"{best['id']} (AI rank {best.get('overall_rank')}, overlap {ov:.0%})" if ov > 0.3 else "no AI candidate here"
    ep = h["episodes"][0]
    ai_note = next((e["episode_note"] for e in ai["episodes"] if e["episode"] == ep), "")
    print(f"  {h['id']:7} -> {note} | AI episode note: {ai_note[:90]}")
eps_none = [e["episode"] for e in ai["episodes"] if not e["valid"]]
print("\nAI episodes with zero candidates:", eps_none)
dist = collections.Counter(c["scores"][k]["score"] for c in ai["candidates"] for k in c["scores"])
print("AI sub-score distribution (all criteria):", dict(sorted(dist.items())))
for k in ["hook","cliffhanger","standalone","emotion_curiosity","visual","spoiler_safety"]:
    d = collections.Counter(c["scores"][k]["score"] for c in ai["candidates"]); print(f"  {k:18}", dict(sorted(d.items())))
tot = sorted(c["score"]["total"] for c in ai["candidates"])
print("AI totals min/median/max:", tot[0], tot[len(tot)//2], tot[-1])
print("confidence:", collections.Counter(c["confidence"] for c in ai["candidates"]))
print("needs human review:", sum(c["needs_human_review"] for c in ai["candidates"]), "of", len(ai["candidates"]))
print("segments per clip:", collections.Counter(c["metrics"]["segments"] for c in ai["candidates"]))
