"""Dev helper used during the build session: serialises the responder's (Claude's) answers into the
exchange/responses/<task>.json files in the exact schema the tool requested. It only formats JSON;
all timings are computed later by the tool from the event ids."""
import json, sys
from pathlib import Path
KEYS = ["hook", "cliffhanger", "standalone", "emotion_curiosity", "visual", "spoiler_safety"]
RESP = Path("outputs/ai_full/exchange/responses")

def C(title, segs, hook_id, hook, cliff_id, cliff, why, stranger, evidence, scores, reasons, ctx, spoiler,
      spoiler_note, weak, conf, ep=None):
    return {"title": title,
            "segments": [{"episode": s[0], "start_id": s[1], "end_id": s[2]} if len(s) == 3 else
                         {"episode": ep, "start_id": s[0], "end_id": s[1]} for s in segs],
            "hook_id": hook_id, "hook": hook, "cliffhanger_id": cliff_id, "cliffhanger": cliff,
            "why_it_works": why, "stranger_view": stranger, "evidence_ids": evidence,
            "scores": {k: {"score": v, "reason": r} for k, v, r in zip(KEYS, scores, reasons)},
            "context_dependencies": ctx, "spoiler_risk": spoiler, "spoiler_note": spoiler_note,
            "weaknesses": weak, "confidence": conf}

def save(task, note, cands):
    RESP.mkdir(parents=True, exist_ok=True)
    (RESP / f"{task}.json").write_text(json.dumps({"episode_note": note, "candidates": cands}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", task, len(cands))
