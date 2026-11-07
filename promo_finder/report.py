"""Write run outputs: results.json (machine), summary.md + report.html (editor), CSVs (review/ranking)."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .prompts import SCORE_KEYS

SHORT = {"hook": "Hook", "cliffhanger": "Cliff", "standalone": "Standalone", "emotion_curiosity": "Emotion",
         "visual": "Visual", "spoiler_safety": "SpoilerSafe"}


def cost_estimate(tasks: list[dict], cfg: dict) -> dict:
    """Measured cost if the provider reported usage; otherwise a projection from prompt sizes."""
    prices = cfg.get("pricing", {})
    model = cfg["provider"]["anthropic"]["model"]
    price = prices.get(model)
    measured_in = sum((t.get("usage") or {}).get("input_tokens", 0) or 0 for t in tasks)
    measured_out = sum((t.get("usage") or {}).get("output_tokens", 0) or 0 for t in tasks)
    proj_in = sum(t.get("prompt_chars", 0) for t in tasks) / 4
    proj_out = sum(t.get("response_chars", 0) or 0 for t in tasks) / 4
    out = {"pricing_model": model, "note": "", "measured_tokens": {"input": measured_in, "output_incl_thinking": measured_out}}
    if cfg["provider"]["name"] != "anthropic":
        # Prices in [pricing] are Anthropic list prices; applying them to another provider would be misleading.
        if measured_in and price:
            out["hypothetical_usd_at_listed_price"] = round(
                measured_in / 1e6 * price["input"] + measured_out / 1e6 * price["output"], 2)
            out["note"] = (f"Provider is {cfg['provider']['name']}, not billed here. hypothetical_usd_at_listed_price is what "
                           f"these measured tokens would cost at {model} list prices, for comparison only.")
        return out
    if measured_in and price:
        out["measured_usd"] = round(measured_in / 1e6 * price["input"] + measured_out / 1e6 * price["output"], 3)
    if proj_in and price:
        out["projected_tokens"] = {"input": int(proj_in), "visible_output": int(proj_out)}
        out["projected_usd_excluding_thinking"] = round(proj_in / 1e6 * price["input"] +
                                                        proj_out / 1e6 * price["output"], 3)
        out["note"] = ("Projection = characters/4 at the listed price for the configured Anthropic model. It excludes "
                       "model 'thinking' tokens, which can add several times the visible output; treat as a lower bound.")
    return out


def _segs(c: dict) -> str:
    return " + ".join(f"E{s['episode']} {s['start_tc']}-{s['end_tc']}" for s in c["segments"])


def _scores_inline(c: dict) -> str:
    return " ".join(f"{SHORT[k]} {c['scores'][k]['score']}" for k in SCORE_KEYS)


def _flags(c: dict) -> str:
    return "; ".join(c.get("review_flags") or []) or "-"


def write_all(out: Path, results: dict, ranking: dict, parse_report: dict | None) -> None:
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csvs(out, ranking)
    (out / "summary.md").write_text(summary_md(results, ranking), encoding="utf-8")
    (out / "report.html").write_text(report_html(results, ranking), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps(results["manifest"], ensure_ascii=False, indent=2),
                                           encoding="utf-8")


def _write_csvs(out: Path, ranking: dict) -> None:
    rows = []
    for group in ("shortlist", "alternatives", "duplicates"):
        for c in ranking[group]:
            rows.append({
                "group": group, "shortlist_rank": c.get("shortlist_rank", ""), "overall_rank": c.get("overall_rank", ""),
                "id": c["id"], "origin": c["origin"], "episodes": " ".join(map(str, c["episodes"])),
                "segments": _segs(c), "duration_s": c["metrics"]["total_duration"], "score": c["score"]["total"],
                "base": c["score"]["base"], "penalties": "; ".join(f"-{p['points']:g} {p['code']}" for p in c["score"]["penalties"]),
                **{k: c["scores"][k]["score"] for k in SCORE_KEYS},
                "confidence": c["confidence"], "needs_human_review": c["needs_human_review"], "flags": _flags(c),
                "spoiler_risk": c["spoiler_risk"], "title": c["title"], "hook": c["hook"], "cliffhanger": c["cliffhanger"],
                "why_it_works": c["why_it_works"], "duplicate_of": c.get("duplicate_of") or "",
                "why_lower": " | ".join(c.get("why_lower") or []),
            })
    _csv(out / "ranking.csv", rows)
    review = [{"shortlist_rank": c.get("shortlist_rank", ""), "id": c["id"], "segments": _segs(c),
               "duration_s": c["metrics"]["total_duration"], "tool_score": c["score"]["total"],
               "tool_confidence": c["confidence"], "hook_stops_scroll_Y/N": "", "ending_unresolved_Y/N": "",
               "stranger_can_follow_Y/N": "", "boundaries_ok_Y/N": "", "would_post_Y/N": "", "reviewer": "",
               "notes": ""} for c in ranking["shortlist"] + ranking["alternatives"]]
    _csv(out / "human_review_sheet.csv", review)


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _provenance(m: dict) -> str:
    if m["provider"] == "heuristic":
        return ("**Judgments: HEURISTIC BASELINE (no AI model).** Keyword/structure rules only; every candidate is "
                "low-confidence. Timestamps, durations and checks are exact.")
    if m["provider"] == "file":
        return (f"**Judgments: produced via the file-exchange provider; responder = {m.get('responder')}.** "
                "The tool did not call a model API in this run; it wrote prompts to disk and validated the answers "
                "it read back. Timestamps, durations and checks are computed by code, not by the responder.")
    return f"**Judgments: {m['provider_detail']} via API.**"


def validation_stats(cands: list[dict], cfg: dict) -> dict:
    p = cfg["promo"]
    n = len(cands)
    return {
        "n": n,
        "duration_ok": sum(p["min_seconds"] - 0.05 <= c["metrics"]["total_duration"] <= p["max_seconds"] + 0.05 for c in cands),
        "hook_in_window": sum((c["metrics"].get("hook_offset") or 0) <= p["hook_window_seconds"] + 0.05 for c in cands),
        "cut_on_cliffhanger": sum((c["metrics"].get("cliffhanger_tail") or 0) <= p["cliffhanger_tail_seconds"] + 0.05 for c in cands),
        "continuous": sum(c["metrics"]["segments"] == 1 for c in cands),
        "boundary_adjusted": sum(any(w["code"] == "boundary_adjusted" for w in c["warnings"]) for c in cands),
    }


def summary_md(results: dict, ranking: dict) -> str:
    m = results["manifest"]
    cfg = m["config"]
    L = []
    L.append(f"# Promo Finder results: {m['show_title']}\n")
    L.append(_provenance(m) + "\n")
    L.append("Rankings are editorial recommendations, not predictions of social performance (no engagement data).\n")
    L.append("## Run\n")
    L.append(f"- Status: **{m['status']}**; {m['episodes_found']} episodes found, {m['episodes_mined_ok']} mined OK"
             + (f", pending: {m['episodes_pending']}" if m["episodes_pending"] else "")
             + (f", failed: {m['episodes_failed']}" if m["episodes_failed"] else ""))
    L.append(f"- Skipped input files: {len(m['skipped_files'])}")
    L.append(f"- Valid candidates: {m['counts']['candidates_valid']} (montages: {m['counts']['montages_valid']}); "
             f"rejected by validation: {m['counts']['candidates_rejected']}; near-duplicates removed: {m['counts']['duplicates']}")
    L.append(f"- Montage stage: {m['stages']['montage']}; cold-viewer AI review: {m['stages']['cold_viewer']}")
    if m.get("tasks_pending"):
        L.append(f"- Model tasks still waiting for an answer ({len(m['tasks_pending'])}): {', '.join(m['tasks_pending'])}")
    if m.get("tasks_failed"):
        L.append(f"- Model tasks that FAILED ({len(m['tasks_failed'])}): "
                 + "; ".join(f"{t['task_id']}: {t['error']}" for t in m["tasks_failed"]))
    w = cfg["weights"]
    L.append("- Weights: " + ", ".join(f"{k} {w[k]}" for k in SCORE_KEYS))
    for short in ("shortlist",):
        st = validation_stats(ranking[short], cfg)
        L.append(f"- Automated checks on the {st['n']} shortlisted promos: duration in range {st['duration_ok']}/{st['n']}, "
                 f"hook within {cfg['promo']['hook_window_seconds']:g}s {st['hook_in_window']}/{st['n']}, cut within "
                 f"{cfg['promo']['cliffhanger_tail_seconds']:g}s of the cliffhanger {st['cut_on_cliffhanger']}/{st['n']}, "
                 f"continuous {st['continuous']}/{st['n']}, boundary snapped to finish a line {st['boundary_adjusted']}/{st['n']}")
    L.append("")
    L.append("## Post-first shortlist\n")
    L.append("| # | ID | Segments | Dur | Score | Conf | Sub-scores | Review flags |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in ranking["shortlist"]:
        L.append(f"| {c['shortlist_rank']} | {c['id']} | {_segs(c)} | {c['metrics']['total_duration']:.1f}s | "
                 f"{c['score']['total']:.1f} | {c['confidence']} | {_scores_inline(c)} | {_flags(c)} |")
    L.append("")
    for c in ranking["shortlist"]:
        L += _detail_md(c)
    L.append("## Alternatives (valid, ranked lower)\n")
    L.append("| Overall | ID | Segments | Dur | Score | Conf | Why lower |")
    L.append("|---|---|---|---|---|---|---|")
    for c in ranking["alternatives"]:
        L.append(f"| {c['overall_rank']} | {c['id']} | {_segs(c)} | {c['metrics']['total_duration']:.1f}s | "
                 f"{c['score']['total']:.1f} | {c['confidence']} | {'; '.join(c.get('why_lower', []))} |")
    L.append("")
    if ranking["duplicates"]:
        L.append("## Near-duplicates removed\n")
        for c in ranking["duplicates"]:
            L.append(f"- {c['id']} ({_segs(c)}, score {c['score']['total']:.1f}) duplicates {c['duplicate_of']}: {c['duplicate_reason']}")
        L.append("")
    if results["rejected"]:
        L.append("## Rejected by validation\n")
        for c in results["rejected"]:
            L.append(f"- {c['id']} \"{c.get('title', '')}\": " + "; ".join(c["errors"])
                     + (f" (repair: {c.get('repair_status')})" if c.get("repair_status") else ""))
        L.append("")
    L.append("## Episodes\n")
    L.append("| Ep | Status | Valid | Rejected | Note |")
    L.append("|---|---|---|---|---|")
    for e in results["episodes"]:
        L.append(f"| {e['episode']} | {e['status']} | {', '.join(e['valid']) or '-'} | {', '.join(e['rejected']) or '-'} | "
                 f"{(e['episode_note'] or '').replace('|', '/')} |")
    L.append("")
    return "\n".join(L)


def _detail_md(c: dict) -> list[str]:
    L = [f"### #{c['shortlist_rank']} {c['id']}: {c['title']}\n"]
    for i, s in enumerate(c["segments"], start=1):
        L.append(f"- Segment {i}: Episode {s['episode']} **{s['start_tc']} - {s['end_tc']}** ({s['duration']:.1f}s; "
                 f"{s['start_id']}..{s['end_id']})")
    L.append(f"- Total: **{c['metrics']['total_duration']:.1f}s**; score **{c['score']['total']:.1f}** "
             f"(base {c['score']['base']:.1f}" + "".join(f", -{p['points']:g} {p['code']}" for p in c["score"]["penalties"])
             + f"); confidence **{c['confidence']}** ({'; '.join(c['confidence_reasons'])})")
    L.append(f"- Hook (+{c['metrics'].get('hook_offset', 0):.1f}s): {c['hook']}  \n  Screenplay: `{c['hook_line']}`")
    L.append(f"- Cliffhanger (clip ends {c['metrics'].get('cliffhanger_tail', 0):.1f}s after it): {c['cliffhanger']}  \n"
             f"  Screenplay: `{c['cliffhanger_line']}`")
    L.append(f"- Why it works: {c['why_it_works']}")
    L.append(f"- What a stranger sees: {c['stranger_view']}")
    L.append("- Evidence: " + "; ".join(f"`{e['render']}`" for e in c["evidence"]))
    L.append("- Sub-scores: " + "; ".join(f"{k} {c['scores'][k]['score']} ({c['scores'][k]['reason']})" for k in SCORE_KEYS))
    L.append(f"- Spoiler risk: {c['spoiler_risk']} ({c['spoiler_note']})")
    if c["context_dependencies"]:
        L.append("- Context needed: " + "; ".join(c["context_dependencies"]))
    if c["weaknesses"]:
        L.append("- Weaknesses: " + "; ".join(c["weaknesses"]))
    other = [w["message"] for w in c["warnings"] if w["code"] not in ("late_hook", "cut_after_cliffhanger")]
    if other:
        L.append("- Validation notes: " + "; ".join(other))
    cv = c.get("cold_viewer")
    if cv and cv.get("status") == "ok":
        r = cv["result"]
        L.append(f"- Cold-viewer AI review ({cv['reviewer']}): follows={r['could_follow_without_context']}, "
                 f"first-3s={r['first_3_seconds_stop_scrolling']}/5, open ending={r['ending_leaves_open_question']}, "
                 f"would watch={r['would_watch_next_episode']}/5. Understood: {r['what_i_understood']} "
                 f"Confusing: {'; '.join(r['confusing_points']) or 'nothing'}")
    elif cv:
        L.append(f"- Cold-viewer AI review: {cv['status']}")
    if c.get("review_flags"):
        L.append(f"- **Needs human review:** {', '.join(c['review_flags'])}")
    L.append("")
    return L


def report_html(results: dict, ranking: dict) -> str:
    m = results["manifest"]
    esc = html.escape

    def card(c: dict, label: str) -> str:
        rows = "".join(
            f"<tr><td>{k}</td><td><b>{c['scores'][k]['score']}</b></td><td>{esc(c['scores'][k]['reason'])}</td></tr>"
            for k in SCORE_KEYS)
        pens = "".join(f"<li>-{p['points']:g} {esc(p['code'])}: {esc(p['reason'])}</li>" for p in c["score"]["penalties"])
        hook_ids = {c["hook_line"].split(" ", 1)[0] if c["hook_line"] else None}
        cliff_ids = {c["cliffhanger_line"].split(" ", 1)[0] if c["cliffhanger_line"] else None}
        lines = []
        for t in c["transcript"]:
            cls = "hook" if t["id"] in hook_ids and t["segment"] == 1 else (
                "cliff" if t["id"] in cliff_ids and t["segment"] == len(c["segments"]) else t["kind"])
            lines.append(f"<div class='ln {cls}'><span class='t'>+{t['clip_time']:.1f}s</span> "
                         f"<span class='ep'>E{t['episode']}</span> {esc(t['render'])}</div>")
        cv = c.get("cold_viewer")
        cv_html = ""
        if cv and cv.get("status") == "ok":
            r = cv["result"]
            cv_html = (f"<p><b>Cold-viewer AI review</b> ({esc(cv['reviewer'] or '')}): follows={r['could_follow_without_context']}, "
                       f"first-3s {r['first_3_seconds_stop_scrolling']}/5, open ending={r['ending_leaves_open_question']}, "
                       f"would watch {r['would_watch_next_episode']}/5.<br>{esc(r['what_i_understood'])}<br>"
                       f"Confusing: {esc('; '.join(r['confusing_points']) or 'nothing')}</p>")
        elif cv:
            cv_html = f"<p><b>Cold-viewer AI review:</b> {esc(cv['status'])}</p>"
        segs = "<br>".join(f"Episode {s['episode']}: <b>{s['start_tc']} - {s['end_tc']}</b> ({s['duration']:.1f}s)"
                           for s in c["segments"])
        flags = f"<p class='flag'>Needs human review: {esc(', '.join(c['review_flags']))}</p>" if c["review_flags"] else ""
        why_lower = ("<p><b>Why lower:</b> " + esc("; ".join(c["why_lower"])) + "</p>") if c.get("why_lower") else ""
        return f"""<details class='card'><summary><b>{esc(label)}</b> {esc(c['id'])} | {esc(_segs(c))} |
 {c['metrics']['total_duration']:.1f}s | score <b>{c['score']['total']:.1f}</b> | {esc(c['confidence'])} confidence |
 {esc(c['title'])}</summary>
<div class='grid'><div>
<p>{segs}</p>
<p><b>Hook</b> (+{c['metrics'].get('hook_offset', 0):.1f}s): {esc(c['hook'])}</p>
<p><b>Cliffhanger</b> (ends {c['metrics'].get('cliffhanger_tail', 0):.1f}s after): {esc(c['cliffhanger'])}</p>
<p><b>Why it works:</b> {esc(c['why_it_works'])}</p><p><b>Stranger sees:</b> {esc(c['stranger_view'])}</p>
<p><b>Spoiler:</b> {esc(c['spoiler_risk'])} - {esc(c['spoiler_note'])}</p>
<p><b>Context needed:</b> {esc('; '.join(c['context_dependencies']) or 'none')}</p>
<p><b>Weaknesses:</b> {esc('; '.join(c['weaknesses']) or 'none listed')}</p>
{flags}{why_lower}{cv_html}
<table><tr><th>criterion</th><th>1-5</th><th>reason</th></tr>{rows}</table>
<p>Base {c['score']['base']:.1f}</p><ul>{pens}</ul>
<p><b>Confidence:</b> {esc('; '.join(c['confidence_reasons']))}</p>
</div><div class='tx'>{''.join(lines)}</div></div></details>"""

    parts = [f"<h2>Post-first shortlist</h2>"] + [card(c, f"#{c['shortlist_rank']}") for c in ranking["shortlist"]]
    parts += [f"<h2>Alternatives</h2>"] + [card(c, f"overall {c['overall_rank']}") for c in ranking["alternatives"]]
    if ranking["duplicates"]:
        parts += ["<h2>Near-duplicates</h2>"] + [card(c, f"dup of {c['duplicate_of']}") for c in ranking["duplicates"]]
    rej = "".join(f"<li>{esc(c['id'])} {esc(c.get('title', ''))}: {esc('; '.join(c['errors']))}</li>" for c in results["rejected"])
    prov = esc(_provenance(m).replace("**", ""))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Promo Finder: {esc(m['show_title'])}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:auto;padding:16px;color:#222}}
.card{{border:1px solid #ccc;border-radius:6px;margin:6px 0;padding:6px 10px}} summary{{cursor:pointer}}
.grid{{display:grid;grid-template-columns:1fr 1.2fr;gap:16px}} .tx{{font:12px monospace;max-height:520px;overflow:auto}}
.ln{{padding:1px 4px}} .ln.hook{{background:#fff3b0}} .ln.cliff{{background:#ffd0d0}} .ln.camera{{color:#777}}
.t{{color:#06c}} .ep{{color:#999}} .flag{{color:#b00;font-weight:bold}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:2px 6px;font-size:13px;vertical-align:top}} .prov{{background:#eef;padding:8px}}</style>
</head><body><h1>Promo Finder: {esc(m['show_title'])}</h1><p class='prov'>{prov}</p>
<p>Status {esc(m['status'])}. {m['episodes_found']} episodes; {m['counts']['candidates_valid']} valid candidates,
{m['counts']['candidates_rejected']} rejected, {m['counts']['duplicates']} near-duplicates. Montage: {esc(m['stages']['montage'])}.
Cold-viewer review: {esc(m['stages']['cold_viewer'])}. Yellow = hook line, red = cliffhanger line. Rankings are editorial
recommendations, not performance predictions.</p>
{''.join(parts)}<h2>Rejected by validation</h2><ul>{rej or '<li>none</li>'}</ul></body></html>"""
