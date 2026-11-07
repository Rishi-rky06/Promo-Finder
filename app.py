"""Promo Finder editor UI.  Run:  streamlit run app.py

Open a finished run (demo_runs/ or outputs/<run>/results.json) or start a new run on a folder/zip of screenplays.
Weights re-rank live without calling any model.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.config import load_config, load_dotenv
from promo_finder.pipeline import RunAborted, run
from promo_finder.prompts import SCORE_KEYS
from promo_finder.providers import ProviderError
from promo_finder.scoring import rank

ROOT = Path(__file__).resolve().parent
load_dotenv()
st.set_page_config(page_title="Promo Finder", layout="wide")


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segs(c: dict) -> str:
    return " + ".join(f"E{s['episode']} {s['start_tc']}-{s['end_tc']}" for s in c["segments"])


# ---------------- sidebar: choose / create a run -------------------------------------------------
import os
import tempfile

st.sidebar.header("Run")
runs = sorted(list((ROOT / "demo_runs").glob("*/results.json")) + list((ROOT / "outputs").glob("*/results.json")))
mode = st.sidebar.radio("Source", ["Open a finished run", "Process new screenplays"], index=0 if runs else 1)
results = None
if mode == "Open a finished run":
    if not runs:
        st.sidebar.info("No runs found in demo_runs/ or outputs/.")
    else:
        choice = st.sidebar.selectbox("Run", runs, format_func=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
        results = load_results(choice)
else:
    st.sidebar.warning("Do not upload confidential scripts to a public deployment. Uploads are processed on the server.")
    ups = st.sidebar.file_uploader("Upload screenplay .txt files or one .zip", type=["txt", "zip"], accept_multiple_files=True)
    inp = st.sidebar.text_input("...or a local folder/file/.zip path", value=str(ROOT / "samples" / "synthetic_show"))
    # Only offer providers whose credentials exist, so a public deployment never exposes a key.
    providers = {"heuristic (no model, no key)": ("heuristic", None)}
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic (Claude)"] = ("anthropic", None)
    if os.environ.get("NVIDIA_API_KEY"):
        providers["nvidia (config/nvidia.toml)"] = ("openai_compatible", ROOT / "config" / "nvidia.toml")
    label = st.sidebar.selectbox("Provider", list(providers))
    out_name = st.sidebar.text_input("Output folder name", value="new_run")
    if st.sidebar.button("Run", type="primary"):
        pname, cfg_path = providers[label]
        try:
            src = inp
            if ups:
                tmp = Path(tempfile.mkdtemp(prefix="pf_upload_"))
                for f in ups:
                    (tmp / f.name).write_bytes(f.getvalue())
                src = str(tmp / ups[0].name) if len(ups) == 1 and ups[0].name.endswith(".zip") else str(tmp)
            with st.spinner("Processing..."):
                results = run(src, ROOT / "outputs" / out_name, load_config(cfg_path, {"provider": {"name": pname}}))
            st.sidebar.success(f"Done: outputs/{out_name}")
        except (RunAborted, ProviderError, FileNotFoundError) as exc:
            st.sidebar.error(str(exc))

if results is None:
    st.title("Promo Finder")
    st.write("Pick a finished run or process screenplays from the sidebar.")
    st.stop()

m = results["manifest"]
cfg = copy.deepcopy(m["config"])

st.sidebar.header("Weights (live re-rank)")
for k in SCORE_KEYS:
    cfg["weights"][k] = st.sidebar.slider(k, 0.0, 1.0, float(cfg["weights"][k]), 0.01)
cfg["ranking"]["max_per_episode_in_shortlist"] = st.sidebar.number_input(
    "Max per episode in shortlist", 1, 5, int(cfg["ranking"]["max_per_episode_in_shortlist"]))
cfg["penalties"]["per_extra_segment"] = st.sidebar.number_input(
    "Penalty per extra segment", 0.0, 20.0, float(cfg["penalties"]["per_extra_segment"]))
if sum(cfg["weights"].values()) == 0:
    st.sidebar.error("At least one weight must be > 0")
    st.stop()

ranking = rank(copy.deepcopy(results["candidates"]), cfg)
by_id = {c["id"]: c for c in ranking["ranked"] + ranking["duplicates"]}

# ---------------- header -------------------------------------------------------------------------
st.title(f"Promo Finder: {m['show_title']}")
if m["provider"] == "heuristic":
    st.warning("Judgments come from the HEURISTIC BASELINE (keyword rules, no AI model). Treat all as low confidence.")
elif m["provider"] == "file":
    st.info(f"Judgments answered via the file-exchange provider by: {m.get('responder')}. Timestamps and checks are computed by code.")
else:
    st.info(f"Judgments by {m['provider_detail']} via API.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episodes", f"{m['episodes_mined_ok']}/{m['episodes_found']}")
c2.metric("Valid candidates", m["counts"]["candidates_valid"])
c3.metric("Rejected", m["counts"]["candidates_rejected"])
c4.metric("Near-duplicates", len(ranking["duplicates"]))
c5.metric("Pending model tasks", len(m.get("tasks_pending", [])))
st.caption("Rankings are editorial recommendations, not predictions of social performance (no engagement data).")


def table(cands: list[dict], rank_key: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "rank": c.get(rank_key, ""), "id": c["id"], "segments": segs(c), "sec": c["metrics"]["total_duration"],
        "score": c["score"]["total"], "confidence": c["confidence"],
        **{k[:10]: c["scores"][k]["score"] for k in SCORE_KEYS},
        "review flags": ", ".join(c["review_flags"]) or "", "title": c["title"],
    } for c in cands])


tabs = st.tabs(["Shortlist", "All ranked", "Inspect", "Compare", "Needs human review", "Run log"])
with tabs[0]:
    st.subheader("Post these first")
    st.dataframe(table(ranking["shortlist"], "shortlist_rank"), hide_index=True, width="stretch")
with tabs[1]:
    st.dataframe(table(ranking["ranked"], "overall_rank"), hide_index=True, width="stretch")
    if ranking["duplicates"]:
        st.markdown("**Near-duplicates removed**")
        st.dataframe(pd.DataFrame([{"id": c["id"], "duplicate of": c["duplicate_of"], "why": c["duplicate_reason"],
                                    "score": c["score"]["total"]} for c in ranking["duplicates"]]), hide_index=True)

with tabs[2]:
    ids = [c["id"] for c in ranking["ranked"] + ranking["duplicates"]]
    cid = st.selectbox("Candidate", ids)
    c = by_id[cid]
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown(f"### {c['id']}: {c['title']}")
        st.markdown(f"**Score {c['score']['total']:.1f}** (base {c['score']['base']:.1f}) | **{c['confidence']}** confidence"
                    + (f" | overall rank {c.get('overall_rank')}" if c.get("overall_rank") else ""))
        st.table(pd.DataFrame([{"segment": i + 1, "episode": s["episode"], "start": s["start_tc"], "end": s["end_tc"],
                                "seconds": s["duration"], "events": f"{s['start_id']}..{s['end_id']}"}
                               for i, s in enumerate(c["segments"])]))
        st.markdown(f"**Hook** (+{c['metrics'].get('hook_offset', 0):.1f}s): {c['hook']}")
        st.code(c["hook_line"] or "", language=None)
        st.markdown(f"**Cliffhanger** (clip ends {c['metrics'].get('cliffhanger_tail', 0):.1f}s after): {c['cliffhanger']}")
        st.code(c["cliffhanger_line"] or "", language=None)
        st.markdown(f"**Why it works:** {c['why_it_works']}")
        st.markdown(f"**What a stranger sees:** {c['stranger_view']}")
        st.markdown(f"**Spoiler risk:** {c['spoiler_risk']} {('- ' + c['spoiler_note']) if c['spoiler_note'] else ''}")
        if c["context_dependencies"]:
            st.markdown("**Context needed:** " + "; ".join(c["context_dependencies"]))
        if c["weaknesses"]:
            st.markdown("**Weaknesses:** " + "; ".join(c["weaknesses"]))
        if c["review_flags"]:
            st.error("Needs human review: " + ", ".join(c["review_flags"]))
        if c.get("why_lower"):
            st.markdown("**Why it ranks lower:** " + "; ".join(c["why_lower"]))
        st.table(pd.DataFrame([{"criterion": k, "1-5": c["scores"][k]["score"], "weight": cfg["weights"][k],
                                "reason": c["scores"][k]["reason"]} for k in SCORE_KEYS]))
        for p in c["score"]["penalties"]:
            st.markdown(f"- **-{p['points']:g}** {p['code']}: {p['reason']}")
        st.caption("Confidence: " + "; ".join(c["confidence_reasons"]))
        cv = c.get("cold_viewer")
        if cv:
            st.markdown(f"**Cold-viewer AI review:** {cv['status']}")
            if cv.get("result"):
                st.json(cv["result"])
    with right:
        st.markdown("**Clip transcript** (yellow = hook line, red = cliffhanger line)")
        hook_id = (c["hook_line"] or "").split(" ", 1)[0]
        cliff_id = (c["cliffhanger_line"] or "").split(" ", 1)[0]
        lines = []
        for t in c["transcript"]:
            bg = "#fff3b0" if (t["id"] == hook_id and t["segment"] == 1) else (
                "#ffd0d0" if (t["id"] == cliff_id and t["segment"] == len(c["segments"])) else "transparent")
            color = "#888" if t["kind"] == "camera" else "inherit"
            text = t["render"].replace("<", "&lt;")
            lines.append(f"<div style='background:{bg};color:{color};font:12px monospace;padding:1px 4px'>"
                         f"+{t['clip_time']:.1f}s E{t['episode']} {text}</div>")
        st.markdown("<div style='max-height:700px;overflow:auto'>" + "".join(lines) + "</div>", unsafe_allow_html=True)

with tabs[3]:
    pick = st.multiselect("Compare 2-4 candidates", [c["id"] for c in ranking["ranked"]],
                          default=[c["id"] for c in ranking["shortlist"][:3]])
    if pick:
        rows = {"segments": [segs(by_id[i]) for i in pick], "seconds": [by_id[i]["metrics"]["total_duration"] for i in pick],
                "total score": [by_id[i]["score"]["total"] for i in pick], "confidence": [by_id[i]["confidence"] for i in pick]}
        for k in SCORE_KEYS:
            rows[k] = [by_id[i]["scores"][k]["score"] for i in pick]
        rows["penalties"] = ["; ".join(f"-{p['points']:g} {p['code']}" for p in by_id[i]["score"]["penalties"]) or "-" for i in pick]
        rows["spoiler risk"] = [by_id[i]["spoiler_risk"] for i in pick]
        rows["hook"] = [by_id[i]["hook"] for i in pick]
        rows["cliffhanger"] = [by_id[i]["cliffhanger"] for i in pick]
        st.dataframe(pd.DataFrame(rows, index=pick).T.astype(str), width="stretch")

with tabs[4]:
    flagged = [c for c in ranking["ranked"] if c["needs_human_review"]]
    st.write(f"{len(flagged)} of {len(ranking['ranked'])} ranked candidates are flagged for human review.")
    st.dataframe(table(flagged, "overall_rank"), hide_index=True, width="stretch")

with tabs[5]:
    st.json({k: m[k] for k in ("status", "provider", "provider_detail", "responder", "stages", "counts",
                               "episodes_pending", "episodes_failed", "skipped_files", "tasks_pending",
                               "tasks_failed", "cost_estimate") if k in m})
    if results["rejected"]:
        st.markdown("**Rejected by validation**")
        st.dataframe(pd.DataFrame([{"id": c["id"], "title": c.get("title"), "errors": "; ".join(c["errors"]),
                                    "repair": c.get("repair_status", "")} for c in results["rejected"]]), hide_index=True)
    st.dataframe(pd.DataFrame(m["tasks"])[["task_id", "stage", "status", "error"]], hide_index=True, width="stretch")
