"""Unit tests for parsing, timestamps, validation, scoring and dedupe. Uses only the synthetic sample."""
from pathlib import Path

import pytest

from promo_finder.config import load_config
from promo_finder.parser import discover, parse_text
from promo_finder.scoring import near_duplicate, rank, score_candidate
from promo_finder.timecode import TimecodeError, format_tc, parse_tc
from promo_finder.validate import parse_ref, resolve

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "synthetic_show"


@pytest.fixture(scope="module")
def eps():
    d = discover(SAMPLE)
    return {e.number: e for e in d.episodes}


@pytest.fixture
def cfg():
    return load_config()


def good_scores(v=4):
    keys = ["hook", "cliffhanger", "standalone", "emotion_curiosity", "visual", "spoiler_safety"]
    return {k: {"score": v, "reason": "r"} for k in keys}


def cand(start, end, hook, cliff, ep=1, **kw):
    c = {"title": "t", "segments": [{"episode": ep, "start_id": start, "end_id": end}], "hook_id": hook,
         "hook": "h", "cliffhanger_id": cliff, "cliffhanger": "c", "why_it_works": "w", "stranger_view": "s",
         "evidence_ids": [hook, cliff], "scores": good_scores(), "context_dependencies": [], "spoiler_risk": "low",
         "spoiler_note": "", "weaknesses": [], "confidence": "high"}
    c.update(kw)
    return c


# ---- timecodes -------------------------------------------------------------------------------
def test_timecode_roundtrip():
    assert parse_tc("02:18.00") == pytest.approx(138.0)
    assert parse_tc("00:04.80") == pytest.approx(4.8)
    assert parse_tc("1:02:03.50") == pytest.approx(3723.5)
    assert format_tc(138.0) == "02:18.00"
    assert format_tc(parse_tc("01:27.21")) == "01:27.21"


@pytest.mark.parametrize("bad", ["", "abc", "00:61.00", "12.5"])
def test_timecode_rejects_garbage(bad):
    with pytest.raises(TimecodeError):
        parse_tc(bad)


# ---- parser ----------------------------------------------------------------------------------
def test_discovery_reports_skipped_files(eps):
    d = discover(SAMPLE)
    assert sorted(eps) == [1, 2]
    assert any(s["file"] == "README.txt" for s in d.skipped), "non-screenplay files must be reported, not ignored"


def test_parser_keeps_text_and_timing_verbatim(eps):
    e = eps[1].events[2]
    assert e.kind == "dialogue" and e.speaker == "VICTOR" and e.notes == "Low, Menacing"
    assert e.text == '"You buried the wrong sister."'
    assert (e.start, e.end) == (pytest.approx(3.2), pytest.approx(5.0))
    kinds = {x.kind for x in eps[1].events}
    assert kinds == {"camera", "dialogue", "action", "sfx"}


def test_parser_case_insensitive_headers_and_anomalies():
    text = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: x
--- SCRIPT START ---
SCENE 01
SCENE HEADING: INT. ROOM
SCENE START: 00:00.00
SCENE END: 00:10.00
**ACTION:**
Start: (00:02.00) | End: (00:01.00)
Backwards timing.
**BOB (O.S.) (Angry):**
Start: (00:03.00) | End: (00:12.00)
"Too long."
--- SCRIPT END ---"""
    ep = parse_text(text, "x.txt")
    assert ep.number == 7 and ep.scenes[0].heading == "INT. ROOM" and ep.scenes[0].end == 10.0
    assert ep.events[1].speaker == "BOB" and ep.events[1].notes == "O.S.; Angry"
    joined = " ".join(ep.anomalies)
    assert "before start" in joined and "outside scene" in joined and "CONTINUITY NOTES" in joined


def test_parse_ref():
    assert parse_ref("L5", 3) == (3, "L005")
    assert parse_ref("E07-L045", None) == (7, "L045")
    assert parse_ref("e7:l45", 1) == (7, "L045")
    assert parse_ref("nonsense", 1) == (None, None)


# ---- validation ------------------------------------------------------------------------------
def test_valid_candidate_exact_times(eps, cfg):
    c = resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="x", origin="model", default_episode=1)
    assert c["valid"], c["errors"]
    assert c["segments"][0]["start_tc"] == "00:03.00" and c["segments"][0]["end_tc"] == "00:38.00"
    assert c["metrics"]["total_duration"] == pytest.approx(35.0)
    assert c["metrics"]["hook_offset"] == pytest.approx(0.2)
    assert c["hook_line"].startswith("L003") and "wrong sister" in c["hook_line"]


def test_duration_out_of_range_is_an_error(eps, cfg):
    c = resolve(cand("L002", "L008", "L003", "L008"), eps, cfg, cid="x", origin="model", default_episode=1)
    assert not c["valid"] and any("outside 20-60s" in e for e in c["errors"])


def test_unknown_ids_and_wrong_episode(eps, cfg):
    c = resolve(cand("L002", "L999", "L003", "L999"), eps, cfg, cid="x", origin="model", default_episode=1)
    assert not c["valid"] and any("unknown event id" in e for e in c["errors"])
    c = resolve(cand("L002", "L012", "L003", "L012", ep=2), eps, cfg, cid="x", origin="model", default_episode=1)
    assert not c["valid"] and any("refers to episode 2" in e for e in c["errors"])


def test_late_hook_and_cut_after_cliffhanger_are_measured(eps, cfg):
    # Start on the establishing shot (hook line at +3.2s) and keep running ~10s past the cliffhanger line.
    c = resolve(cand("L001", "L017", "L003", "L012"), eps, cfg, cid="x", origin="model", default_episode=1)
    codes = {w["code"] for w in c["warnings"]}
    assert c["valid"] and "late_hook" in codes and "cut_after_cliffhanger" in codes
    assert c["metrics"]["hook_offset"] == pytest.approx(3.2)


def test_hook_must_be_in_first_segment(eps, cfg):
    c = cand("L002", "L010", "L014", "L017")
    c["segments"].append({"episode": 1, "start_id": "L013", "end_id": "L017"})
    r = resolve(c, eps, cfg, cid="x", origin="model", default_episode=1)
    assert not r["valid"] and any("first segment" in e for e in r["errors"])


def test_boundary_snaps_to_finish_dialogue(eps, cfg):
    # Nothing is cut mid-line here, so no adjustment.
    c = resolve(cand("L002", "L015", "L003", "L014"), eps, cfg, cid="x", origin="model", default_episode=1)
    assert not any(w["code"] == "boundary_adjusted" for w in c["warnings"])
    # Stretch VICTOR's line L006 to 5.5-10.5 so a clip starting at L007 (10.0) would cut it mid-sentence.
    text = (SAMPLE / "episode-001-sample.txt").read_text(encoding="utf-8")
    ep = parse_text(text.replace("Start: (00:07.00) | End: (00:09.80)", "Start: (00:05.50) | End: (00:10.50)"), "e1")
    c = resolve(cand("L007", "L017", "L007", "L017"), {1: ep}, cfg, cid="x", origin="model", default_episode=1)
    assert any(w["code"] == "boundary_adjusted" for w in c["warnings"])
    # 5.5 would now cut MAYA's line L004 (5.4-6.6) mid-sentence, so the snap cascades to 5.4
    assert c["segments"][0]["start"] == pytest.approx(5.4)


def test_evidence_is_quoted_from_screenplay(eps, cfg):
    c = resolve(cand("L002", "L017", "L003", "L017", evidence_ids=["L006", "L999"]), eps, cfg, cid="x",
                origin="model", default_episode=1)
    assert c["evidence"][0]["render"] == eps[1].event("L006").render()
    assert any(w["code"] == "bad_evidence" for w in c["warnings"])


def test_overlapping_segments_rejected(eps, cfg):
    c = cand("L002", "L010", "L003", "L017")
    c["segments"].append({"episode": 1, "start_id": "L008", "end_id": "L017"})
    r = resolve(c, eps, cfg, cid="x", origin="model", default_episode=1)
    assert not r["valid"] and any("overlap" in e for e in r["errors"])


def test_montage_across_episodes(eps, cfg):
    c = cand("E01-L002", "E01-L010", "E01-L003", "E02-L012", ep=1)
    c["segments"].append({"episode": 2, "start_id": "E02-L005", "end_id": "E02-L012"})
    r = resolve(c, eps, cfg, cid="M1", origin="montage", default_episode=None)
    assert r["valid"], r["errors"]
    assert r["episodes"] == [1, 2] and r["metrics"]["segments"] == 2
    assert r["transcript"][-1]["episode"] == 2


# ---- scoring ---------------------------------------------------------------------------------
def test_weighted_score_and_penalties(eps, cfg):
    c = resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="x", origin="model", default_episode=1)
    s = score_candidate(c, cfg)["score"]
    assert s["base"] == pytest.approx(75.0) and s["total"] == pytest.approx(75.0)  # all 4/5 -> 75
    late = resolve(cand("L001", "L017", "L003", "L012"), eps, cfg, cid="y", origin="model", default_episode=1)
    s2 = score_candidate(late, cfg)["score"]
    pen = cfg["penalties"]
    # opens on 3.2s of establishing shot: late hook AND dead air; plus 13.5s after the cliffhanger line
    assert s2["total"] == pytest.approx(75.0 - pen["late_hook"] - pen["dead_air_open"] - pen["cut_after_cliffhanger"])
    assert late["confidence"] == "low"  # high, minus three measured problems


def test_weights_change_ranking(eps, cfg):
    a = cand("L002", "L017", "L003", "L017")
    a["scores"]["hook"]["score"], a["scores"]["visual"]["score"] = 5, 1
    b = cand("L002", "L017", "L003", "L017")
    b["scores"]["hook"]["score"], b["scores"]["visual"]["score"] = 1, 5
    ra = lambda w: score_candidate(resolve(a, eps, w, cid="a", origin="model", default_episode=1), w)["score"]["total"]
    rb = lambda w: score_candidate(resolve(b, eps, w, cid="b", origin="model", default_episode=1), w)["score"]["total"]
    assert ra(cfg) > rb(cfg)
    visual_heavy = load_config(overrides={"weights": {"hook": 0.05, "visual": 0.6}})
    assert rb(visual_heavy) > ra(visual_heavy)


def test_dedupe_and_variety_cap(eps, cfg):
    a = resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="a", origin="model", default_episode=1)
    b = resolve(cand("L003", "L016", "L003", "L016"), eps, cfg, cid="b", origin="model", default_episode=1)
    assert near_duplicate(a, b, cfg)
    r = rank([a, b], cfg)
    assert [c["id"] for c in r["ranked"]] == ["a"] and r["duplicates"][0]["duplicate_of"] == "a"


def test_heuristic_candidates_always_low_confidence(eps, cfg):
    c = resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="x", origin="heuristic", default_episode=1)
    assert score_candidate(c, cfg)["confidence"] == "low"


def test_optional_episode_depth_penalty(eps, cfg):
    on = load_config(overrides={"penalties": {"episode_depth_per_10": 10.0}})
    c1 = score_candidate(resolve(cand("L002", "L017", "L003", "L017"), eps, on, cid="a", origin="model", default_episode=1), on)
    c2 = score_candidate(resolve(cand("L005", "L013", "L005", "L012", ep=2), eps, on, cid="b", origin="model", default_episode=2), on)
    assert not any(p["code"] == "episode_depth" for p in c1["score"]["penalties"])
    assert any(p["code"] == "episode_depth" and p["points"] == 1.0 for p in c2["score"]["penalties"])
    off = score_candidate(resolve(cand("L005", "L013", "L005", "L012", ep=2), eps, cfg, cid="b", origin="model", default_episode=2), cfg)
    assert not any(p["code"] == "episode_depth" for p in off["score"]["penalties"])


def test_cold_viewer_score_breaks_ties(eps, cfg):
    a = score_candidate(resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="a", origin="model", default_episode=1), cfg)
    b = score_candidate(resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="b", origin="model", default_episode=1), cfg)
    a["cold_viewer"] = {"status": "ok", "result": {"first_3_seconds_stop_scrolling": 2, "would_watch_next_episode": 2,
                                                   "could_follow_without_context": True, "ending_leaves_open_question": True,
                                                   "confusing_points": []}}
    b["cold_viewer"] = {"status": "ok", "result": {"first_3_seconds_stop_scrolling": 5, "would_watch_next_episode": 5,
                                                   "could_follow_without_context": True, "ending_leaves_open_question": True,
                                                   "confusing_points": []}}
    from promo_finder.scoring import sort_key
    assert sorted([a, b], key=sort_key)[0]["id"] == "b"


def test_camera_only_hook_is_flagged_and_penalised(eps, cfg):
    cam = resolve(cand("L002", "L017", "L002", "L017"), eps, cfg, cid="c", origin="model", default_episode=1)  # L002 = camera
    line = resolve(cand("L002", "L017", "L003", "L017"), eps, cfg, cid="d", origin="model", default_episode=1)  # L003 = dialogue
    assert any(w["code"] == "camera_only_hook" for w in cam["warnings"])
    assert not any(w["code"] == "camera_only_hook" for w in line["warnings"])
    assert score_candidate(cam, cfg)["score"]["total"] == score_candidate(line, cfg)["score"]["total"] - cfg["penalties"]["camera_only_hook"]
