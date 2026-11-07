"""End-to-end pipeline: discover -> parse -> mine -> validate/repair -> montage -> rank -> cold-viewer review -> report."""
from __future__ import annotations

import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import heuristic, prompts, report
from .models import Episode
from .parser import discover
from .providers import PendingResponse, Provider, ProviderError, build_provider
from .scoring import rank
from .validate import resolve

VERSION = "1.0.0"


class RunAborted(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Runner:
    def __init__(self, cfg: dict, out_dir: Path, provider: Provider | None, responder_label: str):
        self.cfg = cfg
        self.out = out_dir
        self.provider = provider
        self.responder_label = responder_label
        self.tasks: list[dict] = []
        self.raw_dir = out_dir / "raw_model_io"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # -- model calls -------------------------------------------------------------------------------
    def call(self, task_id: str, stage: str, system: str, user: str, schema: dict) -> tuple[dict | None, str]:
        """One model call with full bookkeeping. Returns (parsed JSON or None, status)."""
        assert self.provider is not None
        rec = {"task_id": task_id, "stage": stage, "status": None, "error": None, "usage": {}, "cached": False,
               "elapsed_s": 0.0, "prompt_chars": len(system) + len(user)}
        raw = {"task_id": task_id, "stage": stage, "request": {"system": system, "user": user, "schema": schema}}
        try:
            res = self.provider.generate_json(task_id, system, user, schema)
            rec.update(status="ok", usage=res.usage, cached=res.cached, elapsed_s=round(res.elapsed_s, 2),
                       response_chars=len(res.raw_text), model=res.model, provider=res.provider)
            raw.update(provider=res.provider, model=res.model, usage=res.usage, cached=res.cached,
                       response_text=res.raw_text)
            data = res.data
        except PendingResponse:
            rec["status"] = "pending"
            data = None
        except ProviderError as exc:
            rec.update(status="error", error=str(exc))
            raw["error"] = str(exc)
            data = None
            if exc.fatal:
                self.tasks.append(rec)
                raise RunAborted(str(exc))
        self.tasks.append(rec)
        if rec["status"] != "pending":
            (self.raw_dir / f"{task_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
        _log(f"  [{rec['status']:7}] {task_id}" + (f"  ({rec['error']})" if rec["error"] else "")
             + (" (cache)" if rec["cached"] else ""))
        return data, rec["status"]

    # -- per-episode mining with one validation-repair loop ---------------------------------------
    def mine(self, ep: Episode, episodes: dict[int, Episode], show_title: str, outline: str) -> dict:
        n = self.cfg["promo"]["candidates_per_episode"]
        if self.provider is None:
            data = heuristic.mine_episode(ep, self.cfg, series_len=len(episodes), n=n)
            self.tasks.append({"task_id": f"mine_E{ep.number:02d}", "stage": "mine", "status": "ok",
                               "provider": "heuristic", "error": None})
            (self.raw_dir / f"mine_E{ep.number:02d}.json").write_text(
                json.dumps({"task_id": f"mine_E{ep.number:02d}", "provider": "heuristic", "parsed": data},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            origin = "heuristic"
        else:
            system, user = prompts.mining_prompt(ep, show_title, outline, n)
            data, status = self.call(f"mine_E{ep.number:02d}", "mine", system, user, prompts.MINING_SCHEMA)
            origin = "model"
        out = {"episode": ep.number, "status": "ok" if data is not None else status,
               "episode_note": (data or {}).get("episode_note", ""), "candidates": [], "rejected": []}
        if data is None:
            return out
        resolved = [resolve(c, episodes, self.cfg, cid=f"E{ep.number:02d}-C{i}", origin=origin,
                            default_episode=ep.number) for i, c in enumerate(data.get("candidates", []), start=1)]
        resolved = self.repair(resolved, [ep.number], prompts.render_transcript(ep.events), episodes,
                               f"repair_E{ep.number:02d}", default_episode=ep.number, origin=origin)
        out["candidates"] = [c for c in resolved if c["valid"]]
        out["rejected"] = [c for c in resolved if not c["valid"]]
        return out

    def repair(self, resolved: list[dict], ep_numbers: list[int], transcript: str, episodes, task_prefix: str,
               default_episode, origin) -> list[dict]:
        attempts = self.cfg["promo"]["repair_attempts"] if self.provider is not None else 0
        for attempt in range(1, attempts + 1):
            bad = [c for c in resolved if not c["valid"] and c.get("repair_status") is None]
            if not bad:
                break
            failures = [{"candidate": c["raw_candidate"], "errors": c["errors"],
                         "candidate_json": json.dumps(c["raw_candidate"], ensure_ascii=False)} for c in bad]
            system, user = prompts.repair_prompt(ep_numbers, transcript, failures)
            data, status = self.call(f"{task_prefix}_{attempt}", "repair", system, user, prompts.MINING_SCHEMA)
            if data is None:
                for c in bad:
                    c["repair_status"] = f"repair {status}"
                break
            fixed = data.get("candidates", [])
            for c in bad:
                c["repair_status"] = "not returned by model"
            for j, f in enumerate(fixed):
                if j >= len(bad):
                    break
                orig = bad[j]
                new = resolve(f, episodes, self.cfg, cid=orig["id"] + "r", origin=origin,
                              default_episode=default_episode)
                new["repaired"] = True
                new["repair_of"] = orig["errors"]
                orig["repair_status"] = "replaced" if new["valid"] else "repair also failed"
                resolved.append(new)
        return [c for c in resolved if c["valid"] or c.get("repair_status") != "replaced"]

    # -- cross-episode montage --------------------------------------------------------------------
    def montage(self, singles: list[dict], episodes, show_title: str) -> tuple[list[dict], list[dict], str]:
        mcfg = self.cfg["montage"]
        if self.provider is None or not mcfg.get("enabled"):
            return [], [], "skipped (heuristic provider or disabled)"
        pre = rank(copy.deepcopy(singles), self.cfg)["ranked"][: mcfg["source_candidates"]]
        blocks, eps = [], set()
        for c in pre:
            seg = c["segments"][0]
            ep = episodes[seg["episode"]]
            evs = [e for s in c["segments"] for e in episodes[s["episode"]].events
                   if s["start"] - 0.05 <= e.start < s["end"] - 0.05]
            eps.add(seg["episode"])
            blocks.append(f"[{c['id']}] Episode {seg['episode']}: {c['title']}\nHook: {c['hook']}\n"
                          f"Cliffhanger: {c['cliffhanger']}\n" +
                          prompts.render_transcript(evs, prefix=f"E{ep.number:02d}-"))
        system, user = prompts.montage_prompt(show_title, blocks, mcfg["max_montages"])
        data, status = self.call("montage", "montage", system, user, prompts.MINING_SCHEMA)
        if data is None:
            return [], [], status
        resolved = [resolve(c, episodes, self.cfg, cid=f"M{i}", origin="montage", default_episode=None)
                    for i, c in enumerate(data.get("candidates", []), start=1)]
        resolved = self.repair(resolved, sorted(eps), "\n\n".join(blocks), episodes, "repair_montage",
                               default_episode=None, origin="montage")
        return [c for c in resolved if c["valid"]], [c for c in resolved if not c["valid"]], "ok"

    # -- context-free AI review -------------------------------------------------------------------
    def cold_view(self, ranked: list[dict]) -> str:
        rcfg = self.cfg["review"]
        if self.provider is None or not rcfg.get("cold_viewer"):
            return "skipped (heuristic provider or disabled)"
        statuses = []
        for c in ranked[: rcfg["cold_viewer_top_n"]]:
            clip = [f"[{t['clip_time']:5.1f}s] " + t["render"].split("] ", 1)[1] for t in c["transcript"]]
            first3 = [line for line, t in zip(clip, c["transcript"]) if t["clip_time"] <= 3.0]
            system, user = prompts.cold_viewer_prompt(clip, first3)
            data, status = self.call(f"review_{c['id']}", "cold_viewer", system, user, prompts.COLD_VIEWER_SCHEMA)
            c["cold_viewer"] = {"status": status, "result": data,
                                "reviewer": f"{self.provider.describe()}" if data else None}
            statuses.append(status)
        return "ok" if statuses and all(s == "ok" for s in statuses) else (
            ",".join(sorted(set(statuses))) or "nothing to review")


def run(input_path: str | Path, out_dir: str | Path, cfg: dict, show_title: str | None = None,
        responder_label: str = "unspecified") -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    disc = discover(input_path)
    if not disc.episodes:
        raise RunAborted(f"No screenplay episodes found under {input_path} (skipped: {disc.skipped})")
    episodes = {ep.number: ep for ep in disc.episodes}
    show_title = show_title or Path(input_path).name.removesuffix("-script")
    parse_report = {"input": str(input_path), "episodes_found": len(disc.episodes), "skipped_files": disc.skipped,
                    "warnings": disc.warnings, "episodes": [ep.summary() for ep in disc.episodes]}
    (out / "parse_report.json").write_text(json.dumps(parse_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Parsed {len(disc.episodes)} episodes ({sum(len(e.events) for e in disc.episodes)} timed events); "
         f"skipped files: {len(disc.skipped)}")

    provider = build_provider(cfg, out, responder_label=responder_label)
    runner = Runner(cfg, out, provider, responder_label)
    outline = "\n".join(f"Ep {e.number}: {e.logline}" for e in disc.episodes)
    _log(f"Provider: {provider.describe() if provider else 'heuristic fallback (no model)'}")

    workers = cfg["provider"].get("workers", 4) if (provider and provider.live) else 1
    _log("Stage 1/4: mining episodes")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_episode = list(pool.map(lambda ep: runner.mine(ep, episodes, show_title, outline), disc.episodes))

    singles = [c for r in per_episode for c in r["candidates"]]
    rejected = [c for r in per_episode for c in r["rejected"]]
    pending_eps = [r["episode"] for r in per_episode if r["status"] == "pending"]
    failed_eps = [r["episode"] for r in per_episode if r["status"] == "error"]

    _log("Stage 2/4: cross-episode montage")
    if pending_eps or failed_eps:
        montages, m_rejected, montage_status = [], [], "skipped: mining incomplete"
    else:
        montages, m_rejected, montage_status = runner.montage(singles, episodes, show_title)
    rejected += m_rejected

    _log("Stage 3/4: ranking + cold-viewer review")
    ranking = rank(singles + montages, cfg)
    review_status = "skipped: mining incomplete" if pending_eps else runner.cold_view(ranking["ranked"])
    ranking = rank(singles + montages, cfg)  # re-rank with review penalties/confidence applied
    if provider is not None and cfg["review"].get("cold_viewer") and not pending_eps:
        for _ in range(3):  # reviews can move candidates into the shortlist; every shortlisted pick gets reviewed
            missing = [c for c in ranking["shortlist"] if not c.get("cold_viewer")]
            if not missing:
                break
            runner.cold_view(missing)
            ranking = rank(singles + montages, cfg)

    status = "complete" if not (pending_eps or failed_eps) else "incomplete"
    usage = {"input_tokens": 0, "output_tokens": 0}
    for t in runner.tasks:
        for k in usage:
            usage[k] += (t.get("usage") or {}).get(k, 0) or 0
    manifest = {
        "tool": "promo-finder", "version": VERSION, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1), "status": status,
        "input": str(input_path), "show_title": show_title,
        "episodes_found": len(disc.episodes), "episodes_mined_ok": sum(r["status"] == "ok" for r in per_episode),
        "episodes_pending": pending_eps, "episodes_failed": failed_eps, "skipped_files": disc.skipped,
        "provider": cfg["provider"]["name"], "provider_detail": provider.describe() if provider else "heuristic",
        "responder": responder_label if cfg["provider"]["name"] == "file" else None,
        "stages": {"montage": montage_status, "cold_viewer": review_status},
        "tasks_pending": [t["task_id"] for t in runner.tasks if t["status"] == "pending"],
        "tasks_failed": [{"task_id": t["task_id"], "error": t["error"]} for t in runner.tasks if t["status"] == "error"],
        "counts": {"candidates_valid": len(singles) + len(montages), "candidates_rejected": len(rejected),
                   "montages_valid": len(montages), "shortlist": len(ranking["shortlist"]),
                   "alternatives": len(ranking["alternatives"]), "duplicates": len(ranking["duplicates"])},
        "usage_measured": usage,
        "cost_estimate": report.cost_estimate(runner.tasks, cfg),
        "tasks": runner.tasks,
        "config": cfg,
    }
    results = {"manifest": manifest, "episodes": per_episode_summary(per_episode),
               "candidates": singles + montages, "rejected": rejected,
               "ranking": {k: [c["id"] for c in v] for k, v in ranking.items()}}
    report.write_all(out, results, ranking, parse_report)
    _log(f"Done: status={status}; {len(singles) + len(montages)} valid candidates, {len(rejected)} rejected; "
         f"outputs in {out}")
    return results


def per_episode_summary(per_episode: list[dict]) -> list[dict]:
    return [{"episode": r["episode"], "status": r["status"], "episode_note": r["episode_note"],
             "valid": [c["id"] for c in r["candidates"]], "rejected": [c["id"] for c in r["rejected"]]}
            for r in per_episode]


def rerank(results_path: str | Path, cfg: dict, out_dir: str | Path) -> dict:
    """Re-score and re-rank a finished run with new weights/penalties. No model calls."""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    cands = copy.deepcopy(results["candidates"])
    ranking = rank(cands, cfg)
    results["candidates"] = cands
    results["ranking"] = {k: [c["id"] for c in v] for k, v in ranking.items()}
    results["manifest"]["reranked_with"] = {"weights": cfg["weights"], "penalties": cfg["penalties"],
                                            "ranking": cfg["ranking"]}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.write_all(out, results, ranking, None)
    return results
