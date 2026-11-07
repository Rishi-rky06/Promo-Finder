"""Command line interface.

    python -m promo_finder run <screenplays dir|file|zip> --out outputs/my_run [--provider heuristic|anthropic|openai_compatible|file]
    python -m promo_finder rerank outputs/my_run/results.json --weights hook=0.35,cliffhanger=0.3 --out outputs/my_run_rerank
    python -m promo_finder parse <screenplays>
    python -m promo_finder check-provider --provider anthropic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, load_dotenv, parse_weight_overrides
from .parser import discover
from .pipeline import RunAborted, rerank, run
from .providers import ProviderError, build_provider


def _cfg(args) -> dict:
    over: dict = parse_weight_overrides(getattr(args, "weights", None))
    prov = getattr(args, "provider", None)
    if prov:
        over.setdefault("provider", {})["name"] = prov
    model = getattr(args, "model", None)
    if model:
        name = prov or load_config(args.config)["provider"]["name"]
        if name in ("anthropic", "openai_compatible"):
            over.setdefault("provider", {}).setdefault(name, {})["model"] = model
    if getattr(args, "no_montage", False):
        over["montage"] = {"enabled": False}
    if getattr(args, "no_review", False):
        over["review"] = {"cold_viewer": False}
    return load_config(args.config, over)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find and rank promo clips in microdrama screenplays.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a screenplay file, folder or .zip")
    r.add_argument("input")
    r.add_argument("--out", default="outputs/run")
    r.add_argument("--config")
    r.add_argument("--provider", choices=["anthropic", "openai_compatible", "file", "heuristic"])
    r.add_argument("--model", help="override the model id for the chosen provider")
    r.add_argument("--weights", help="e.g. hook=0.3,cliffhanger=0.3 (others keep config values)")
    r.add_argument("--show-title")
    r.add_argument("--responder", default="unspecified",
                   help="file provider only: who/what answered the exported prompts (recorded in the manifest)")
    r.add_argument("--no-montage", action="store_true")
    r.add_argument("--no-review", action="store_true")

    rr = sub.add_parser("rerank", help="re-score a finished run with new weights (no model calls)")
    rr.add_argument("results")
    rr.add_argument("--out", required=True)
    rr.add_argument("--config")
    rr.add_argument("--weights")

    p = sub.add_parser("parse", help="parse only and print the parse report")
    p.add_argument("input")

    c = sub.add_parser("check-provider", help="send one tiny request to confirm credentials and model id work")
    c.add_argument("--config")
    c.add_argument("--provider", choices=["anthropic", "openai_compatible"])
    c.add_argument("--model")

    lm = sub.add_parser("list-models", help="list model ids available to your key on an OpenAI-compatible endpoint")
    lm.add_argument("--config")

    load_dotenv()
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            cfg = _cfg(args)
            res = run(args.input, args.out, cfg, show_title=args.show_title, responder_label=args.responder)
            m = res["manifest"]
            print(json.dumps({"status": m["status"], "episodes_found": m["episodes_found"],
                              "episodes_mined_ok": m["episodes_mined_ok"], "pending": m["episodes_pending"],
                              "failed": m["episodes_failed"], **m["counts"], "out": str(Path(args.out).resolve())},
                             indent=2))
            return 0 if m["status"] == "complete" else 2
        if args.cmd == "rerank":
            cfg = _cfg(args)
            res = rerank(args.results, cfg, args.out)
            print("shortlist:", ", ".join(res["ranking"]["shortlist"]))
            return 0
        if args.cmd == "parse":
            d = discover(args.input)
            for ep in d.episodes:
                s = ep.summary()
                print(f"Episode {s['episode']:>3}  {s['duration_tc']}  scenes {s['scenes']}  events {s['events']:>4}  "
                      f"anomalies {len(s['anomalies'])}  {s['source_file']}")
                for a in s["anomalies"]:
                    print(f"      ! {a}")
            print(f"{len(d.episodes)} episodes; skipped {d.skipped}; warnings {d.warnings}")
            return 0
        if args.cmd == "list-models":
            import os
            from openai import OpenAI

            c = load_config(args.config)["provider"]["openai_compatible"]
            key = os.environ.get(c["api_key_env"], "")
            if not key:
                raise ProviderError(f"{c['api_key_env']} is not set (put it in .env)", fatal=True)
            ids = sorted(m.id for m in OpenAI(base_url=c["base_url"], api_key=key).models.list())
            print(chr(10).join(ids))
            print(f"{len(ids)} models at {c['base_url']}", file=sys.stderr)
            return 0
        if args.cmd == "check-provider":
            cfg = _cfg(args)
            prov = build_provider(cfg, Path("."))
            if prov is None:
                print("heuristic provider needs no credentials")
                return 0
            res = prov.inner.generate_json("check", "Reply in JSON.", 'Return {"ok": true}.',
                                           {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                            "required": ["ok"], "additionalProperties": False})
            print(f"OK: {res.provider} {res.model} -> {res.data} usage={res.usage}")
            return 0
    except (RunAborted, ProviderError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
