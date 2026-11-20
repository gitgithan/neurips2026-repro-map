"""Step 1: download the hackathon's official claim list and save it as data/claims.json.

Source: https://huggingface.co/datasets/Neurips-2026-MLIntern-Repro/papers (public, no token needed), pinned to one revision
so that a re-run gives the same rows. Adds one derived field, closed_api_hint (the paper's text names a hosted model).
Usage: uv run python pipeline/1_fetch_claims.py"""
import json, os, re, tomllib
os.environ.setdefault("PYTHONHASHSEED", "0")
from pathlib import Path
from datasets import load_dataset
ROOT = Path(__file__).resolve().parent.parent
REPO, REV = "Neurips-2026-MLIntern-Repro/papers", "8187c351d8201824c2e606d9fb8e299b4709a98c"
CLOSED = re.compile(tomllib.loads((ROOT / "signals.toml").read_text())["closed_api"]["pattern"], re.I)
KEEP = ["arxiv_id", "title", "authors", "presentation", "track", "difficulty", "background", "method", "claim", "claim_source", "metric",
        "comparison", "falsification_criterion", "image_kind", "caption"]
ds = load_dataset(REPO, split="train", revision=REV)
ds = ds.remove_columns([c for c in ds.column_names if c == "image"])   # the claim screenshots are not needed (and would require Pillow)
print("columns:", ds.column_names)
rows = []
for row_index, r in enumerate(ds):
    o = {k: r.get(k) for k in KEEP}; o["row"] = row_index; o["dataset_revision"] = REV     # position in the official list (links to that exact record) and the revision it was read from
    o["authors"] = list(o["authors"]) if isinstance(o["authors"], (list, tuple)) else [a.strip() for a in str(o["authors"] or "").split(",") if a.strip()]
    for k in r:                                    # keep whatever URL columns the dataset has (arXiv, OpenReview, ...)
        if k.endswith("_url") and isinstance(r[k], str): o[k] = r[k]
    o.setdefault("arxiv_url", f"https://arxiv.org/abs/{o['arxiv_id']}")
    text = " ".join(str(o.get(k) or "") for k in ("title", "background", "method", "claim", "metric", "comparison"))
    o["closed_api_hint"] = bool(CLOSED.search(text))
    rows.append(o)
assert len({o["arxiv_id"] for o in rows}) == len(rows)
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data/claims.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print(f"wrote data/claims.json: {len(rows)} papers from {REPO}@{REV[:8]}; closed-API hint on {sum(o['closed_api_hint'] for o in rows)}")
