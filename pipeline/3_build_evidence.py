"""Step 3: one evidence record per paper -> data/papers_evidence.json (+ a flat data/papers_evidence.csv), plus the map layout.

Reads data/claims.json (step 1) and data/audit_results.json, data/audit/ (step 2). Nothing is scored against anyone's taste here:
the record holds what was found and where, three small heuristic scores (turnkey, checkability, readability) with their inputs,
and a 2D position + nearest neighbours from a text embedding of title + claim.
Usage: uv run --group embed python pipeline/3_build_evidence.py [--no-embed]   (--no-embed reuses data/embeddings.npy)"""
import argparse, csv, json, os, re, tomllib
os.environ.setdefault("PYTHONHASHSEED", "0")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"; AUD = DATA / "audit"
CFG = tomllib.loads((ROOT / "signals.toml").read_text()); CK, MAP = CFG["checkability"], CFG["map"]
EMB_REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ap = argparse.ArgumentParser(); ap.add_argument("--no-embed", action="store_true"); args = ap.parse_args()
jl = lambda p, d=None: json.loads(Path(p).read_text()) if Path(p).exists() else d
claims = sorted(jl(DATA / "claims.json"), key=lambda r: r["arxiv_id"])
audit = {o["arxiv_id"]: o for o in jl(DATA / "audit_results.json")}
scan = jl(AUD / "repo_audit.json", {}); pages = jl(AUD / "project_pages.json", {}); clones = jl(AUD / "clone_status.json", {})
SPREAD = re.compile(r"\b\d+\s+(?:random\s+)?(?:seeds|runs|trials|replicates|repetitions)|standard (?:deviation|error)|confidence interval|±|\+/-", re.I)
MANY = re.compile(r"\b(\d{1,3}|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:\w+\s+){0,2}(benchmarks|datasets|tasks|suites|environments|domains|games)\b", re.I)
WORDNUM = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
clip = lambda v: max(0.0, min(10.0, v))
EMAIL = re.compile(r"[\w.+\-{}|,]*\s?@\s?[\w-]+(?:\.[\w-]+)+")   # also catches grouped forms such as {name1,name2}@host
HOME = re.compile(r"(/home/|/Users/)[\w.-]+"); KEY = re.compile(r"\b(?:hf_|sk-|ghp_)[A-Za-z0-9]{20,}")
def short(s, n=170):
    """Quoted evidence is trimmed and scrubbed: no e-mail addresses, no personal home paths, nothing that looks like a token."""
    return KEY.sub("[token removed]", HOME.sub(r"\1…", EMAIL.sub("[e-mail removed]", " ".join(str(s).split()))))[:n]

def checkability(r, a):
    why, v = [], CK["base"]
    def add(key, text):
        nonlocal v; v += CK[key]; why.append(f"{CK[key]:+d} {text}")
    if r["image_kind"] == "table": add("table_claim", "claim is a table (exact numbers)")
    if SPREAD.search(str(r["metric"])): add("reports_spread", "metric reports a spread or number of runs")
    if re.search(r"\d", str(r["claim"])): add("numbers_in_claim", "claim sentence states numbers")
    nb = max([WORDNUM.get(x.lower()) or int(x) for x, _ in MANY.findall(f"{r['claim']} {r['metric']}")] or [0])
    if 5 <= nb <= 200: add("many_benchmarks", f"claim aggregates over {nb} benchmarks / tasks")
    if a.get("claim_in_appendix"): add("claim_in_appendix", "claimed result is in the appendix")
    return round(clip(v), 1), why

recs = []
for r in claims:
    aid = r["arxiv_id"]; a = audit.get(aid, {}); pdf = jl(AUD / "pdf" / f"{aid}.json", {}) or {}; sc = scan.get(aid, {}) if a.get("repo_status") == "cloned" else {}
    cands = pdf.get("link_candidates") or []
    def links(kind, n=8):
        out = []
        for c in sorted([c for c in cands if c["kind"] == kind], key=lambda c: -c.get("own_score", 0))[:n]:
            out.append({"url": c["url"], "pages": c["pages"][:4], "where": sorted(set(c.get("regions") or [])), "likely_own": c.get("own_score", 0) >= 4,
                        "context": short((c.get("contexts") or [""])[0], 200)})
        return out
    ev = lambda d: {k: short((v or [""])[0]) for k, v in (d or {}).items()}
    wts = sc.get("friction_weights") or {}
    ck, ck_why = checkability(r, a)
    repo = None
    if a.get("repo_url"):
        c = clones.get(aid, {})
        repo = {"url": a["repo_url"], "found_via": a.get("repo_link_source"), "confidence": a.get("repo_link_confidence"), "why_this_one": a.get("repo_link_why"),
                "status": a.get("repo_status"), "head_sha": a.get("repo_head_sha"), "head_date": a.get("repo_head_date"), "size_mb": c.get("size_mb"),
                "large_files_skipped": c.get("n_large_blobs_skipped")}
    recs.append({
        "arxiv_id": aid, "title": r["title"],
        "labels": {"presentation": r["presentation"], "track": r["track"], "difficulty": r["difficulty"], "claim_kind": r["image_kind"]},
        "claim_source": r.get("claim_source"),     # where the organisers say the claim is, e.g. "Table 2, page 6"
        # copied unchanged from the organisers' dataset so that a row can be read on its own; that dataset is the authority
        "official": {"claim": r.get("claim"), "metric": r.get("metric"), "comparison": r.get("comparison"), "falsification_criterion": r.get("falsification_criterion"),
                     "copied_from": "https://huggingface.co/datasets/Neurips-2026-MLIntern-Repro/papers", "dataset_revision": r.get("dataset_revision")},
        "links": {"official_claim_record": f"https://huggingface.co/datasets/Neurips-2026-MLIntern-Repro/papers/viewer/default/train?row={r['row']}" if r.get("row") is not None else None,
                  "arxiv": r.get("arxiv_url"), "pdf": f"https://arxiv.org/pdf/{aid}", "openreview": r.get("openreview_url"), "papers_with_code": r.get("papers_with_code_url"),
                  "repo": repo, "github_in_pdf": links("github"), "huggingface_in_pdf": links("hf"), "project_pages": links("project_page"), "anonymous_code": links("anonymous4open"),
                  "github_on_project_page": (pages.get(aid) or {}).get("github_links") or []},
        "paper": {"pdf_status": a.get("pdf_status"), "pages": a.get("total_pages"), "main_body_pages": a.get("main_body_pages"), "appendix_pages": a.get("appendix_pages"),
                  "claim_page": a.get("claim_page"), "claim_location_check": a.get("claim_location_check"), "claim_in_appendix": a.get("claim_in_appendix"), "read_mins": a.get("claim_reading_mins"), "read_basis": a.get("reading_basis"),
                  "equation_density": a.get("equation_density"), "theorems_in_main_body": a.get("theorem_count_main"),
                  "reference_count": a.get("reference_count"), "reference_count_basis": a.get("reference_count_basis"),
                  "appendix_mentions": {k: (a.get("appendix_evidence") or {}).get(k, 0) for k in ("hyperparameter", "prompt_template", "seeds")},
                  "code_statements": {k: short(v, 240) for k, v in (a.get("code_statements") or {}).items()},
                  "hardware_mentions": [short(m, 160) for m in (a.get("compute_mentions") or [])[:4]], "names_hosted_model": bool(r.get("closed_api_hint"))},
        "repo_scan": None if not sc or "turnkey_score" not in sc else {
            "n_files": sc.get("n_files"), "n_python_files": sc.get("n_py"), "n_notebooks": sc.get("n_ipynb"), "readme_chars": sc.get("readme_chars"), "dependency_files": sc.get("dep_spec") or [],
            "good_signs": ev(sc.get("boosters")), "friction": {k: {"weight": wts.get(k, 1), "evidence": v} for k, v in ev(sc.get("soft_friction")).items()},
            "blockers": ev(sc.get("hard_blockers")), "ignored_matches": ev(sc.get("ignored_signals")), "closed_api_use": sc.get("closed_api"),
            "licence": sc.get("licence"), "has_tests": bool(sc.get("has_tests")), "facts": ev(sc.get("facts")),
            "score_before_blockers": sc.get("turnkey_raw_before_blockers")},
        "scores": {"turnkey": a.get("turnkey_score"), "turnkey_basis": "repo scan" if sc and "turnkey_score" in sc else ("no repo link found" if a.get("repo_status") == "no_link_found" else a.get("repo_status")),
                   "checkability": ck, "checkability_why": ck_why, "readability": a.get("readability_score")},
    })

# ---- map: embedding of title + claim only -> 2D layout, nearest neighbours, automatic topic names
SRC = {c["arxiv_id"]: c for c in claims}     # the organisers' rows; only the four "official" fields above are copied into the evidence files
texts = [f"{r['title']}. {SRC[r['arxiv_id']]['claim']}" for r in recs]; emb_path = DATA / "embeddings.npy"
if args.no_embed and emb_path.exists() and np.load(emb_path).shape[0] == len(texts): emb = np.load(emb_path); print("embeddings: reused", emb.shape)
else:
    import torch; torch.manual_seed(0); torch.use_deterministic_algorithms(True, warn_only=True)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(MAP["embedding_model"], revision=EMB_REV, device="cpu").encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    np.save(emb_path, emb); print("embeddings: encoded", emb.shape)
import umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
sim = emb @ emb.T; np.fill_diagonal(sim, -1); nb = np.argsort(-sim, axis=1)[:, :MAP["neighbours"]]
xy = umap.UMAP(n_components=2, n_neighbors=MAP["umap_neighbors"], min_dist=MAP["umap_min_dist"], spread=1.5, metric="cosine", random_state=MAP["seed"]).fit_transform(emb)
xy = xy - xy.mean(0); xy = xy / np.abs(xy).max()
lab = KMeans(n_clusters=MAP["topics"], random_state=0, n_init=10).fit_predict(emb)
tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.25, token_pattern=r"[A-Za-z][A-Za-z-]{2,}")
X = tf.fit_transform([f"{r['title']} {r['title']} {SRC[r['arxiv_id']]['claim']}" for r in recs]); vocab = np.array(tf.get_feature_names_out()); names = {}
for k in range(MAP["topics"]):
    keep = []
    for t in vocab[np.asarray(X[lab == k].mean(0)).ravel().argsort()[::-1][:12]]:
        if not any(t in u or u in t for u in keep): keep.append(t)
    names[k] = " / ".join(keep[:3])
cent = np.stack([emb[lab == k].mean(0) for k in range(MAP["topics"])]); cent /= np.linalg.norm(cent, axis=1, keepdims=True); csim = emb @ cent.T
for i, r in enumerate(recs):
    also = [int(k) for k in np.argsort(-csim[i]) if k != lab[i] and csim[i, k] >= csim[i, lab[i]] - 0.02][:2]     # a paper can sit between topics
    r["auto_labels"] = {"source": "computed by this tool from the text; not assigned by authors, the conference or the hackathon", "topics": [names[int(lab[i])]] + [names[k] for k in also], "areas": []}
    r["map"] = {"x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "topic": int(lab[i]), "topic_name": names[int(lab[i])],
                "similar": [[recs[j]["arxiv_id"], round(float(sim[i, j]), 3)] for j in nb[i]]}
# ---- search vocabulary: terms a visitor might type, each with a vector in the same embedding space as the papers
# (a) words and two-word phrases that occur in at least 3 papers; (b) the glossary, embedded by its plain-language description.
# The page looks a typed term up here, so it can search by meaning without loading any model in the browser.
voc_vec, voc_meta = DATA / "search_vectors.npy", DATA / "search_terms.json"
gloss = tomllib.loads((ROOT / "search_glossary.toml").read_text())["terms"]
from sklearn.feature_extraction.text import CountVectorizer
docs = [" ".join(str(x or "") for x in (r["title"], SRC[r["arxiv_id"]]["claim"], SRC[r["arxiv_id"]]["method"])) for r in recs]
cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.3, binary=True, token_pattern=r"[A-Za-z][A-Za-z0-9+]{2,}")
dfc = np.asarray(cv.fit_transform(docs).sum(0)).ravel(); cand = sorted(zip(cv.get_feature_names_out(), dfc), key=lambda t: -t[1])
uni = [(t, int(c)) for t, c in cand if " " not in t][:5000]; bi = [(t, int(c)) for t, c in cand if " " in t and c >= 4][:1500]
terms = {t: {"n": c, "kind": "corpus"} for t, c in uni + bi}
for t, d in gloss.items(): terms[t.lower()] = {"n": terms.get(t.lower(), {}).get("n", 0), "kind": "glossary", "means": d}
names = sorted(terms)
if args.no_embed and voc_vec.exists() and jl(voc_meta, {}).get("names") == names and jl(voc_meta, {}).get("gloss") == gloss: TV = np.load(voc_vec).astype("float32"); print("search vectors: reused", TV.shape)
else:
    from sentence_transformers import SentenceTransformer
    PRE = "Represent this sentence for searching relevant passages: "
    TV = SentenceTransformer(MAP["embedding_model"], revision=EMB_REV, device="cpu").encode([PRE + terms[t].get("means", t) for t in names], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    np.save(voc_vec, TV.astype("float16")); print("search vectors: encoded", TV.shape)
# papers embedded once more with the method summary included: better recall for search than title + claim alone
ps_path = DATA / "embeddings_search.npy"
if args.no_embed and ps_path.exists() and np.load(ps_path).shape[0] == len(recs): print("search paper embeddings: reused")
else:
    from sentence_transformers import SentenceTransformer
    PS = SentenceTransformer(MAP["embedding_model"], revision=EMB_REV, device="cpu").encode([f"{r['title']}. {SRC[r['arxiv_id']]['claim']} {(SRC[r['arxiv_id']]['method'] or '')[:500]}" for r in recs], batch_size=16, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    np.save(ps_path, PS); print("search paper embeddings: encoded", PS.shape)
voc_meta.write_text(json.dumps({"names": names, "gloss": gloss, "terms": terms}, ensure_ascii=False))
# areas: glossary terms whose description is unusually close to the paper (several per paper are normal; none is common)
PS = np.load(ps_path); gl = [t for t in names if terms[t]["kind"] == "glossary"]; G = TV[[names.index(t) for t in gl]]
Z = PS @ G.T; Z = (Z - Z.mean(0)) / Z.std(0)
for i, r in enumerate(recs):
    r["auto_labels"]["areas"] = [gl[j] for j in np.argsort(-Z[i])[:3] if Z[i, j] >= 2.5]
(DATA / "papers_evidence.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
# ---- the CSV is a flat view of the JSON: every column is computed from one JSON record by the rule written next to it.
# (csv column, where it comes from in the JSON record, how)
def _scan(r, key): return "; ".join(r["repo_scan"][key]) if r["repo_scan"] else "not scanned"     # blank = scanned and nothing found
def _fact(r, key): return (key in r["repo_scan"]["facts"]) if r["repo_scan"] else "not scanned"
_repo = lambda r: r["links"]["repo"] or {}
CSV_COLUMNS = [
    ("arxiv_id", "arxiv_id", lambda r: r["arxiv_id"]), ("title", "title", lambda r: r["title"]),
    ("presentation", "labels.presentation", lambda r: r["labels"]["presentation"]), ("track", "labels.track", lambda r: r["labels"]["track"]),
    ("difficulty", "labels.difficulty", lambda r: r["labels"]["difficulty"]), ("claim_kind", "labels.claim_kind", lambda r: r["labels"]["claim_kind"]),
    ("claim_source", "claim_source", lambda r: r["claim_source"]),
    ("official_claim", "official.claim", lambda r: r["official"]["claim"]), ("official_metric", "official.metric", lambda r: r["official"]["metric"]),
    ("official_comparison", "official.comparison", lambda r: r["official"]["comparison"]),
    ("official_falsification_criterion", "official.falsification_criterion", lambda r: r["official"]["falsification_criterion"]),
    ("repo_url", "links.repo.url", lambda r: _repo(r).get("url")), ("repo_status", "links.repo.status (no_link_found when links.repo is null)", lambda r: _repo(r).get("status") or "no_link_found"),
    ("repo_last_commit", "links.repo.head_date (date part)", lambda r: (_repo(r).get("head_date") or "")[:10]),
    ("turnkey", "scores.turnkey", lambda r: r["scores"]["turnkey"]), ("turnkey_basis", "scores.turnkey_basis", lambda r: r["scores"]["turnkey_basis"]),
    ("blockers", "repo_scan.blockers (names)", lambda r: _scan(r, "blockers")), ("friction", "repo_scan.friction (names)", lambda r: _scan(r, "friction")),
    ("good_signs", "repo_scan.good_signs (names)", lambda r: _scan(r, "good_signs")),
    ("licence", "repo_scan.licence", lambda r: (r["repo_scan"]["licence"] or "none found") if r["repo_scan"] else "not scanned"),
    ("has_tests", "repo_scan.has_tests", lambda r: r["repo_scan"]["has_tests"] if r["repo_scan"] else "not scanned"),
    ("demo_code", "repo_scan.facts has demo_code", lambda r: _fact(r, "demo_code")), ("environment_code", "repo_scan.facts has environment_code", lambda r: _fact(r, "environment_code")),
    ("checkability", "scores.checkability", lambda r: r["scores"]["checkability"]), ("checkability_why", "scores.checkability_why (joined)", lambda r: " | ".join(r["scores"]["checkability_why"])),
    ("readability", "scores.readability", lambda r: r["scores"]["readability"]), ("read_mins", "paper.read_mins", lambda r: r["paper"]["read_mins"]),
    ("pages", "paper.pages", lambda r: r["paper"]["pages"]), ("reference_count", "paper.reference_count", lambda r: r["paper"]["reference_count"]),
    ("claim_page", "paper.claim_page", lambda r: r["paper"]["claim_page"]), ("claim_in_appendix", "paper.claim_in_appendix", lambda r: r["paper"]["claim_in_appendix"]),
    ("claim_location_check", "paper.claim_location_check", lambda r: r["paper"]["claim_location_check"]),
    ("names_hosted_model", "paper.names_hosted_model", lambda r: r["paper"]["names_hosted_model"]),
    ("hardware_mentions", "paper.hardware_mentions (count)", lambda r: len(r["paper"]["hardware_mentions"])),
    ("code_statements", "paper.code_statements (kinds)", lambda r: "; ".join(r["paper"]["code_statements"])),
    ("other_github_links_in_pdf", "links.github_in_pdf (count, excluding links.repo.url)", lambda r: sum(1 for g in r["links"]["github_in_pdf"] if g["url"].lower() != (_repo(r).get("url") or "").lower())),
    ("huggingface_links_in_pdf", "links.huggingface_in_pdf (count)", lambda r: len(r["links"]["huggingface_in_pdf"])),
    ("project_page", "links.project_pages[0].url", lambda r: (r["links"]["project_pages"] or [{}])[0].get("url")),
    ("auto_topics", "auto_labels.topics (joined)", lambda r: "; ".join(r["auto_labels"]["topics"])), ("auto_areas", "auto_labels.areas (joined)", lambda r: "; ".join(r["auto_labels"]["areas"])),
    ("official_claim_record", "links.official_claim_record", lambda r: r["links"]["official_claim_record"]),
    ("arxiv", "links.arxiv", lambda r: r["links"]["arxiv"]), ("openreview", "links.openreview", lambda r: r["links"]["openreview"]),
]
with open(DATA / "papers_evidence.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow([c for c, _, _ in CSV_COLUMNS])
    for r in recs: w.writerow([fn(r) for _, _, fn in CSV_COLUMNS])
(DATA / "csv_columns.json").write_text(json.dumps([{"csv_column": c, "json_path": p_} for c, p_, _ in CSV_COLUMNS], indent=1))
# consistency check: read both files back and compare every cell with the value computed from the JSON
back = json.loads((DATA / "papers_evidence.json").read_text()); norm = lambda v: "" if v is None else str(v)
with open(DATA / "papers_evidence.csv", newline="") as f: rows = list(csv.DictReader(f))
assert len(rows) == len(back) == len({x["arxiv_id"] for x in rows}), "row count / ids differ between CSV and JSON"
bad = [(x["arxiv_id"], c) for x, r in zip(rows, back) for c, _, fn in CSV_COLUMNS if x[c] != norm(fn(r))]
assert not bad, f"CSV and JSON disagree: {bad[:5]}"
print(f"consistency check: {len(rows)} rows x {len(CSV_COLUMNS)} columns in the CSV all equal the values in the JSON")
n_repo = sum(bool(r["repo_scan"]) for r in recs)
print(f"wrote data/papers_evidence.json + .csv: {len(recs)} papers, {n_repo} with a scanned repo, {sum(bool(r['repo_scan'] and r['repo_scan']['blockers']) for r in recs)} with a blocker; topics: {names}")
