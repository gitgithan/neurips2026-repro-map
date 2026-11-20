"""Step 2: scan every paper's PDF and (if it links one) its GitHub repository.

Stages (each resumable, each caches under data/audit/):
  download  arXiv PDFs -> data/pdfs/<id>.pdf           (sequential, ~1 request / 3 s, backs off on 429/5xx)
  parse     PyMuPDF: pages, sections, every link with its page and context -> data/audit/pdf/<id>.json
  pages     fetch *.github.io project pages of papers with no GitHub link in the PDF, look for a repo link
  head      HTTP HEAD on each primary GitHub URL        -> data/audit/github_head.json
  clone     git clone --depth 1 --filter=blob:limit=2m --no-checkout, then check out only small blobs
  scan      read the cloned files and match the patterns in signals.toml (READ ONLY - nothing from a repo is ever executed)
  merge     combine everything -> data/audit_results.json

Usage: uv run python pipeline/2_scan_pdfs_and_repos.py <stage> [--only id,id,...] [--limit N] [--force]
Nothing here is guessed: unknown values stay null and carry a *_basis / status field saying why.
"""
import os, sys
os.environ.setdefault("PYTHONHASHSEED", "0")
import re, json, time, random, asyncio, argparse, subprocess, shutil, traceback, collections, tomllib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
random.seed(0)

ROOT = Path(__file__).resolve().parent.parent
DATA, PDFS, AUD, REPOS = ROOT / "data", ROOT / "data/pdfs", ROOT / "data/audit", ROOT / "data/repos"
LOGDIR = ROOT / "logs"
UA = "paper-audit-script/0.1"
CFG = tomllib.loads((ROOT / "signals.toml").read_text())   # every pattern and weight used by the repo scan lives there


# ---------------------------------------------------------------- logging (tee + keep failed logs)
class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "a", buffering=1), sys.__stdout__
    def write(self, s): self.out.write(s); self.f.write(s)
    def flush(self): self.out.flush(); self.f.flush()

def start_log(name):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    p = LOGDIR / f"{name}.log"
    sys.stdout = sys.stderr = Tee(p)
    print(f"\n===== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} argv={sys.argv[1:]} =====")
    return p

def fail_log(p):
    n = 1
    while (p.parent / f"{p.stem}.fail{n}.log").exists(): n += 1
    sys.stdout.flush(); shutil.copy(p, p.parent / f"{p.stem}.fail{n}.log")

def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default

def jdump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False)); tmp.replace(p)

def papers():
    return jload(DATA / "claims.json")   # written by pipeline/1_fetch_claims.py

def priority_ids(rows):
    def key(r): return (r["presentation"] == "poster", r["difficulty"] != "standard", r["arxiv_id"])   # orals/spotlights first, only so partial runs are useful
    return [r["arxiv_id"] for r in sorted(rows, key=key)]


# ---------------------------------------------------------------- stage: download
def stage_download(args):
    import httpx
    PDFS.mkdir(parents=True, exist_ok=True)
    rows = papers(); ids = priority_ids(rows)
    if args.only: ids = [i for i in ids if i in args.only]
    st_path = AUD / "download_status.json"; status = jload(st_path, {})
    todo = [i for i in ids if not (PDFS / f"{i}.pdf").exists()]
    print(f"download: {len(ids)} ids, {len(ids) - len(todo)} cached, {len(todo)} to fetch")
    backoff, consec_fail = 60, 0
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": UA}) as c:
        for n, aid in enumerate(todo):
            t0 = time.time(); ok = False
            for attempt in range(4):
                try:
                    r = c.get(f"https://arxiv.org/pdf/{aid}")
                    if r.status_code == 200 and r.content[:5] == b"%PDF-":
                        (PDFS / f"{aid}.pdf.part").write_bytes(r.content); (PDFS / f"{aid}.pdf.part").rename(PDFS / f"{aid}.pdf")
                        status[aid] = {"status": "ok", "bytes": len(r.content), "final_url": str(r.url).split("/")[-1]}
                        ok = True; backoff = 60; break
                    if r.status_code in (429, 500, 502, 503, 504):
                        print(f"  {aid}: HTTP {r.status_code}, backing off {backoff}s (attempt {attempt + 1})")
                        time.sleep(backoff); backoff = min(backoff * 2, 900); continue
                    status[aid] = {"status": "failed", "http": r.status_code, "content_type": r.headers.get("content-type", "")}
                    break
                except Exception as e:
                    status[aid] = {"status": "failed", "error": type(e).__name__}
                    print(f"  {aid}: {type(e).__name__}, retry in 20s"); time.sleep(20)
            else:
                status[aid] = {"status": "failed", "error": "retries_exhausted"}
            consec_fail = 0 if ok else consec_fail + 1
            if not ok: print(f"  {aid}: FAILED {status.get(aid)}")
            if consec_fail >= 5:
                print("  5 consecutive failures - sleeping 10 min before continuing"); time.sleep(600); consec_fail = 0
            if n % 10 == 0:
                jdump(st_path, status); print(f"  [{n + 1}/{len(todo)}] {aid} {status.get(aid, {}).get('status')} {status.get(aid, {}).get('bytes', '')}")
            time.sleep(max(0.0, 3.0 - (time.time() - t0)))
    jdump(st_path, status)
    have = sum((PDFS / f"{i}.pdf").exists() for i in ids)
    print(f"download done: {have}/{len(ids)} PDFs on disk; failed: {[i for i in ids if not (PDFS / f'{i}.pdf').exists()]}")


# ---------------------------------------------------------------- stage: parse
GH_RX = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)
HF_RX = re.compile(r"(?:huggingface\.co|hf\.co)/((?:datasets/|spaces/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)", re.I)
IO_RX = re.compile(r"(?:https?://)?([A-Za-z0-9-]+)\.github\.io(/[A-Za-z0-9_./-]*)?", re.I)
ANON_RX = re.compile(r"anonymous\.4open\.science/r/[A-Za-z0-9_.-]+", re.I)
GH_SKIP_OWNERS = {"orgs", "features", "about", "settings", "topics", "sponsors", "user-attachments", "marketplace", "apps", "login", "site"}
INFRA_OWNERS = {"huggingface", "pytorch", "openai", "google", "google-research", "google-deepmind", "deepmind", "facebookresearch", "microsoft", "tensorflow",
                "nvidia", "nvlabs", "scikit-learn", "numpy", "scipy", "jax-ml", "meta-llama", "vllm-project", "unslothai", "langchain-ai", "ollama", "ggerganov",
                "anthropics", "open-mmlab", "lightning-ai", "ray-project", "farama-foundation", "pyg-team", "dmlc", "apple", "salesforce", "stanfordnlp",
                "mlfoundations", "rwightman", "karpathy", "lucidrains", "eleutherai", "bigscience-workshop", "allenai", "thudm", "qwenlm", "deepseek-ai", "black-forest-labs",
                "comfyanonymous", "automatic1111", "compvis", "stability-ai", "physical-intelligence", "lifelong-robot-learning", "openvla", "princeton-nlp", "swe-bench"}
OWN_RX = re.compile(r"(our|the)\s+(code|implementation|source|codebase|repository|data|dataset|benchmark|models?)|code\s*(and\s+\w+\s*)?(is|are|will be|can be)?\s*(publicly\s+|freely\s+|made\s+)*(available|released|open|found|accessed)"
                    r"|available\s+at|released\s+at|open-?sourced?\s+(at|on|in)|project\s+(page|website)|\bcode\s*:|github\s*:|repository\s*:|we\s+release|is\s+released|publicly\s+available", re.I)
HEAD_RX = re.compile(r"^(\d{1,2})((?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{1,80})$")
NUM_ONLY = re.compile(r"^(\d{1,2})((?:\.\d{1,2}){0,2})\.?$")
SEC_KIND = [("intro", re.compile(r"^introduction", re.I)),
            ("related", re.compile(r"related work|background|preliminar|literature|prior work|notation", re.I)),
            ("end", re.compile(r"conclusion|discussion|limitation|future work|broader impact|acknowledg|summary|ethic|reproducib|impact statement", re.I)),
            ("experiments", re.compile(r"experiment|evaluation|result|empirical|ablation|case stud|analys[ie]s|benchmarking|numerical|simulation stud", re.I))]
SETUP_RX = re.compile(r"setup|set-up|setting|implementation|detail|protocol|dataset|baseline|benchmark|configuration|tasks?\b|metric|training", re.I)
MATH_CH = set("=∑∫∏√≤≥∈∉∀∃∥‖→←↦∇∂×·⊤⊥∞≈≜∝⊙⊗⊕∩∪⊂⊆±−")
THM_RX = re.compile(r"\b(Theorem|Lemma|Proposition|Corollary)\s+\d|\bProof\b")
GPU_RX = re.compile(r"((?:\d+\s*[×x]\s*)?(?:NVIDIA\s+)?(?:Tesla\s+)?(?:A100|H100|H200|V100|A6000|A5000|A40|L40S?|RTX\s?-?\d{4}\s?\w{0,3}|\bT4\b|\bL4\b|TITAN\s?\w+|P100|B200|GB200|TPU\s?v?\d\w*))"
                    r"|(\d[\d,.]*\s*(?:k\s*)?(?:GPU|CPU|TPU)[- ]?(?:hours|days))|(single\s+(?:GPU|CPU)|CPU[- ]only|on\s+a\s+laptop|consumer[- ]grade\s+GPU|without\s+(?:a\s+)?GPU|M[1-4]\s+(?:Pro|Max|Ultra)?\s*MacBook)", re.I)
WORD_RX = re.compile(r"[A-Za-z]{2,}")
nwords = lambda s: len(WORD_RX.findall(s))

def is_eq_line(l):
    l = l.strip()
    if not l or len(l) > 90: return False
    mathy = sum((ch in MATH_CH) or ("Ͱ" <= ch <= "Ͽ") or ("\U0001d400" <= ch <= "\U0001d7ff") for ch in l)
    return mathy >= 1 and len(re.findall(r"[A-Za-z]{4,}", l)) <= 3

def eq_density_raw(text):
    lines = [l for l in text.split("\n") if l.strip()]
    return round(sum(is_eq_line(l) for l in lines) / len(lines), 4) if lines else None

def title_keys(title):
    head = title.split(":")[0] if ":" in title else ""
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", head)] if head and len(head.split()) <= 4 else []
    return [t for t in toks if len(t) >= 3 and t not in {"the", "for", "and", "with", "via", "from", "towards"}]

_CODE = re.compile(r"\b(source code|code|implementation|scripts?|codebase)\b", re.I)
_ASSET = re.compile(r"\b(code|implementation|codebase|data|datasets?|weights|checkpoints?)\b", re.I)
CODE_STMT = {"in_supplementary_material": (_CODE, re.compile(r"(?:provided|included|available|released|attached|found|submitted|is|are)\s+(?:in|as|with|via)\s+(?:the\s+|our\s+)?(?:submitted\s+)?supplementa", re.I)),
             "on_request": (_ASSET, re.compile(r"\b(?:up)?on\s+(?:reasonable\s+)?request\b", re.I)),
             "promised_for_later": (_ASSET, re.compile(r"will\s+be\s+(?:made\s+)?(?:publicly\s+)?(?:available|released|open[- ]sourced)|will\s+release|(?:up)?on\s+(?:acceptance|publication)\b|to\s+be\s+released", re.I)),
             "on_openreview": (_CODE, re.compile(r"openreview", re.I))}

def parse_one(job):
    aid, title, claim_source = job
    out = {"arxiv_id": aid}
    try:
        import fitz
        doc = fitz.open(PDFS / f"{aid}.pdf")
        n = doc.page_count; out["total_pages"] = n
        ptext, plinks = [], []
        for pg in doc:
            ptext.append(pg.get_text("text"))
            ls = []
            for l in pg.get_links():
                if l.get("uri"):
                    r = l["from"]
                    try: ctx = pg.get_text("text", clip=fitz.Rect(r.x0 - 260, r.y0 - 26, r.x1 + 120, r.y1 + 8))
                    except Exception: ctx = ""
                    ls.append((l["uri"], " ".join(ctx.split())[:420]))
            plinks.append(ls)
        doc.close()
        # --- structure: headings, references, appendix, checklist
        heads, last_top, ref_page, ref_line = [], 0, None, None       # heads: (page_idx, line_idx, top, level, title)
        for pi, t in enumerate(ptext):
            lines = t.split("\n")
            for li, l in enumerate(lines):
                s = l.strip()
                if ref_page is None and pi >= 2 and re.fullmatch(r"(References|Bibliography|REFERENCES|References and Notes)", s):
                    ref_page, ref_line = pi, li
                if ref_page is not None: continue
                m = HEAD_RX.match(s)
                if not m:
                    m2 = NUM_ONLY.match(s)
                    if m2 and li + 1 < len(lines):
                        nxt = lines[li + 1].strip()
                        if re.match(r"^[A-Z][A-Za-z]", nxt) and len(nxt) <= 80 and not nxt.endswith("."):
                            m = HEAD_RX.match(f"{s} {nxt}")
                if not m: continue
                top, sub, ttl = int(m.group(1)), m.group(2), m.group(3).strip()
                if ttl.endswith((".", ",")) or nwords(ttl) > 12 or nwords(ttl) == 0: continue
                if not sub and top == 1 and re.match(r"introduction", ttl, re.I) and last_top != 0: heads, last_top = [], 0   # affiliations "1 Univ..." were mistaken for headings
                if not sub and top == last_top + 1: last_top = top; heads.append((pi, li, top, 1, ttl))
                elif sub and top == last_top and top > 0: heads.append((pi, li, top, 1 + sub.count("."), ttl))
        app_page = chk_page = None
        if ref_page is not None:
            for pi in range(ref_page, n):
                lines = ptext[pi].split("\n")
                for li, l in enumerate(lines):
                    if pi == ref_page and li <= ref_line: continue
                    s = l.strip()
                    if chk_page is None and re.search(r"NeurIPS Paper Checklist|^Paper Checklist$", s): chk_page = pi
                    if app_page is not None or chk_page is not None: continue
                    nxt = lines[li + 1].strip() if li + 1 < len(lines) else ""
                    if re.match(r"^(Appendix|Appendices|APPENDIX|Supplementary Materials?|Supplemental Materials?|Technical Appendi(x|ces).*|Contents of (the )?Appendix|Appendix [A-Z]\b.*|A\.?\s{1,3}[A-Z][a-z]+(\s[A-Za-z:,&-]+){0,9})$", s) \
                            or (re.fullmatch(r"A\.?", s) and re.match(r"^[A-Z][a-z]{2,}", nxt) and len(nxt) <= 90 and not nxt.endswith(".")):
                        app_page = pi
        out.update(main_body_pages=(ref_page + 1) if ref_page is not None else None,
                   references_start_page=(ref_page + 1) if ref_page is not None else None,
                   appendix_start_page=(app_page + 1) if app_page is not None else None,
                   checklist_start_page=(chk_page + 1) if chk_page is not None else None)
        end_app = chk_page if (chk_page is not None and app_page is not None and chk_page > app_page) else n
        out["appendix_pages"] = (end_app - app_page) if app_page is not None else None
        out["sections_detected"] = [h[4] for h in heads if h[3] == 1]
        # --- claim page
        m = re.search(r"page\s+(\d+)", claim_source or "", re.I)
        cp = int(m.group(1)) if m else None
        out["claim_page"] = cp
        # does the stated page really carry the stated table / figure? (the official list gives "Table 2, page 6"; arXiv versions can differ)
        lm = re.search(r"\b(Table|Figure|Fig\.?)\s*([A-Z]?\.?\d+)", claim_source or "")
        if not (lm and cp): out["claim_location_check"] = "not checked (the source names no table/figure or no page)"
        else:
            lrx = re.compile(("Table" if lm.group(1) == "Table" else r"Fig(?:ure|\.)") + r"\s*" + re.escape(lm.group(2)) + r"(?!\d)", re.I)
            has = lambda i: 0 <= i < n and bool(lrx.search(ptext[i]))
            out["claim_location_check"] = ("stated page is beyond the end of this PDF" if cp > n else "found on the stated page" if has(cp - 1) else
                                           "found one page away from the stated page" if has(cp - 2) or has(cp) else
                                           "found on a different page of this PDF" if any(has(i) for i in range(n)) else "label not found in this PDF's text")
        if cp is not None:
            if app_page is not None: out["claim_in_appendix"], out["claim_in_appendix_basis"] = cp >= app_page + 1, "appendix_start_detected"
            elif ref_page is not None: out["claim_in_appendix"], out["claim_in_appendix_basis"] = cp > ref_page + 1, "after_references_start"
            else: out["claim_in_appendix"], out["claim_in_appendix_basis"] = cp > 9, "page_gt_9_rule"
        else: out["claim_in_appendix"], out["claim_in_appendix_basis"] = None, "no_page_in_claim_source"
        # --- scoped reading text: abstract + method sections + setup of experiments (+ claim page if outside body)
        body_end = ref_page if ref_page is not None else min(n - 1, 9)
        def span(p0, l0, p1, l1):
            chunks = []
            for pi in range(p0, min(p1, n - 1) + 1):
                ls = ptext[pi].split("\n")
                chunks.append("\n".join(ls[(l0 if pi == p0 else 0):(l1 if pi == p1 else len(ls))]))
            return "\n".join(chunks)
        tops = [h for h in heads if h[3] == 1]
        body_stop = (ref_page, ref_line) if ref_page is not None else (body_end, 10 ** 6)
        main_text = span(0, 0, *body_stop)
        out["main_body_words"] = nwords(main_text); out["total_words"] = sum(nwords(t) for t in ptext)
        scoped, basis = None, None
        if len(tops) >= 3:
            p0 = ptext[0].split("\n")
            ai = next((i for i, l in enumerate(p0) if l.strip().lower() == "abstract"), None)
            abstract = span(0, ai + 1 if ai is not None else 0, tops[0][0], tops[0][1])
            if ai is None: abstract = " ".join(abstract.split()[-300:])
            parts, kinds = {"abstract": abstract, "method": "", "setup": ""}, []
            for i, h in enumerate(tops):
                stop = (tops[i + 1][0], tops[i + 1][1]) if i + 1 < len(tops) else body_stop
                kind = next((k for k, rxp in SEC_KIND if rxp.search(h[4])), "method")
                kinds.append(kind)
                if kind == "method": parts["method"] += "\n" + span(h[0], h[1], *stop)
                elif kind == "experiments":
                    subs = [x for x in heads if x[3] >= 2 and x[2] == h[2]]
                    got = False
                    for j, sh in enumerate(subs):
                        if SETUP_RX.search(sh[4]) and sh[3] == 2:
                            nxt = next(((x[0], x[1]) for x in subs[j + 1:] if x[3] == 2), stop)
                            parts["setup"] += "\n" + span(sh[0], sh[1], *nxt); got = True
                    if not got and not parts["setup"]:
                        parts["setup"] += "\n" + " ".join(span(h[0], h[1], *stop).split()[:350])
            extra = ""
            if cp is not None and 1 <= cp <= n and cp - 1 > body_end: extra = ptext[cp - 1]
            scoped = "\n".join([parts["abstract"], parts["method"], parts["setup"], extra])
            out["scope_words"] = {k: nwords(v) for k, v in parts.items()} | {"claim_page_outside_body": nwords(extra)}
            out["section_kinds"] = kinds; basis = "sections"
        out["scoped_words"] = nwords(scoped) if scoped else None
        out["eq_density_raw"] = eq_density_raw(scoped) if scoped else eq_density_raw(main_text)
        out["reading_basis"] = basis or "sections_not_detected"
        out["theorem_count_main"] = len(THM_RX.findall(main_text))
        # what the authors say about where their code is (one quoted sentence per kind; the checklist boilerplate is skipped)
        body = " ".join(" ".join(t.split()) for t in (ptext[:chk_page] if chk_page is not None else ptext)); cs = {}
        for sent in re.split(r"(?<=[.!?])\s+", body):
            if not 25 <= len(sent) <= 500: continue
            for kind, (need, rx_) in CODE_STMT.items():
                if kind not in cs and rx_.search(sent) and need.search(sent): cs[kind] = sent[:260]
        out["code_statements"] = cs
        # reference count: highest "[n]" label in the reference list if it is numbered, otherwise the number of lines ending in a year
        out["reference_count"], out["reference_count_basis"] = None, None
        if ref_page is not None:
            stop = min(x for x in (app_page, chk_page, n) if x is not None and x >= ref_page)
            ref_text = "\n".join(ptext[ref_page].split("\n")[ref_line + 1:] + ptext[ref_page + 1:max(stop, ref_page + 1)] + ([ptext[stop]] if stop < n and stop > ref_page else []))
            nums = [int(x) for x in re.findall(r"(?m)^\s*\[(\d{1,3})\]", ref_text)]
            if len(nums) >= 5: out["reference_count"], out["reference_count_basis"] = max(nums), "numbered list"
            else:
                yrs = len(re.findall(r"(?m)\b(?:19|20)\d{2}[a-z]?\.?\s*$", ref_text))
                if yrs >= 5: out["reference_count"], out["reference_count_basis"] = yrs, "lines ending in a year (approximate)"
        app_text = "\n".join(ptext[app_page:end_app]) if app_page is not None else ("\n".join(ptext[ref_page + 1:]) if ref_page is not None else "")
        out["appendix_evidence"] = {k: len(re.findall(p, app_text, re.I)) for k, p in
                                    {"hyperparameter": r"hyper-?\s?parameters?", "learning_rate": r"learning rate", "batch_size": r"batch size",
                                     "prompt_template": r"prompt template|system prompt|prompts? used|full prompt", "seeds": r"random seeds?|\bseeds?\b"}.items()}
        full = "\n".join(ptext)
        gm = []
        for m in GPU_RX.finditer(full):
            s = " ".join(full[max(0, m.start() - 60):m.end() + 40].split())
            if s not in gm: gm.append(s)
            if len(gm) >= 8: break
        out["compute_mentions"] = gm
        # --- link candidates
        keys = title_keys(title)
        def region(pi, pos_line=None):
            if chk_page is not None and pi >= chk_page: return "checklist"
            if app_page is not None and pi >= app_page and not (pi == ref_page and app_page == ref_page): return "appendix"
            if ref_page is not None and (pi > ref_page or (pi == ref_page and (pos_line is None or pos_line > ref_line))):
                return "references" if app_page is None or pi < app_page else "appendix"
            return "body"
        cands = collections.OrderedDict()
        def add(kind, key, url, pi, src, ctx, reg):
            c = cands.setdefault((kind, key.lower()), {"kind": kind, "url": url, "pages": [], "sources": [], "regions": [], "contexts": []})
            if pi + 1 not in c["pages"]: c["pages"].append(pi + 1)
            if src not in c["sources"]: c["sources"].append(src)
            if reg not in c["regions"]: c["regions"].append(reg)
            if ctx and len(c["contexts"]) < 3 and ctx not in c["contexts"]: c["contexts"].append(ctx)
        def scan(s, pi, src, ctx_fn, reg_fn):
            for m in GH_RX.finditer(s):
                owner, repo = m.group(1), re.sub(r"(\.git|[.,;:)\]]+)$", "", m.group(2))
                repo = re.sub(r"[.,;:)\]]+$", "", repo)
                if owner.lower() in GH_SKIP_OWNERS or not repo or repo.lower() in {"issues", "pull", "blob", "tree"}: continue
                add("github", f"{owner}/{repo}", f"https://github.com/{owner}/{repo}", pi, src, ctx_fn(m), reg_fn(m))
            for m in HF_RX.finditer(s):
                path = re.sub(r"[.,;:)\]]+$", "", m.group(1))
                if path.lower().split("/")[0] in {"docs", "blog", "papers", "models", "datasets", "spaces", "join", "pricing", "learn", "tasks"} and "/" not in path.strip("/"): continue
                if path.lower().split("/")[0] in {"docs", "blog", "papers", "learn", "tasks"}: continue
                add("hf", path, f"https://huggingface.co/{path}", pi, src, ctx_fn(m), reg_fn(m))
            for m in IO_RX.finditer(s):
                add("project_page", m.group(0).lower().rstrip("/."), "https://" + re.sub(r"^https?://", "", m.group(0)).rstrip(".,;)"), pi, src, ctx_fn(m), reg_fn(m))
            for m in ANON_RX.finditer(s):
                add("anonymous4open", m.group(0), "https://" + m.group(0), pi, src, ctx_fn(m), reg_fn(m))
        for pi in range(n):
            t = ptext[pi]
            starts = [0]
            for l in t.split("\n"): starts.append(starts[-1] + len(l) + 1)
            import bisect
            def reg_text(m, pi=pi, starts=starts): return region(pi, bisect.bisect_right(starts, m.start()) - 1)
            def ctx_text(m, t=t): return " ".join(t[max(0, m.start() - 220):m.end() + 60].split())
            scan(t, pi, "text", ctx_text, reg_text)
            # URLs wrapped across a line break ("github.com/foo/bar-\nbaz") are only recoverable from annotations
            for uri, ctx in plinks[pi]:
                def reg_annot(m, pi=pi, uri=uri, t=t, starts=starts):
                    tail = uri.rstrip("/").split("/")[-1][:25]
                    pos = t.find(tail) if tail else -1
                    return region(pi, bisect.bisect_right(starts, pos) - 1 if pos >= 0 else None)
                scan(uri, pi, "annot", lambda m, ctx=ctx: ctx, reg_annot)
        cl = []
        for (kind, key), c in cands.items():
            sc, why = 0, []
            ctx = " ".join(c["contexts"])
            if 1 in c["pages"]: sc += 6; why.append("page1")
            if OWN_RX.search(ctx): sc += 4; why.append("own-code-context")
            name = re.sub(r"[^a-z0-9]", "", key.split("/")[-1])
            if any((k in name) if len(k) >= 4 else (k == name) for k in keys): sc += 3; why.append("title-key-in-name")
            if "checklist" in c["regions"]: sc += 2; why.append("checklist")
            if "appendix" in c["regions"]: sc += 1; why.append("appendix")
            if c["regions"] == ["references"]: sc -= 5; why.append("references-only")
            if kind == "github" and key.split("/")[0] in INFRA_OWNERS and "title-key-in-name" not in why: sc -= 3; why.append("infra-owner")
            if len(c["pages"]) >= 3: sc += 1; why.append("3+pages")
            c.update(key=key, own_score=sc, own_why=why); cl.append(c)
        out["link_candidates"] = cl
        gh = sorted([c for c in cl if c["kind"] == "github"], key=lambda c: (-c["own_score"], c["pages"][0]))
        if gh and gh[0]["own_score"] >= 4:
            out["github_primary"] = gh[0]["url"]; out["github_primary_confidence"] = "high" if gh[0]["own_score"] >= 6 else "medium"
            out["github_primary_why"] = gh[0]["own_why"]
            out["github_primary_ambiguous"] = len(gh) > 1 and gh[1]["own_score"] >= gh[0]["own_score"] - 1 and gh[1]["own_score"] >= 4
        else:
            out["github_primary"] = None; out["github_primary_confidence"] = None
        out["hf_own_links"] = [c["url"] for c in cl if c["kind"] == "hf" and c["own_score"] >= 4]
        out["project_pages"] = [c["url"] for c in cl if c["kind"] == "project_page" and c["own_score"] >= 4]
        out["anonymous_links"] = [c["url"] for c in cl if c["kind"] == "anonymous4open"]
        out["parse_status"] = "ok"
    except Exception as e:
        out["parse_status"] = f"error:{type(e).__name__}:{str(e)[:120]}"
    jdump(AUD / "pdf" / f"{aid}.json", out)
    return aid, out["parse_status"]

def stage_parse(args):
    rows = papers()
    jobs = [(r["arxiv_id"], r["title"], r["claim_source"]) for r in rows if (PDFS / f"{r['arxiv_id']}.pdf").exists()
            and (args.force or not (AUD / "pdf" / f"{r['arxiv_id']}.json").exists())]
    if args.only: jobs = [j for j in jobs if j[0] in args.only]
    print(f"parse: {len(jobs)} PDFs to parse")
    bad = []
    # do not set max_tasks_per_child here: with it the pool can deadlock part-way (workers gone, parent waiting forever)
    with ProcessPoolExecutor(max_workers=5) as ex:
        for k, (aid, st) in enumerate(ex.map(parse_one, jobs, chunksize=2)):
            if st != "ok": bad.append((aid, st)); print("  parse error", aid, st)
            if k % 100 == 0: print(f"  [{k + 1}/{len(jobs)}]")
    done = len(list((AUD / "pdf").glob("*.json")))
    print(f"parse done: {done} parsed files on disk; errors this run: {bad}")


# ---------------------------------------------------------------- stage: pages (project-page -> repo link)
def stage_pages(args):
    import httpx
    out = jload(AUD / "project_pages.json", {})
    todo = []
    for f in sorted((AUD / "pdf").glob("*.json")):
        a = jload(f)
        if a.get("parse_status") == "ok" and not a.get("github_primary") and a.get("project_pages") and a["arxiv_id"] not in out:
            todo.append((a["arxiv_id"], a["project_pages"][0]))
    print(f"pages: {len(todo)} project pages to fetch")
    with httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent": UA}) as c:
        for aid, url in todo:
            try:
                r = c.get(url)
                owner = re.match(r"https://([a-z0-9-]+)\.github\.io", url, re.I).group(1).lower()
                found = []
                for m in GH_RX.finditer(r.text if r.status_code == 200 else ""):
                    o, rp = m.group(1), re.sub(r"(\.git|[.,;:)\]]+)$", "", m.group(2))
                    u = f"https://github.com/{o}/{rp}"
                    if o.lower() not in GH_SKIP_OWNERS and u not in found: found.append(u)
                # template credits (nerfies etc.) are common on project pages -> only trust same-owner links or the page's own repo
                slug = url.rstrip("/").split("/")[-1].lower()
                own = [u for u in found if u.split("/")[3].lower() == owner and not u.lower().endswith(f"{owner}.github.io") and u.split("/")[4].lower() != slug + ".github.io"]
                out[aid] = {"page": url, "http": r.status_code, "github_links": found, "same_owner_links": own}
            except Exception as e:
                out[aid] = {"page": url, "error": type(e).__name__}
            print(" ", aid, url, out[aid].get("same_owner_links"), out[aid].get("error", ""))
            jdump(AUD / "project_pages.json", out); time.sleep(1.0)
    print(f"pages done: {sum(bool(v.get('same_owner_links')) for v in out.values())}/{len(out)} pages yielded a same-owner repo link")


def primary_repo(a, pages):
    """-> (url, source, confidence) using PDF first, then the paper's own project page."""
    if a.get("github_primary"): return a["github_primary"], "pdf", a.get("github_primary_confidence")
    pp = pages.get(a["arxiv_id"], {})
    if pp.get("same_owner_links"):
        ls = pp["same_owner_links"]
        return ls[0], "project_page", "medium" if len(ls) == 1 else "low"
    return None, None, None

def all_audits():
    return {f.stem: jload(f) for f in sorted((AUD / "pdf").glob("*.json"))}


# ---------------------------------------------------------------- stage: head
def stage_head(args):
    import httpx
    pages = jload(AUD / "project_pages.json", {}); cache = jload(AUD / "github_head.json", {})
    urls = sorted({u for a in all_audits().values() for u in [primary_repo(a, pages)[0]] if u and (u not in cache or not isinstance(cache[u], int) or cache[u] == 429)})
    print(f"head: {len(urls)} GitHub URLs to check ({len(cache)} cached)")
    async def run():
        sem = asyncio.Semaphore(6)
        async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers={"User-Agent": UA}) as c:
            async def one(u):
                async with sem:
                    for attempt in range(4):
                        try:
                            r = await c.head(u); cache[u] = r.status_code
                            if r.status_code == 429: await asyncio.sleep(30 * (attempt + 1)); continue
                            if str(r.url).rstrip("/").lower() != u.lower(): cache[u + "#final"] = str(r.url)
                            break
                        except Exception as e:
                            cache[u] = type(e).__name__; await asyncio.sleep(3)
                    await asyncio.sleep(0.3)
            await asyncio.gather(*(one(u) for u in urls))
    asyncio.run(run()); jdump(AUD / "github_head.json", cache)
    vals = [v for k, v in cache.items() if not k.endswith("#final")]
    print("head done:", dict(collections.Counter(map(str, vals))))


# ---------------------------------------------------------------- stage: clone (never executes repo content)
GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1", GIT_ASKPASS="/bin/true", GIT_CONFIG_NOSYSTEM="1")
CHECKOUT_RX = re.compile(r"(^|/)(readme[^/]*|licen[sc]e[^/]*|copying[^/]*|requirements[^/]*\.txt|pyproject\.toml|setup\.(py|cfg)|environment[^/]*\.ya?ml|dockerfile[^/]*|uv\.lock|makefile|[^/]+\.(py|sh|md|rst|toml|cfg|ya?ml|ipynb|txt))$", re.I)

def git(cwd, *a, timeout=120, inp=None):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never", *a], cwd=cwd, env=GIT_ENV, capture_output=True, text=True, timeout=timeout, input=inp)

def clone_one(job):
    aid, url = job
    dest = REPOS / aid
    res = {"arxiv_id": aid, "url": url}
    try:
        if dest.exists(): shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = git(ROOT, "clone", "--depth", "1", "--single-branch", "--no-tags", "--filter=blob:limit=2m", "--no-checkout", url + ".git", str(dest), timeout=240)
        if r.returncode != 0:
            res.update(status="clone_failed", error=" ".join(r.stderr.split())[-240:]); shutil.rmtree(dest, ignore_errors=True); return res
        res["head_sha"] = git(dest, "rev-parse", "HEAD").stdout.strip()
        res["head_date"] = git(dest, "log", "-1", "--format=%cI").stdout.strip()
        git(dest, "remote", "remove", "origin")      # no lazy blob fetches after this point
        tree = [l.split("\t", 1) for l in git(dest, "ls-tree", "-r", "HEAD").stdout.splitlines() if "\t" in l]
        missing = {l[1:].split()[0] for l in git(dest, "rev-list", "--objects", "--missing=print", "HEAD").stdout.splitlines() if l.startswith("?")}
        files = [(meta.split()[2], path, meta.split()[0]) for meta, path in tree if meta.split()[1] == "blob"]
        res["n_files"] = len(files); res["n_large_blobs_skipped"] = sum(o in missing for o, _, _ in files)
        (dest / ".audit_filelist.txt").write_text("\n".join(("LARGE>2M\t" if o in missing else "\t") + p for o, p, _ in files))
        want = [p for o, p, mode in files if o not in missing and mode != "120000" and CHECKOUT_RX.search(p) and not p.startswith('"')][:6000]
        if want:
            r = git(dest, "checkout", "HEAD", "--pathspec-from-file=-", "--pathspec-file-nul", timeout=180, inp="\0".join(":(literal)" + p for p in want))
            if r.returncode != 0: res["checkout_warning"] = " ".join(r.stderr.split())[-200:]
        res["n_checked_out"] = len(want)
        du = subprocess.run(["du", "-sm", str(dest)], capture_output=True, text=True).stdout.split()
        res["size_mb"] = int(du[0]) if du else None
        res["status"] = "cloned"
    except subprocess.TimeoutExpired:
        res.update(status="clone_failed", error="timeout"); shutil.rmtree(dest, ignore_errors=True)
    except Exception as e:
        res.update(status="clone_failed", error=f"{type(e).__name__}:{str(e)[:160]}"); shutil.rmtree(dest, ignore_errors=True)
    return res

def stage_clone(args):
    pages = jload(AUD / "project_pages.json", {}); head = jload(AUD / "github_head.json", {}); st = jload(AUD / "clone_status.json", {})
    auds = all_audits(); order = [i for i in priority_ids(papers()) if i in auds]
    if args.only: order = [i for i in order if i in args.only]
    jobs = []
    for aid in order:
        url, _, _ = primary_repo(auds[aid], pages)
        if not url or head.get(url) == 404: continue
        url = head.get(url + "#final", url).rstrip("/")
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url): continue
        if st.get(aid, {}).get("url") == url and (st[aid]["status"] == "cloned" and (REPOS / aid).exists() or st[aid]["status"] == "clone_failed" and not args.force): continue
        jobs.append((aid, url))
    if args.limit: jobs = jobs[:args.limit]
    print(f"clone: {len(jobs)} repos to clone ({sum(v['status'] == 'cloned' for v in st.values())} already cloned)")
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, res in enumerate(ex.map(clone_one, jobs)):
            st[res["arxiv_id"]] = res
            if res["status"] != "cloned": print("  FAILED", res["arxiv_id"], res["url"], res.get("error"))
            if k % 25 == 0: jdump(AUD / "clone_status.json", st); print(f"  [{k + 1}/{len(jobs)}] {res['arxiv_id']} {res['status']} {res.get('size_mb')}MB")
    jdump(AUD / "clone_status.json", st)
    print("clone done:", dict(collections.Counter(v["status"] for v in st.values())))


# ---------------------------------------------------------------- stage: audit (static, read-only)
# scope: "readme" = README*/docs md only; "code" = py/sh/setup/docker/yaml; "any" = both
HARD = [(k["name"], k["pattern"], k["scope"]) for k in CFG["keyword"]]       # raw keyword hits; classified into blocker / friction / ignored in audit_one
SOFT = [(k["name"], k["pattern"], k["scope"]) for k in CFG["friction"]]
BOOST = [(k["name"], k["pattern"], k["scope"]) for k in CFG["good_sign"]]
FACT = [(k["name"], k["pattern"], k["scope"]) for k in CFG.get("fact", [])]     # recorded, never scored
WT = CFG["weights"]
CLOSED_KEY = re.compile(r"OPENAI_API_KEY|ANTHROPIC_API_KEY")
JUDGE_RX = re.compile(r"judge|grader|evaluator|llm[-_ ]as[-_ ]a|optional|scor(er|ing)|annotat", re.I)
LOCAL_MODEL_RX = re.compile(r"from_pretrained\(|vllm|torch\.load|AutoModel|llama_cpp|ollama", re.I)

def read_text(p, cap=400_000):
    try:
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 2_500_000: return ""
        return p.read_bytes()[:cap].decode("utf-8", "replace")
    except Exception: return ""

def audit_one(aid, closed_hint):
    d = REPOS / aid
    res = {"arxiv_id": aid}
    listing = [l.split("\t", 1) for l in read_text(d / ".audit_filelist.txt", 5_000_000).splitlines() if "\t" in l]
    names = [p for _, p in listing]; low = [p.lower() for p in names]
    readme_files, code_files = [], []
    for p in sorted(d.rglob("*")):
        if ".git" in p.relative_to(d).parts or p.name == ".audit_filelist.txt" or p.is_symlink() or not p.is_file(): continue
        rel = str(p.relative_to(d)); s = p.suffix.lower()
        if s in (".md", ".rst") or p.name.lower().startswith("readme"): readme_files.append(rel)
        elif s in (".py", ".sh", ".toml", ".cfg", ".yaml", ".yml", ".ipynb", ".txt") or p.name.lower().startswith(("dockerfile", "makefile")): code_files.append(rel)
    # root README first, shallow files first, cap the amount read
    readme_files.sort(key=lambda r: (r.count("/"), not r.lower().startswith("readme"), r)); code_files.sort(key=lambda r: (r.count("/"), r))
    readme_files, code_files = readme_files[:40], code_files[:1500]
    hits = collections.defaultdict(list)
    def grep(rel, scope):
        t = read_text(d / rel)
        if not t: return
        for group, pats in (("hard", HARD), ("soft", SOFT), ("boost", BOOST), ("fact", FACT)):
            for name, pat, sc in pats:
                if sc != "any" and sc != scope: continue
                if len(hits[(group, name)]) >= 3: continue
                m = re.search(pat, t, re.I | re.M)
                if m:
                    ln = t.count("\n", 0, m.start()) + 1
                    line = t[t.rfind("\n", 0, m.start()) + 1:m.end() + 80].split("\n")[0]
                    hits[(group, name)].append(f"{rel}:{ln}: {' '.join(line.split())[:160]}")
        if scope == "code" and len(hits[("closed", "api_key")]) < 6:
            m = CLOSED_KEY.search(t)
            if m: hits[("closed", "api_key")].append(f"{rel}:{t.count(chr(10), 0, m.start()) + 1}")
        if scope == "code" and rel.endswith(".py") and LOCAL_MODEL_RX.search(t): hits[("meta", "local_model_code")].append(rel) if len(hits[("meta", "local_model_code")]) < 3 else None
    for rel in readme_files: grep(rel, "readme")
    budget = 60_000_000
    for rel in code_files:
        if budget <= 0: break
        try: budget -= (d / rel).stat().st_size
        except Exception: pass
        grep(rel, "code")
    root_readme = next((r for r in readme_files if "/" not in r and r.lower().startswith("readme")), None)
    rd = read_text(d / root_readme) if root_readme else ""
    n_py = sum(p.endswith(".py") for p in low); n_ipynb = sum(p.endswith(".ipynb") for p in low)
    n_code = n_py + n_ipynb + sum(p.endswith((".sh", ".jl", ".r", ".cpp", ".c", ".cc", ".rs", ".go", ".js", ".ts", ".java", ".m", ".lean")) for p in low)
    base = lambda p: p.rsplit("/", 1)[-1]
    def first_file(rx):
        return next((p for p in names if re.fullmatch(rx, base(p.lower()))), None)
    file_boost = {"uv.lock": first_file(r"uv\.lock"), "Dockerfile": first_file(r"dockerfile.*|.*\.dockerfile"),
                  "eval.py": first_file(r"(.*_)?(eval|evaluate|evaluation)(_.*)?\.py"), "reproduce.sh": first_file(r"(reproduc\w*|repro|run_all\w*)\.sh")}
    dep_spec = [b for b in ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml", "environment.yaml", "uv.lock", "poetry.lock", "pixi.toml", "conda.yaml")
                if any(base(p) == b or (b == "requirements.txt" and re.fullmatch(r"requirements.*\.txt", base(p))) for p in low)]
    facts = {n: hits[("fact", n)][:1] for n, _, _ in FACT if hits[("fact", n)]}
    lic_file = next((p for p in names if "/" not in p and re.match(r"(licen[sc]e|copying)", p.lower())), None); lic = None
    if lic_file:
        lt = read_text(d / lic_file)[:1500]
        if not lt:   # licence files have no extension, so the selective checkout skipped them; the blob is still in the local clone
            try: lt = git(d, "show", f"HEAD:{lic_file}", timeout=30).stdout[:1500]
            except Exception: lt = ""
        lic = next((nm for nm, rx_ in (("MIT", r"MIT License|Permission is hereby granted, free of charge"), ("Apache-2.0", r"Apache License"), ("BSD", r"Redistribution and use in source and binary forms"),
                                       ("AGPL", r"GNU AFFERO"), ("LGPL", r"GNU LESSER"), ("GPL", r"GNU GENERAL PUBLIC"), ("CC BY-NC", r"NonCommercial"), ("CC BY", r"Creative Commons"), ("MPL", r"Mozilla Public")) if re.search(rx_, lt, re.I)), "other / custom" if lt.strip() else "present, not readable")
    has_tests = any(re.match(r"(.*/)?tests?/", p) or re.fullmatch(r"(.*/)?test_[^/]+\.py", p) for p in low)
    hard = {n: hits[("hard", n)] for n, _, _ in HARD if hits[("hard", n)]}
    if n_code == 0: hard["no_code_files"] = [f"{len(names)} files, 0 code files (README-only / placeholder repo)"]
    closed = hits[("closed", "api_key")]
    closed_class = None
    if closed:
        ctx_judge = bool(JUDGE_RX.search(" ".join(closed))) or bool(re.search(r"(judge|grader|evaluator|optional)[^\n]{0,120}(openai|gpt|claude|anthropic)|(openai|gpt|claude|anthropic)[^\n]{0,120}(judge|grader|evaluator|optional)", rd, re.I))
        has_local = bool(hits[("meta", "local_model_code")])
        if closed_hint and not has_local and not ctx_judge:
            closed_class = "target_model_closed_api"; hard["closed_api_target"] = closed[:3]
        else: closed_class = "referenced_optional_or_judge" if (ctx_judge or has_local) else "referenced_unclear"
    soft = {n: hits[("soft", n)] for n, _, _ in SOFT if hits[("soft", n)]}
    # A keyword match alone is weak evidence: "support for X coming soon" in a vendored sub-README, "TODO:" in docs or a
    # Google Drive link for one optional file do not stop anyone. So a match sets the score to 0 only when it says the code
    # itself is missing or a VM is required; everything else is weighted friction, or is recorded under ignored_signals.
    depth = lambda hit: hit.split(":", 1)[0].count("/")
    kw, hard, weights, ignored = hard, {k: v for k, v in hard.items() if k in ("no_code_files", "closed_api_target")}, {}, {}
    cs_lines = [" ".join(l.split())[:160] for l in rd.splitlines() if re.search(r"coming\s+soon|to\s+be\s+released|will\s+be\s+released", l, re.I)]
    cs_code = [l for l in cs_lines if re.search(r"\b(code|implementation)\b[^.]{0,40}(coming\s+soon|will\s+be\s+released|to\s+be\s+released)|(coming\s+soon|to\s+be\s+released)[^.]{0,20}\bcode\b", l, re.I)
               and not re.search(r"\b(training|train|data|collection|inference|eval\w*)\s+(code|scripts?)", l, re.I)]
    cs_part = [l for l in cs_lines if l not in cs_code and re.search(r"code|script|weights?|checkpoints?|models?|train|dataset|data\b|benchmark", l, re.I)
               and not re.search(r"paper|citation|arxiv|support for|page", l, re.I)]
    if cs_code and n_py < 10: hard["code_not_released"] = cs_code[:2]
    elif cs_code or cs_part: soft["partial_release_pending"] = (cs_code + cs_part)[:2]; weights["partial_release_pending"] = WT["partial_release_pending"]
    if "coming soon" in kw and "code_not_released" not in hard and "partial_release_pending" not in soft: ignored["coming soon"] = kw["coming soon"][:1]
    if "todo:" in kw: ignored["todo:"] = kw["todo:"][:1]
    gated = [h for k in ("request access", "fill out the form", "sign the agreement") for h in kw.get(k, [])]
    if any(depth(h) <= 1 for h in gated): soft["gated_access"] = [h for h in gated if depth(h) <= 1][:2]; weights["gated_access"] = WT["gated_access"]
    elif gated: ignored["gated_access(nested file)"] = gated[:1]
    if "drive.google.com" in kw: soft["google_drive_download"] = kw["drive.google.com"][:1]; weights["google_drive_download"] = WT["google_drive_download"]
    if "pan.baidu.com" in kw: soft["baidu_pan_download"] = kw["pan.baidu.com"][:1]; weights["baidu_pan_download"] = WT["baidu_pan_with_drive_alternative"] if "drive.google.com" in kw else WT["baidu_pan_only"]
    native = kw.get("CUDAExtension", []) + kw.get("cpp_extension", [])
    if native: soft["native_build"] = native[:2]; weights["native_build"] = WT["native_build_top_level"] if any(depth(h) <= 1 for h in native) else WT["native_build_nested"]
    vm = kw.get("/dev/kvm", []) + kw.get("qemu", [])
    if re.search(r"/dev/kvm|\bqemu", rd, re.I): hard["vm_required"] = vm[:2] or ["root README mentions qemu / kvm"]   # only the repo's own README counts: the word also turns up inside base64 blobs in notebooks
    elif vm: soft["vm_mentioned"] = vm[:1]; weights["vm_mentioned"] = WT["vm_mentioned"]
    boost = {n: hits[("boost", n)][:1] for n, _, _ in BOOST if hits[("boost", n)]} | {k: [v] for k, v in file_boost.items() if v}
    friction = sum(weights.get(k, 1) for k in soft)
    raw = WT["base"] + len(boost) - friction - (1 if closed_class and closed_class != "target_model_closed_api" else 0) - (1 if not dep_spec else 0) - (1 if len(rd) < 500 else 0)
    raw = max(0, min(10, raw))
    res.update(facts=facts, licence=lic, licence_file=lic_file, has_tests=has_tests, n_files=len(names), n_py=n_py, n_ipynb=n_ipynb, readme_chars=len(rd), dep_spec=dep_spec, n_large_files=sum(t.startswith("LARGE") for t, _ in listing),
               hard_blockers=hard, soft_friction=soft, friction_weights=weights, ignored_signals=ignored, keyword_hits=kw, boosters=boost, closed_api=closed_class, closed_api_refs=closed[:3],
               turnkey_raw_before_blockers=raw, turnkey_score=0 if hard else raw,
               turnkey_formula="base 3 +1/good sign(7 max) -weighted friction (gated access 2, native build 1-2, baidu 1-2, others 1) -1 closed-API ref -1 no dep spec -1 README<500 chars; "
                               "0 only if: no code files, code announced but not released, closed-API target model, or VM required")
    return res

def stage_scan(args):
    st = jload(AUD / "clone_status.json", {}); rows = {r["arxiv_id"]: r for r in papers()}
    ids = [a for a, v in st.items() if v["status"] == "cloned" and (REPOS / a).exists()]
    if args.only: ids = [i for i in ids if i in args.only]
    print(f"scan: {len(ids)} cloned repos")
    out = jload(AUD / "repo_audit.json", {})
    for k, aid in enumerate(ids):
        try: out[aid] = audit_one(aid, rows[aid]["closed_api_hint"])
        except Exception as e: out[aid] = {"arxiv_id": aid, "audit_error": f"{type(e).__name__}:{str(e)[:160]}"}; print("  scan error", aid, out[aid]["audit_error"])
        if k % 100 == 0: print(f"  [{k + 1}/{len(ids)}]"); jdump(AUD / "repo_audit.json", out)
    jdump(AUD / "repo_audit.json", out)
    ok = [v for v in out.values() if "turnkey_score" in v]
    print(f"scan done: {len(ok)} audited; hard-blocked: {sum(bool(v['hard_blockers']) for v in ok)}; turnkey>=7: {sum(v['turnkey_score'] >= 7 for v in ok)}")
    print("hard blocker counts:", dict(collections.Counter(k for v in ok for k in v["hard_blockers"])))


# ---------------------------------------------------------------- stage: merge
BENCH_N = re.compile(r"\b(\d{1,3}|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:\w+\s+){0,2}(benchmarks|datasets|tasks|suites|environments|domains|games)\b", re.I)
WORDNUM = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

def stage_merge(args):
    rows = papers(); auds = all_audits(); pages = jload(AUD / "project_pages.json", {}); head = jload(AUD / "github_head.json", {})
    clones = jload(AUD / "clone_status.json", {}); raud = jload(AUD / "repo_audit.json", {}); dl = jload(AUD / "download_status.json", {})
    # fallback reading scope for papers whose sections were not detected: median scoped/main-body word ratio of the detected ones
    ratios = sorted(a["scoped_words"] / a["main_body_words"] for a in auds.values() if a.get("parse_status") == "ok" and a.get("scoped_words") and a.get("main_body_words"))
    med_ratio = ratios[len(ratios) // 2] if ratios else None
    eqs = sorted(a["eq_density_raw"] for a in auds.values() if a.get("eq_density_raw") is not None)
    eq_p95 = eqs[int(0.95 * (len(eqs) - 1))] if eqs else None
    print(f"merge: {len(auds)} parsed PDFs; median scoped/main ratio={med_ratio}; eq_density_raw p95={eq_p95}")
    res = []
    for r in rows:
        aid = r["arxiv_id"]; a = auds.get(aid)
        o = {"arxiv_id": aid, "pdf_status": "parsed" if a and a.get("parse_status") == "ok" else ("parse_error" if a else ("downloaded" if (PDFS / f"{aid}.pdf").exists() else "not_downloaded:" + str(dl.get(aid, {}).get("http") or dl.get(aid, {}).get("error") or "pending")))}
        for k in ("total_pages", "main_body_pages", "appendix_start_page", "appendix_pages", "checklist_start_page", "claim_page", "claim_in_appendix", "claim_in_appendix_basis",
                  "claim_reading_mins", "reading_basis", "equation_density", "theorem_count_main", "appendix_repro_bonus", "appendix_repro_bonus_why", "readability_score", "compute_mentions",
                  "repo_url", "repo_link_source", "repo_link_confidence", "repo_link_why", "repo_status", "repo_head_sha", "repo_head_date", "github_candidates", "hf_links", "hf_own_links", "project_pages", "anonymous_links",
                  "turnkey_score", "turnkey_raw_before_blockers", "hard_blockers", "soft_friction", "boosters", "closed_api", "repo_n_py", "repo_dep_spec"):
            o[k] = None
        m = re.search(r"page\s+(\d+)", r.get("claim_source") or "", re.I)
        o["claim_page"] = int(m.group(1)) if m else None
        if o["pdf_status"] == "parsed":
            for k in ("total_pages", "main_body_pages", "appendix_start_page", "appendix_pages", "checklist_start_page", "claim_in_appendix", "claim_in_appendix_basis", "theorem_count_main", "compute_mentions", "hf_own_links", "project_pages", "anonymous_links", "reference_count", "reference_count_basis", "appendix_evidence", "claim_location_check", "code_statements"):
                o[k] = a.get(k)
            o["github_candidates"] = [{"url": c["url"], "pages": c["pages"], "own_score": c["own_score"], "why": c["own_why"]} for c in a["link_candidates"] if c["kind"] == "github"]
            o["hf_links"] = [{"url": c["url"], "pages": c["pages"], "own_score": c["own_score"]} for c in a["link_candidates"] if c["kind"] == "hf"]
            # reading time: words(abstract + method + setup of claim) / 200 wpm * (1 + 0.5 * equation_density)
            eq = min(1.0, a["eq_density_raw"] / eq_p95) if (a.get("eq_density_raw") is not None and eq_p95) else 0.0
            o["equation_density"] = round(eq, 3)
            if a.get("scoped_words"): words, o["reading_basis"] = a["scoped_words"], "sections(abstract+method+setup)"
            elif a.get("main_body_words") and med_ratio: words, o["reading_basis"] = a["main_body_words"] * med_ratio, f"fallback:main_body_words*{med_ratio:.2f}(corpus-median scope ratio; sections not detected)"
            else: words = None
            if words:
                o["claim_reading_mins"] = round(max(5.0, min(45.0, words / 200 * (1 + 0.5 * eq))), 1)
                # appendix bonus (0..2): +1 long paper with appendix (>18 pages), +1 explicit hyperparameter/prompt evidence in appendix, -1 claim aggregates 5+ benchmarks
                ev = a.get("appendix_evidence") or {}
                why, b = [], 0
                if (a["total_pages"] or 0) > 18 and (a.get("appendix_pages") or (a["total_pages"] - (a.get("main_body_pages") or a["total_pages"]))) > 0: b += 1; why.append(">18 pages with appendix")
                if ev.get("hyperparameter", 0) >= 2 or (ev.get("learning_rate", 0) and ev.get("batch_size", 0)) or ev.get("prompt_template", 0) >= 2: b += 1; why.append("appendix mentions hyperparameters/prompts")
                nb = max([WORDNUM.get(x.lower()) or int(x) for x, _ in BENCH_N.findall(f"{r['claim']} {r['metric']}")] or [0])
                if 5 <= nb <= 200: b -= 1; why.append(f"claim aggregates across {nb} benchmarks/tasks")
                o["appendix_repro_bonus"], o["appendix_repro_bonus_why"] = max(0, min(2, b)), why
                rs = 10 - (o["claim_reading_mins"] - 5) / 40 * 6
                tc = a.get("theorem_count_main") or 0
                rs -= 1.5 if tc >= 5 else (0.75 if tc >= 2 else 0)
                if o["claim_in_appendix"] and (o["claim_page"] or 0) > 20: rs -= 1
                rs += 0.5 * o["appendix_repro_bonus"]
                o["readability_score"] = round(max(0.0, min(10.0, rs)), 1)
            url, src, conf = primary_repo(a, pages)
            if url:
                o["repo_url"], o["repo_link_source"], o["repo_link_confidence"], o["repo_link_why"] = url, src, conf, a.get("github_primary_why")
                h = head.get(url); c = clones.get(aid)
                if c and c["status"] == "cloned" and c.get("url", "").lower().rstrip("/") in {url.lower(), str(head.get(url + "#final", "")).lower().rstrip("/")}:
                    o["repo_status"], o["repo_head_sha"], o["repo_head_date"] = "cloned", c.get("head_sha"), c.get("head_date")
                    ra = raud.get(aid)
                    if ra and "turnkey_score" in ra:
                        for k in ("turnkey_score", "turnkey_raw_before_blockers", "hard_blockers", "soft_friction", "boosters", "closed_api"): o[k] = ra[k]
                        o["repo_n_py"], o["repo_dep_spec"] = ra["n_py"], ra["dep_spec"]
                elif h == 404: o["repo_status"] = "dead"
                elif c and c["status"] == "clone_failed": o["repo_status"] = "clone_failed"
                else: o["repo_status"] = "link_found_not_cloned"
                o["github_http"] = h
            else:
                o["repo_status"] = "no_link_found"
        else:
            o["repo_status"] = "pdf_not_audited"
        res.append(o)
    assert len(res) == len(rows) == len({o["arxiv_id"] for o in res})
    jdump(DATA / "audit_results.json", res)
    cnt = collections.Counter(o["repo_status"] for o in res)
    stats = {"n_rows": len(res), "pdf_on_disk": sum((PDFS / f"{r['arxiv_id']}.pdf").exists() for r in rows), "pdf_parsed": sum(o["pdf_status"] == "parsed" for o in res),
             "reading_time_from_sections": sum(bool(o["reading_basis"]) and o["reading_basis"].startswith("sections") for o in res), "reading_time_fallback": sum(bool(o["reading_basis"]) and o["reading_basis"].startswith("fallback") for o in res),
             "quick_read_le_15": sum((o["claim_reading_mins"] or 99) <= 15 for o in res), "repo_link_found": sum(bool(o["repo_url"]) for o in res),
             "github_alive_200": sum(o.get("github_http") == 200 for o in res), "repo_dead_404": cnt["dead"], "repo_cloned": cnt["cloned"], "repo_clone_failed": cnt["clone_failed"],
             "repo_no_link": cnt["no_link_found"], "repo_audited": sum(o["turnkey_score"] is not None for o in res), "hard_blocked": sum(bool(o["hard_blockers"]) for o in res),
             "turnkey_ge_7": sum((o["turnkey_score"] or 0) >= 7 for o in res)}
    print("merge stats:", json.dumps(stats, indent=1)); print("repo_status:", dict(cnt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "parse", "pages", "head", "clone", "scan", "merge"])
    ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); args.only = set(filter(None, args.only.split(",")))
    logp = start_log("2_scan_" + args.stage)
    try:
        globals()[f"stage_{args.stage}"](args)
        print(f"===== stage {args.stage} finished OK {time.strftime('%H:%M:%S')} =====")
    except BaseException:
        traceback.print_exc(); print(f"===== stage {args.stage} FAILED ====="); fail_log(logp); raise
