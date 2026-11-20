"""Step 4: build index.html, the interactive paper map (one self-contained file: no CDN, no server, no tracking).

Reads data/papers_evidence.json, data/embeddings_search.npy, data/search_terms.json and data/search_vectors.npy (step 3).
The page ranks nothing by itself: the visitor sets weights, types interests, ticks constraints and stars papers, and everything
is computed in the browser. Settings live in the URL hash so a configuration can be shared as a link.
Usage: uv run python pipeline/4_build_map.py"""
import base64, json, os
os.environ.setdefault("PYTHONHASHSEED", "0")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
recs = json.loads((DATA / "papers_evidence.json").read_text())
# The organisers' text columns travel inside each record's "official" block, so the page can be rebuilt from the shipped data alone.
q8 = lambda M: np.clip(np.round(M / np.linalg.norm(M, axis=1, keepdims=True) * 127), -127, 127).astype(np.int8)
ES = q8(np.load(DATA / "embeddings_search.npy")); assert ES.shape[0] == len(recs)
voc = json.loads((DATA / "search_terms.json").read_text()); TV = q8(np.load(DATA / "search_vectors.npy").astype("float32")); assert TV.shape[0] == len(voc["names"])
VOC = [[t, voc["terms"][t]["n"], voc["terms"][t].get("means", "")] for t in voc["names"]]
b64 = lambda M: base64.b64encode(M.tobytes()).decode()
r1 = lambda v: round(v, 1) if isinstance(v, float) else v
data = []
for r in recs:
    L, P, S, K, M, A = r["links"], r["paper"], r["repo_scan"] or {}, r["scores"], r["map"], r["auto_labels"]; repo = L["repo"] or {}; O = r["official"]
    lk = lambda xs: [[x["url"], x["pages"][:2], int(x["likely_own"])] for x in xs]
    data.append({"id": r["arxiv_id"], "t": r["title"], "au": (O.get("authors") or [])[:12], "pres": r["labels"]["presentation"], "trk": r["labels"]["track"], "diff": r["labels"]["difficulty"],
                 "kind": r["labels"]["claim_kind"], "pages": P["pages"], "mb": P["main_body_pages"], "cpg": P["claim_page"], "capp": bool(P["claim_in_appendix"]), "mins": r1(P["read_mins"]),
                 "refs": P.get("reference_count"), "loc": P.get("claim_location_check") or "", "cst": P.get("code_statements") or {}, "rec": L.get("official_claim_record"), "appm": P.get("appendix_mentions") or {},
                 "rd": r1(K["readability"]), "tk": r1(K["turnkey"]), "ck": r1(K["checkability"]), "ckw": K["checkability_why"], "hw": P["hardware_mentions"], "hosted": P["names_hosted_model"],
                 "repo": repo.get("url"), "rs": repo.get("status") or "no_link_found", "rvia": repo.get("found_via"), "rconf": repo.get("confidence"), "rdate": (repo.get("head_date") or "")[:10], "rsha": (repo.get("head_sha") or "")[:8],
                 "npy": S.get("n_python_files"), "deps": S.get("dependency_files") or [], "hard": S.get("blockers") or {}, "soft": S.get("friction") or {}, "boost": S.get("good_signs") or {}, "ign": S.get("ignored_matches") or {},
                 "lic": S.get("licence"), "tests": bool(S.get("has_tests")), "facts": S.get("facts") or {},
                 "gh": lk(L["github_in_pdf"]), "hf": lk(L["huggingface_in_pdf"]), "pp": lk(L["project_pages"]), "anon": lk(L["anonymous_code"]), "orv": L.get("openreview"), "pwc": L.get("papers_with_code"),
                 "sc": M["topic"], "scl": M["topic_name"], "tops": A["topics"], "areas": A["areas"], "x": M["x"], "y": M["y"], "nb": M["similar"],
                 "src": r["claim_source"] or "", "claim": O.get("claim") or "", "metric": O.get("metric") or "", "cmp": O.get("comparison") or "", "crit": O.get("falsification_criterion") or "",
                 "txt": " ".join(" ".join(str(x or "") for x in (O.get("claim"), O.get("comparison"), O.get("metric"), O.get("method"), O.get("background"))).lower().replace("-", " ").split())[:2600]})
HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeurIPS 2026 Reproduction Paper Map</title>
<style>
:root{--bg:#0d1017;--panel:#161b26;--line:#2a3142;--ink:#e8ebf2;--mut:#93a0b8;--acc:#ffb000;--blue:#6ea8fe;--good:#7ee2b0;--bad:#ff9c9c}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 system-ui,sans-serif;display:flex;flex-direction:column}
header{padding:7px 12px;border-bottom:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:var(--panel)}
header h1{font-size:15px;margin:0 8px 0 0}select,input,button,textarea{background:#0f141d;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 7px;font:inherit}
button{cursor:pointer}button.on{background:var(--acc);color:#111;border-color:var(--acc)}label{color:var(--mut);display:inline-flex;gap:4px;align-items:center}
#main{flex:1;display:flex;min-height:0}#left{flex:0 0 42%;display:flex;flex-direction:column;min-width:160px}#mapwrap{flex:1;position:relative;min-height:0}#gl{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
#vs{flex:0 0 7px;cursor:col-resize;background:var(--line)}#hs{flex:0 0 7px;cursor:row-resize;background:var(--line)}#vs:hover,#hs:hover,.dragging{background:var(--acc)!important}
#right{flex:1;display:flex;flex-direction:column;min-width:260px}#prefs{flex:0 0 auto;max-height:55%;overflow:auto;padding:8px 10px;background:var(--panel)}
#tablewrap{flex:1;position:relative;min-height:80px;display:flex;flex-direction:column}#tw{flex:1;overflow:auto}
.steps{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}.steps span{background:#0f141d;border:1px solid var(--line);border-radius:14px;padding:2px 10px;color:var(--mut)}.steps b{color:var(--acc)}
h2{font-size:12px;margin:9px 0 3px;color:var(--acc);text-transform:uppercase;letter-spacing:.04em}h2 small{color:var(--mut);text-transform:none;letter-spacing:0;font-weight:400}
#int{width:100%;height:44px;resize:vertical;font-size:15px;padding:9px 11px;border:2px solid var(--acc);border-radius:8px;background:#0a0e15}#int:focus,#q:focus{outline:none;box-shadow:0 0 0 3px rgba(255,176,0,.3)}#int::placeholder,#q::placeholder{color:#7f8aa3}.biglab{display:block;font-weight:700;color:var(--ink);font-size:14px;margin-bottom:4px}.biglab .mut{font-weight:400;font-size:12px}#q{border:1px solid #55617a;background:#0a0e15;padding:6px 9px;min-width:230px}.row{display:flex;gap:6px 10px;flex-wrap:wrap;align-items:center;margin-top:4px}
.chip{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 9px;cursor:pointer;background:#0f141d;white-space:nowrap}.chip:hover{border-color:var(--acc)}.chip i{color:var(--mut);font-style:normal}
#sl{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:1px 14px}#sl label{justify-content:space-between}#sl input{width:100px;padding:0}#sl b{color:var(--ink);width:16px;text-align:right;display:inline-block}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
table{border-collapse:collapse;width:100%}th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--panel);cursor:pointer;z-index:1}#tb tr:hover{background:#1b2231}tr.sel{background:#263250}td:nth-child(2){min-width:210px}
.b{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;font-weight:700;margin:1px 2px 1px 0;border:1px solid var(--line);white-space:nowrap}
.oral{background:#ffb000;color:#111;border-color:#ffb000}.spotlight{background:#6ea8fe;color:#111;border-color:#6ea8fe}.poster{color:var(--mut)}
.harder{background:#4a1d1d;color:#ff9c9c;border-color:#7a2b2b}.standard{background:#173326;color:#7ee2b0;border-color:#24543d}.trk{color:#c9b6ff;border-color:#4b3f7a}
.auto{border-style:dashed;font-weight:400;font-style:italic;color:#c8d2e6}.src{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.04em;margin-right:3px}
.mut{color:var(--mut)}.bad{color:var(--bad)}.good{color:var(--good)}.sem{color:var(--blue)}
.bar{display:inline-flex;height:7px;border-radius:3px;overflow:hidden;vertical-align:middle;margin-right:5px;background:#0f141d}.bar i{display:block;height:100%}
.over{position:absolute;inset:0;background:var(--panel);overflow:auto;padding:10px 14px;display:none;z-index:5}.over h3{margin:0 30px 4px 0;font-size:15px}
.over h4{margin:9px 0 2px;font-size:12px;color:var(--acc);text-transform:uppercase;letter-spacing:.04em}.over code{color:var(--mut);font-size:11px;word-break:break-all}.kv{margin:1px 0}.over a{word-break:break-all}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:6px;margin:9px 0 4px}.tile{background:#0f141d;border:1px solid var(--line);border-radius:8px;padding:6px 10px}.tile b{display:block;font-size:15px;line-height:1.25}.tile span{color:var(--mut);font-size:11px}.tile.warn{border-color:#7a2b2b}.tile.warn b{color:var(--bad)}.tile.ok{border-color:#24543d}.tile.ok b{color:var(--good)}
.issue{border-left:3px solid #ff6b6b;background:#1b1518;padding:6px 10px;margin:5px 0;border-radius:0 7px 7px 0}.issue b{display:block;margin-bottom:1px}.issue div{color:var(--mut)}.issue code{display:block;margin-top:3px}.issue.ok{border-left-color:#3ecf8e;background:#13201a}
.dl{display:grid;grid-template-columns:125px 1fr;gap:6px 12px;margin:4px 0}.dl dt{color:var(--mut)}.dl dd{margin:0}
.pill{display:inline-block;border:1px solid var(--line);border-radius:14px;padding:2px 11px;margin:1px 2px;text-decoration:none;background:#0f141d;white-space:nowrap}.pill:hover{border-color:var(--acc)}
details.sec{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:0 10px}details.sec[open]{padding-bottom:8px}details.sec>summary{cursor:pointer;font-weight:700;padding:7px 0;color:var(--ink)}.sub{color:var(--acc);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 2px}
@media (max-width:820px){.dl{grid-template-columns:1fr;gap:1px 0}.dl dt{margin-top:6px}}
.over li{cursor:pointer;margin:2px 0}.over li:hover{color:var(--acc)}.over ul{margin:4px 0;padding-left:16px}.x{position:absolute;right:10px;top:8px;z-index:6}
#cmpt th,#cmpt td{border:1px solid var(--line);min-width:170px;max-width:300px;font-size:12px}#cmpt th:first-child,#cmpt td:first-child{min-width:120px;max-width:150px;color:var(--mut);position:sticky;left:0;background:var(--panel)}#cmpt th{position:static;cursor:default}
#tip{position:absolute;pointer-events:none;background:#000d;padding:3px 7px;border-radius:5px;display:none;max-width:320px;z-index:4}
#hint{position:absolute;left:10px;bottom:6px;color:var(--mut);font-size:11px;pointer-events:none}
#legend{position:absolute;right:8px;top:8px;background:rgba(22,27,38,.9);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:11px;max-height:70%;overflow:auto}
#legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}#legend div{cursor:pointer;white-space:nowrap}#legend div.off{opacity:.35}#legend.min div:not(.hd){display:none}
a{color:var(--blue)}details.fold{margin-top:5px}details.fold>summary{cursor:pointer;color:var(--ink);font-weight:600}details.about{color:var(--mut);font-size:12px}details.about td,details.about th{border:1px solid var(--line);padding:3px 6px;position:static;cursor:default}
.sug{display:flex;gap:6px;flex-wrap:wrap;margin:2px 0}.sug .chip{max-width:260px;overflow:hidden;text-overflow:ellipsis}
@media (max-width:820px){html,body{height:auto}body{display:block}#main{display:block}#left{height:46vh;min-width:0}#vs,#hs{display:none}#right{min-width:0}#prefs{max-height:none}
 #tw{overflow-x:auto;overflow-y:visible}th{position:static}.hidem{display:none}body{overflow-x:hidden}#right,#prefs,header{max-width:100vw}#sl{grid-template-columns:1fr}header h1{width:100%}#q{flex:1;min-width:0}td,th{padding:4px 5px}.over{position:fixed;inset:0}#legend{max-height:40%}#hint{display:none}td:nth-child(2){min-width:150px}}
</style></head><body>
<header><h1>NeurIPS 2026 Reproduction Paper Map</h1>
<label style="color:var(--ink)">Find a paper <input id="q" placeholder="type part of a title or an arXiv id" size="28"></label>
<label>Colour dots by <select id="fCol"><option value="sem">topic (automatic)</option><option value="score">my score</option><option value="pres">oral / spotlight</option><option value="tk">turnkey</option></select></label>
<button id="bLinks" title="draw lines joining each paper to its 4 most similar papers">Similarity lines</button><span id="cnt" class="mut"></span></header>
<div id="main"><div id="left"><div id="mapwrap"><canvas id="gl"></canvas><div id="hint">scroll = zoom · drag = pan · hover = title · click = details · double-click = reset</div><div id="tip"></div><div id="legend"></div></div></div>
<div id="vs" title="drag to resize"></div>
<div id="right"><div id="prefs">
<label class="biglab" for="int">What are you interested in? <span class="mut">Type topics separated by commas — the list below re-orders as you type</span></label>
<textarea id="int" placeholder="e.g.  healthcare, brain decoding, LoRA, robot manipulation"></textarea>
<div class="row" id="sugg"></div><div id="mstat" class="mut"></div>
<div class="row"><b style="color:var(--acc)">Shortlist</b><span id="short" class="sug"></span><button id="bCmp">Compare</button></div>
<details class="fold" id="fFilt"><summary>Hide papers I cannot work with <span class="mut" id="nFilt"></span></summary>
<div class="row" id="cons"></div>
<div class="row"><label><input type="checkbox" id="c_match"> show only papers matching my interests</label><label><input type="checkbox" id="mfirst" checked> list matching papers first</label></div>
<div class="row"><label>Topic <select id="fTop"><option value="">all</option></select></label><label>Track <select id="fTrk"><option value="">all</option></select></label></div></details>
<details class="fold" id="wts"><summary>Change how "my score" is weighted <span class="mut">— equal weights by default; there is no built-in ranking</span></summary>
<div class="row"><span class="mut">Presets:</span><button data-pre="bal">Equal weights</button><button data-pre="quick">Quick start</button><button data-pre="run">Runnable repo only</button><button data-pre="talk">Orals &amp; spotlights</button><button data-pre="topic">Just my interests</button></div>
<div id="sl"></div></details>
<details class="fold"><summary>Suggestions <span class="mut" id="sugnote"></span></summary>
<div class="mut">Close to what you asked for</div><div class="sug" id="sugNear"></div>
<div class="mut">Something different — runnable papers from topics your interests do not touch</div><div class="sug" id="sugFar"></div></details>
<details class="fold"><summary>Share, export, reset</summary>
<div class="row"><button id="bOnly">Show only shortlist</button><button id="share">Copy link to these settings</button><button id="csv">Download ranked CSV</button><button id="reset">Reset all</button><span class="mut" id="msg"></span></div></details>
<details class="fold about"><summary>What the numbers and labels mean, and where they come from</summary>
<p><b>No code was run to produce anything on this page.</b> Scripts read each paper's PDF and a shallow copy of its repository. Nobody has measured how well these scores predict whether a paper reproduces. Click a paper to see the evidence behind its numbers.</p>
<table><tr><th>Label</th><th>Who assigned it</th><th>How it reached this page</th></tr>
<tr><td>Oral / spotlight / poster, track</td><td>The NeurIPS 2026 programme committee</td><td>Copied unchanged from the hackathon's official claim list</td></tr>
<tr><td>Standard / "harder than most", the claim to verify, table-or-figure, metric</td><td>The hackathon organisers</td><td>Copied unchanged from the official claim list (hosted on Hugging Face)</td></tr>
<tr><td>Topics and areas (dashed, italic)</td><td>Nobody — computed by this tool from the text</td><td>Automatic and rough. A paper can carry several, or none</td></tr>
<tr><td>Turnkey, checkable, read time, match, my score</td><td>Nobody — heuristics of this tool</td><td>Rules are listed in the project README</td></tr></table>
<p>Nothing here comes from the paper authors' own keywords (the claim list has none) or from Papers with Code. That site has its own categories for NeurIPS 2026 papers; they are not used here because the site does not permit automated collection. Only the link to each paper's page is shown.</p>
<p><b>Turnkey 0–10</b>: how runnable the repository looks. Starts at 3; +1 per good sign (Hugging Face data or weights loaded in code, lock file, Dockerfile, eval script, reproduce script); −1 or −2 per friction (Google Drive downloads, C++/CUDA build, gated data, video scraping, no dependency file, very short README). <b>0</b> only when there are no code files, the README says the code is not released, or the method under test is a hosted closed model. <b>"no repo found"</b> = no GitHub link in the PDF or on its project page; the code may still exist.</p>
<p><b>Checkable 0–10</b>: how cleanly you can tell whether you matched the claim (exact numbers in a table, a reported spread, few benchmarks). <b>Read</b>: estimated minutes for abstract + method + setup.</p>
<p><b>Match</b>: an <span class="good">exact</span> match means your word appears (at the start of a word) in the title (100%) or in the claim, method or background (70%). A <span class="sem">related</span> match (up to 60%) means the paper's text is unusually close in meaning to your word, judged by a text-embedding model; it is approximate and can be wrong.</p>
<p><b>Similarity</b> (the "sim" numbers and the lines on the map) is the cosine similarity between text embeddings of two papers' title + claim. In this collection two random papers score about 0.68 and a paper's nearest neighbours 0.80–0.90. High similarity means similar wording and subject. It does not mean shared code, data or authors.</p></details>
</div><div id="hs" title="drag to resize"></div>
<div id="tablewrap"><div id="tw"><table><thead><tr id="hd"></tr></thead><tbody id="tb"></tbody></table></div>
<div class="over" id="card"></div><div class="over" id="cmp"></div></div></div></div>
<script>
const D=__DATA__, VOC=__VOC__, DIM=__DIM__;
const dec=s=>new Int8Array(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer), E=dec("__ES__"), TV=dec("__TV__");
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const IDX=Object.fromEntries(D.map((r,i)=>[r.id,i])), VIDX=Object.fromEntries(VOC.map((v,i)=>[v[0],i])); const SEMN={}; D.forEach(r=>SEMN[r.sc]=r.scl);
const NK=o=>Object.keys(o).length, nice=k=>String(k).replace(/_/g,' ');
const W=[['tk','Runnable repo (turnkey)','#3ecf8e'],['ck','Checkable claim','#6ea8fe'],['rd','Quick to read','#c9b6ff'],['pres','Oral / spotlight','#ffb000'],['std','Not "harder than most"','#5eead4'],['match','Matches my interests','#ff8fab']], MI=5;
const PRE={bal:[5,5,5,5,5,5],quick:[9,3,6,0,6,5],run:[10,0,0,0,0,0],talk:[3,2,2,10,2,5],topic:[0,0,0,0,0,10]};
const CONS=[['norepo','no repository was found',r=>r.rs!=='cloned'],['blk','the repository has a blocker',r=>NK(r.hard)>0],['gated','data or weights need an access request',r=>'gated_access' in r.soft],
 ['hosted','the text names a hosted model (GPT, Claude, Gemini…)',r=>r.hosted],['supp','code is only in supplementary material / promised / on request',r=>r.rs!=='cloned'&&NK(r.cst||{})>0],['native','a C++/CUDA build is needed',r=>'native_build' in r.soft],['dl','downloads go through Google Drive / Baidu',r=>'google_drive_download' in r.soft||'baidu_pan_download' in r.soft],
 ['harder','the organisers labelled it "harder than most"',r=>r.diff==='harder'],['app','the claimed result is in the appendix',r=>r.capp],['poster','it is a poster (keep orals and spotlights)',r=>r.pres==='poster']];
const S={w:PRE.bal.slice(),sort:'score',dir:-1,sel:null,cons:{},likes:[],short:[],only:false,sem:null,links:false,hov:-1,mf:false};
// ---- state <-> URL hash
function save(){const h='w='+S.w.join('.')+'&q='+encodeURIComponent($('#int').value)+'&l='+S.likes.join('~')+'&s='+S.short.join('~')+'&c='+CONS.filter(c=>S.cons[c[0]]).map(c=>c[0]).join('.');try{history.replaceState(null,'','#'+h)}catch(e){}}
function load(){const p=new URLSearchParams(location.hash.slice(1));if(p.get('w')){const w=p.get('w').split('.').map(Number);if(w.length===W.length&&w.every(x=>x>=0&&x<=10))S.w=w}
 if(p.get('q'))$('#int').value=p.get('q');const ids=k=>(p.get(k)||'').split('~').filter(x=>x in IDX);S.likes=ids('l');S.short=ids('s').slice(0,5);(p.get('c')||'').split('.').forEach(k=>{if(CONS.some(c=>c[0]===k))S.cons[k]=true})}
// ---- search: exact keyword + meaning (vocabulary vectors precomputed at build time; no model runs in the browser)
function terms(){return [...new Set($('#int').value.toLowerCase().replace(/-/g,' ').split(/[,;\n]+/).map(s=>s.trim().replace(/\s+/g,' ')).filter(s=>s.length>1))]}
function tvec(t){let ids=[];if(t in VIDX)ids=[VIDX[t]];else{const ws=t.split(' ');for(let i=0;i<ws.length;i++){const bi=ws[i]+' '+(ws[i+1]||'');if(bi in VIDX){ids.push(VIDX[bi]);i++}else if(ws[i] in VIDX)ids.push(VIDX[ws[i]])}}
 if(!ids.length)return null;const v=new Float32Array(DIM);ids.forEach(j=>{for(let k=0;k<DIM;k++)v[k]+=TV[j*DIM+k]});return v}
function zsim(v){const s=new Float32Array(D.length);let m=0,q=0;for(let i=0;i<D.length;i++){let a=0;const o=i*DIM;for(let k=0;k<DIM;k++)a+=E[o+k]*v[k];s[i]=a;m+=a}m/=D.length;for(let i=0;i<D.length;i++)q+=(s[i]-m)**2;q=Math.sqrt(q/D.length)||1;for(let i=0;i<D.length;i++)s[i]=(s[i]-m)/q;return s}
function pvec(i){const v=new Float32Array(DIM);for(let k=0;k<DIM;k++)v[k]=E[i*DIM+k];return v}
let INFO=[];
function score(){const T=terms(),L=S.likes.map(id=>IDX[id]),useM=T.length||L.length,w=S.w,den=w.reduce((s,x,i)=>s+(i===MI&&!useM?0:x),0)||1;
 const RX=T.map(t=>new RegExp('(^|[^a-z0-9])'+t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'))),ZS=T.map(t=>{const v=tvec(t);return v?zsim(v):null}),ZL=L.map(j=>zsim(pvec(j)));
 INFO=T.map((t,i)=>({t,exact:0,rel:0,vec:!!ZS[i],means:t in VIDX?VOC[VIDX[t]][2]:''}));
 D.forEach((r,i)=>{let kw=0,n=0,sm=0;const tl=r.t.toLowerCase().replace(/-/g,' ');r.kwhit=[];r.semhit=[];
  RX.forEach((x,ti)=>{const v=x.test(tl)?1:x.test(r.txt)?.7:0;if(v){n++;INFO[ti].exact++;kw=Math.max(kw,v);r.kwhit.push(T[ti])}else if(ZS[ti]){const s=Math.min(.6,Math.max(0,(ZS[ti][i]-2.2)/1.8*.6));if(s>=.1){INFO[ti].rel++;sm=Math.max(sm,s);r.semhit.push(T[ti])}}});
  if(n>1)kw=Math.min(1,kw+.15*(n-1));let lk=0;ZL.forEach((z,li)=>{lk=Math.max(lk,L[li]===i?1:Math.min(1,Math.max(0,(z[i]-1.5)/2)))});
  r.match=useM?Math.max(kw,sm,lk):null;
  const c=[(r.tk??0)/10,(r.ck??5)/10,(r.rd??5)/10,r.pres==='oral'?1:r.pres==='spotlight'?.5:0,r.diff==='standard'?1:0,r.match??0];
  r.parts=c.map((x,k)=>(k===MI&&!useM?0:x*w[k])/den*100);r.score=Math.round(r.parts.reduce((a,b)=>a+b,0)*10)/10;r.base=Math.round((r.score-r.parts[MI])*10)/10});
 S.mf=useM&&$('#mfirst').checked;[...D].sort(byScore).forEach((r,k)=>r.rank=k+1);status(T,L,useM);suggest(useM)}
const tier=r=>r.kwhit&&r.kwhit.length?2:r.match>0?1:0;   // exact word > related by meaning or similar to a starred paper > no match
const byScore=(a,b)=>!S.mf?b.score-a.score:tier(a)!==tier(b)?tier(b)-tier(a):tier(a)===1&&a.match!==b.match?b.match-a.match:b.score-a.score;   // related matches: closest in meaning first
function status(T,L,useM){const nm=D.filter(r=>r.match>0).length;
 $('#mstat').innerHTML=!T.length?(L.length?`${nm} papers are similar to the ${L.length} you starred.`:''):`<b style="color:var(--ink)">${nm} papers match.</b> `+INFO.map(o=>{
  const near=!o.exact&&!o.rel?VOC.filter(v=>v[0]!==o.t&&(v[0].startsWith(o.t.slice(0,4))||o.t.startsWith(v[0]))).sort((a,b)=>b[1]-a[1]).slice(0,4).map(v=>`<span class="chip" data-add="${esc(v[0])}">${esc(v[0])}</span>`).join(' '):'';
  return `<span title="${esc(o.means)}"><b>${esc(o.t)}</b>: <span class="good">${o.exact} exact</span>${o.vec?` · <span class="sem">${o.rel} related by meaning</span>`:''}${o.means?' <span class="mut">(searched as: '+esc(o.means.slice(0,70))+'…)</span>':''}${!o.exact&&!o.rel?` <span class="bad">— nothing found${o.vec?'':'; this word is not in the collection'}</span>${near?' try: '+near:''}`:''}</span>`}).join(' &nbsp;·&nbsp; ')+(S.w[MI]===0?' <span class="bad">The "Matches my interests" weight is 0.</span>':'')}
// suggestions while typing the last term
function typing(){const raw=$('#int').value,last=raw.split(/[,;\n]/).pop().trim().toLowerCase().replace(/-/g,' ');const box=$('#sugg');if(last.length<2){box.innerHTML='';return}
 const hit=VOC.filter(v=>v[0].startsWith(last)&&v[0]!==last).sort((a,b)=>(b[2]?1:0)-(a[2]?1:0)||b[1]-a[1]).slice(0,7);const more=hit.length<7?VOC.filter(v=>!v[0].startsWith(last)&&v[0].includes(last)).sort((a,b)=>b[1]-a[1]).slice(0,7-hit.length):[];
 box.innerHTML=hit.concat(more).map(v=>`<span class="chip" data-comp="${esc(v[0])}" title="${esc(v[2])}">${esc(v[0])} <i>${v[2]?'topic':v[1]}</i></span>`).join('')}
function setLast(t){const parts=$('#int').value.split(/([,;\n])/);parts[parts.length-1]=(parts.length>1?' ':'')+t;$('#int').value=parts.join('')+', ';$('#sugg').innerHTML='';$('#int').focus();render()}
// "close" and "something different"
function suggest(useM){const free=r=>!S.short.includes(r.id)&&pass(r,true),chip=r=>`<span class="chip" data-open="${r.id}" title="${esc(r.t)}">${esc(r.t.slice(0,38))}… <i>${esc(r.tops[0])}</i></span>`;
 if(!useM){$('#sugnote').textContent='— appear once you type an interest above, or star (☆) a paper';const h='<span class="mut">Nothing yet. Type an interest in the box at the top (for example “robotics”), or press ☆ on a paper you like, and suggestions will appear here.</span>';$('#sugNear').innerHTML=h;$('#sugFar').innerHTML='<span class="mut">Needs your interests first, so it knows what “different” means for you.</span>';return}
 const M=D.filter(r=>r.match>0&&free(r)).sort((a,b)=>b.match-a.match||b.score-a.score);$('#sugNear').innerHTML=M.slice(0,5).map(chip).join('')||'<span class="mut">No paper matches your interests with the current switches. Try another word, or untick a switch under “Hide papers I cannot work with”.</span>';
 const mine=new Set(D.filter(r=>r.match>0).sort((a,b)=>b.match-a.match).slice(0,40).map(r=>r.sc)),seen=new Set(),far=[];
 D.filter(r=>!(r.match>0)&&!mine.has(r.sc)&&free(r)&&r.rs==='cloned'&&!NK(r.hard)).sort((a,b)=>b.base-a.base).forEach(r=>{if(far.length<5&&!seen.has(r.sc)){seen.add(r.sc);far.push(r)}});
 $('#sugFar').innerHTML=far.map(chip).join('')||'<span class="mut">Your interests already touch every topic, so there is nothing far away to suggest.</span>';$('#sugnote').textContent=`— ${Math.min(5,M.length)} close to your interests, ${far.length} different (click to open)`}
// ---- filters + table
[...new Set(D.map(r=>r.sc))].sort((a,b)=>a-b).forEach(c=>$('#fTop').add(new Option(SEMN[c],c)));[...new Set(D.map(r=>r.trk))].sort().forEach(c=>$('#fTrk').add(new Option(c,c)));
function pass(r,noOnly){const q=$('#q').value.toLowerCase();
 if($('#fTop').value!==''&&String(r.sc)!==$('#fTop').value)return false;if($('#fTrk').value&&r.trk!==$('#fTrk').value)return false;if(S.sem!=null&&colKey(r)!==S.sem)return false;
 for(const c of CONS)if(S.cons[c[0]]&&c[2](r))return false;if($('#c_match').checked&&!(r.match>0))return false;if(S.only&&!noOnly&&!S.short.includes(r.id))return false;
 if(q&&!(r.t.toLowerCase().includes(q)||r.id.includes(q)))return false;return true}
const badges=r=>`<span title="Assigned by the NeurIPS 2026 programme committee; copied from the claim list"><span class="src">NeurIPS</span><span class="b ${r.pres}">${r.pres.toUpperCase()}</span><span class="b trk">${esc(r.trk)}</span></span> <span title="Assigned by the hackathon organisers in the official claim list"><span class="src">Hackathon</span><span class="b ${r.diff}">${r.diff==='harder'?'HARDER THAN MOST':'standard'}</span><span class="b mut">${r.kind} claim</span></span>`;
const autos=r=>`<span title="Computed by this tool from the paper's text. Automatic and rough; a paper can have several or none. Not assigned by the authors, NeurIPS, the hackathon or Papers with Code."><span class="src">Auto</span>${r.tops.map(t=>`<span class="b auto">${esc(t)}</span>`).join('')}${r.areas.map(t=>`<span class="b auto" style="color:#ffd27a">${esc(t)}</span>`).join('')}</span>`;
const claimLinks=r=>`<a href="${esc(r.rec)}" target="_blank" title="this paper's record in the official claim list, with the picture of the claimed table or figure">official record</a>`;
const repo=r=>r.repo?`<a href="${esc(r.repo)}" target="_blank">repo</a>`:`<span class="mut">no repo found</span>`;
const tkc=r=>r.rs!=='cloned'?`<span class="mut">– (${r.rs==='dead'?'repo link dead':'no repo found'})</span>`:NK(r.hard)?`<span class="bad">0 — ${esc(Object.keys(r.hard).map(nice).join(', '))}</span>`:`${r.tk}`;
const bar=r=>`<span class="bar" style="width:${Math.max(4,Math.round(r.score*.6))}px" title="${W.map((w,k)=>w[1]+': '+r.parts[k].toFixed(1)).join('\n')}">${r.parts.map((p,k)=>p>0?`<i style="width:${p/Math.max(r.score,.1)*100}%;background:${W[k][2]}"></i>`:'').join('')}</span>`;
const COLS=[['rank','#',''],['t','Paper',''],['labels','Labels (hover for source)','hidem'],['mins','Read','hidem'],['tk','Turnkey',''],['ck','Checkable','hidem'],['match','Match',''],['score','My score',''],['pin','','']];
let VIS=D.map(()=>true);
function render(){score();const rows=D.filter(r=>pass(r)).sort((a,b)=>{if(S.sort==='score'||S.sort==='rank'){const c=byScore(a,b);return S.sort==='rank'?c*S.dir:-c*S.dir}const x=a[S.sort],y=b[S.sort];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*S.dir});
 $('#cnt').textContent=rows.length+' of '+D.length+' papers shown';
 $('#hd').innerHTML=COLS.map(([k,l,c])=>`<th data-k="${k}" class="${c}">${l}${S.sort===k?(S.dir<0?' ▼':' ▲'):''}</th>`).join('');
 $('#tb').innerHTML=rows.slice(0,300).map(r=>`<tr data-id="${r.id}" class="${S.sel===r.id?'sel':''}"><td>${r.rank}</td><td><b>${esc(r.t)}</b><br><a href="https://arxiv.org/abs/${r.id}" target="_blank">${r.id}</a> · ${repo(r)} · ${claimLinks(r)}${r.kwhit.length?` <span class="good" title="your word appears in this paper">${esc(r.kwhit.join(', '))}</span>`:''}${r.semhit.length?` <span class="sem" title="related by meaning (approximate)">≈ ${esc(r.semhit.join(', '))}</span>`:''}</td><td class="hidem">${badges(r)}<br>${autos(r)}</td><td class="hidem">${r.mins!=null?'~'+Math.round(r.mins)+' min':'–'}</td><td>${tkc(r)}</td><td class="hidem">${r.ck??'–'}</td><td>${r.match==null?'<span class="mut">–</span>':Math.round(r.match*100)+'%'}</td><td>${bar(r)}<b>${r.score}</b></td><td style="white-space:nowrap"><button data-like="${r.id}" title="more like this: rank similar papers higher">${S.likes.includes(r.id)?'★':'☆'}</button> <button data-short="${r.id}" class="${S.short.includes(r.id)?'on':''}" title="add to my shortlist (max 5)">${S.short.includes(r.id)?'✓':'+'}</button></td></tr>`).join('')+(rows.length>300?`<tr><td colspan="9" class="mut">showing the first 300 of ${rows.length}; narrow with interests, the switches above, or search</td></tr>`:'');
 $('#short').innerHTML=S.short.length?S.short.map((id,i)=>`<span class="chip" data-open="${id}" title="${esc(D[IDX[id]].t)}">${i+1}. ${esc(D[IDX[id]].t.slice(0,22))}… <i data-rm="${id}">✕</i></span>`).join(''):'<span class="mut">press + on up to 5 papers, then compare them</span>';
 $('#bOnly').classList.toggle('on',S.only);W.forEach(([k],i)=>{$('#w_'+k).value=S.w[i];$('#v_'+k).textContent=S.w[i]});
 CONS.forEach(c=>{$('#c_'+c[0]).checked=!!S.cons[c[0]]});const nf=CONS.filter(c=>S.cons[c[0]]).length+($('#c_match').checked?1:0)+($('#fTop').value!==''?1:0)+($('#fTrk').value?1:0);$('#nFilt').textContent=nf?`— ${nf} active, hiding ${D.length-rows.length} papers`:'— no repository, gated data, hosted model, …';VIS=D.map(r=>pass(r));draw();legend();save()}
// ---- map
const PAL=['#6ea8fe','#ffb000','#3ecf8e','#ff8fab','#c9b6ff','#f97316','#5eead4','#ff6b6b','#a3e635','#e879f9','#38bdf8','#fbbf24','#94a3b8','#fb7185'];
const tkKey=r=>r.rs!=='cloned'?'no repo found':r.tk>=7?'turnkey 7-10':r.tk>=4?'turnkey 4-6':'turnkey 0-3',tkCol={'no repo found':'#5b6578','turnkey 7-10':'#3ecf8e','turnkey 4-6':'#ffb000','turnkey 0-3':'#ff6b6b'};
const scKey=r=>r.rank<=25?'my top 25':r.rank<=100?'my top 100':r.rank<=300?'my top 300':'the rest',scCol={'my top 25':'#ffb000','my top 100':'#3ecf8e','my top 300':'#6ea8fe','the rest':'#3a4356'};
function colKey(r){const m=$('#fCol').value;return m==='sem'?String(r.sc):m==='pres'?r.pres:m==='tk'?tkKey(r):scKey(r)}
function colOf(r){const m=$('#fCol').value;return m==='sem'?PAL[r.sc%PAL.length]:m==='pres'?({oral:'#ffb000',spotlight:'#6ea8fe',poster:'#5b6578'})[r.pres]:m==='tk'?tkCol[tkKey(r)]:scCol[scKey(r)]}
function legend(){const cnt={},col={};D.forEach(r=>{const k=colKey(r);cnt[k]=(cnt[k]||0)+1;col[k]=colOf(r)});const sem=$('#fCol').value==='sem';
 $('#legend').innerHTML='<div class="hd" data-min="1" style="color:#93a0b8">▾ legend — click a group to isolate it</div>'+Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a]).map(k=>`<div data-k="${esc(k)}" class="${S.sem!=null&&S.sem!==k?'off':''}"><i style="background:${col[k]}"></i>${esc(sem?SEMN[k]:k)} (${cnt[k]})</div>`).join('')+'<div style="margin-top:4px;color:#93a0b8;cursor:default">'+(sem?'topics are automatic (main one shown)<br>':'')+'dot size: oral &gt; spotlight &gt; poster<br>white ring: matches your interests<br>square: on your shortlist</div>'}
$('#legend').onclick=e=>{if(e.target.closest('[data-min]')){$('#legend').classList.toggle('min');return}const d=e.target.closest('[data-k]');if(!d)return;S.sem=S.sem===d.dataset.k?null:d.dataset.k;render()};
const cv=$('#gl'),g=cv.getContext('2d');let V={cx:0,cy:0,k:1},CW=1,CH=1,drag=null;
const base=()=>Math.min(CW,CH)*.46,sx=r=>CW/2+(r.x-V.cx)*base()*V.k,sy=r=>CH/2-(r.y-V.cy)*base()*V.k,rad=r=>(r.pres==='oral'?7:r.pres==='spotlight'?5:3)*Math.min(2.2,Math.max(1,Math.sqrt(V.k)*.8));
function size(){const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;if(!w||!h)return;CW=w;CH=h;cv.width=w*dpr;cv.height=h*dpr;g.setTransform(dpr,0,0,dpr,0,0)}
function line(a,b,style,lw){g.strokeStyle=style;g.lineWidth=lw;g.beginPath();g.moveTo(sx(a),sy(a));g.lineTo(sx(b),sy(b));g.stroke()}
function draw(){size();g.clearRect(0,0,CW,CH);const sel=S.sel!=null?IDX[S.sel]:-1,focus=sel>=0?sel:S.hov,anyM=D.some(r=>r.match>0),hov2=sel>=0&&S.hov>=0&&S.hov!==sel?S.hov:-1;
 if(S.links)D.forEach((r,i)=>{if(!VIS[i])return;r.nb.forEach(([id])=>{const j=IDX[id];if((j>i||!D[j].nb.some(n=>n[0]===r.id))&&VIS[j])line(r,D[j],'rgba(147,160,184,.12)',1)})});
 D.forEach((r,i)=>{const x=sx(r),y=sy(r);if(x<-20||y<-20||x>CW+20||y>CH+20)return;const m=r.match>0;g.globalAlpha=!VIS[i]?.06:anyM&&!m?.3:.92;g.fillStyle=colOf(r);g.beginPath();g.arc(x,y,rad(r),0,6.2832);g.fill();
  if(VIS[i]&&m){g.globalAlpha=1;g.strokeStyle='#fff';g.lineWidth=1.4;g.beginPath();g.arc(x,y,rad(r)+2,0,6.2832);g.stroke()}
  if(S.short.includes(r.id)){g.globalAlpha=1;g.strokeStyle='#ffb000';g.lineWidth=2;const q=rad(r)+5;g.strokeRect(x-q,y-q,2*q,2*q)}});g.globalAlpha=1;
 if(focus>=0){const r=D[focus];r.nb.forEach(([id])=>{const n=D[IDX[id]];line(r,n,'rgba(255,176,0,.9)',1.6);g.fillStyle=colOf(n);g.strokeStyle='#ffb000';g.lineWidth=1.5;g.beginPath();g.arc(sx(n),sy(n),rad(n)+2,0,6.2832);g.fill();g.stroke()});
  g.fillStyle=colOf(r);g.strokeStyle='#fff';g.lineWidth=2.5;g.beginPath();g.arc(sx(r),sy(r),rad(r)+4,0,6.2832);g.fill();g.stroke()}
 if(hov2>=0){const r=D[hov2];g.fillStyle=colOf(r);g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.arc(sx(r),sy(r),rad(r)+3,0,6.2832);g.fill();g.stroke()}
 if($('#fCol').value==='sem'){const c={};D.forEach((r,i)=>{if(!VIS[i])return;const o=c[r.sc]||(c[r.sc]=[0,0,0]);o[0]+=sx(r);o[1]+=sy(r);o[2]++});
  g.font='600 12px system-ui';g.textAlign='center';Object.entries(c).forEach(([k,o])=>{if(o[2]<3)return;const t=SEMN[k],x=o[0]/o[2],y=o[1]/o[2],w=g.measureText(t).width;g.fillStyle='rgba(13,16,23,.72)';g.fillRect(x-w/2-4,y-11,w+8,16);g.fillStyle=PAL[k%PAL.length];g.fillText(t,x,y+1)})}}
// every dot that is drawn can be hovered and clicked, including dimmed ones hidden by a switch; visible dots win when two overlap
function pick(e){const b=cv.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top;let best=-1,bd=1e9;D.forEach((r,i)=>{const d=Math.hypot(sx(r)-mx,sy(r)-my)-rad(r)+(VIS[i]?0:4);if(d<bd){bd=d;best=i}});return bd<=12?best:-1}
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,cx:V.cx,cy:V.cy,moved:false};cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'});
cv.addEventListener('pointerup',e=>{const d=drag;drag=null;cv.style.cursor='grab';if(d&&!d.moved){const i=pick(e);if(i>=0)select(D[i].id,true);else if(S.sel){closeCard()}}});
cv.addEventListener('pointermove',e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;if(drag.moved){V.cx=drag.cx-dx/(base()*V.k);V.cy=drag.cy+dy/(base()*V.k);draw()}return}
 const i=pick(e),t=$('#tip');if(i!==S.hov){S.hov=i;draw()}if(i<0){t.style.display='none';cv.style.cursor='grab';return}cv.style.cursor='pointer';const b=cv.getBoundingClientRect(),r=D[i];t.style.display='block';t.style.left=Math.max(0,Math.min(e.clientX-b.left+14,CW-330))+'px';t.style.top=(e.clientY-b.top+14)+'px';t.innerHTML=`<b>${esc(r.t)}</b><br><span style="color:#93a0b8">my #${r.rank} · score ${r.score} · ${r.pres} · ${r.diff} · ${esc(r.tops.join(' + '))}</span>${VIS[i]?'':'<br><span class="bad">hidden from the list by your switches or filters — click to read it anyway</span>'}`});
cv.addEventListener('pointerleave',()=>{$('#tip').style.display='none';if(S.hov!==-1){S.hov=-1;draw()}});
cv.addEventListener('wheel',e=>{e.preventDefault();const b=cv.getBoundingClientRect(),mx=e.clientX-b.left-CW/2,my=e.clientY-b.top-CH/2,k0=V.k,k1=Math.min(40,Math.max(.6,k0*Math.exp(-e.deltaY*.0015)));V.cx+=mx/base()*(1/k0-1/k1);V.cy-=my/base()*(1/k0-1/k1);V.k=k1;draw()},{passive:false});
cv.addEventListener('dblclick',()=>{V={cx:0,cy:0,k:1};draw()});window.addEventListener('resize',draw);
// ---- resizable panels
function splitter(el,horizontal,target,min){el.addEventListener('pointerdown',e=>{el.setPointerCapture(e.pointerId);el.classList.add('dragging');const start=horizontal?e.clientY:e.clientX,box=target.getBoundingClientRect(),s0=horizontal?box.height:box.width;
  const mv=ev=>{const v=Math.max(min,s0+(horizontal?ev.clientY:ev.clientX)-start);if(horizontal){target.style.flex='0 0 '+v+'px';target.style.maxHeight='none'}else target.style.flex='0 0 '+v+'px';draw()};
  const up=()=>{el.classList.remove('dragging');el.removeEventListener('pointermove',mv);el.removeEventListener('pointerup',up)};el.addEventListener('pointermove',mv);el.addEventListener('pointerup',up)})}
splitter($('#vs'),false,$('#left'),160);splitter($('#hs'),true,$('#prefs'),60);
// ---- details
const ev=(o,cls)=>NK(o)?Object.entries(o).map(([k,v])=>`<div class="kv"><span class="${cls}">${esc(nice(k))}</span>${v&&v.weight?` <span class="mut">(−${v.weight})</span>`:''} <code>${esc(typeof v==='string'?v:v.evidence)}</code></div>`).join(''):'<div class="kv mut">none found</div>';
const lnk=(xs,empty)=>xs.length?xs.map(([u,p,own])=>`<div class="kv"><a href="${esc(u)}" target="_blank">${esc(u.replace(/^https?:\/\//,''))}</a> <span class="mut">p.${p.join(', ')}${own?' · looks like the authors\' own':' · probably cited third-party work'}</span></div>`).join(''):`<div class="kv mut">${empty}</div>`;
const OR=r=>r.orv?`<a href="${esc(r.orv)}" target="_blank">OpenReview page</a>`:'OpenReview page';
function issues(r){const o=[],c=r.cst||{},q=encodeURIComponent('"'+r.t+'"'),find=`search <a href="https://github.com/search?q=${q}&type=repositories" target="_blank">GitHub</a> or <a href="https://huggingface.co/search/full-text?q=${q}" target="_blank">Hugging Face</a> by title`;
 if(r.rs==='no_link_found')o.push(['No code repository was found in the PDF or on its project page',`The code may still exist. Look for a supplementary file on the ${OR(r)} (this could not be checked automatically, and downloading may need a free OpenReview account); ${find}; check the links listed below; or ask the authors on OpenReview.`]);
 if(r.rs==='dead')o.push(['The repository link in the paper returns 404','It may have been renamed, moved to an organisation or made private: '+find+', or ask the authors.']);
 if(c.in_supplementary_material&&r.rs!=='cloned')o.push(['The paper says its code is in the supplementary material',`Download the supplementary file from the ${OR(r)}. You may need to sign in with a free OpenReview account to download it. <code>“${esc(c.in_supplementary_material)}”</code>`]);
 if(c.on_openreview&&r.rs!=='cloned')o.push(['The paper points to OpenReview for code',`Look on the ${OR(r)}; a free account may be needed. <code>“${esc(c.on_openreview)}”</code>`]);
 if(c.promised_for_later&&r.rs!=='cloned')o.push(['The paper promised a later release',`It may be out by now: ${find}. <code>“${esc(c.promised_for_later)}”</code>`]);
 if(c.on_request)o.push(['Something is available only on request','Write to the authors early; replies can take days, and data may come with conditions. <code>“'+esc(c.on_request)+'”</code>']);
 if(r.hard.no_code_files)o.push(['The repository has no code files yet','It was read at one commit and may have been filled in since. Open it and check, or ask the authors for a timeline.']);
 if(r.hard.code_not_released)o.push(['The README says the code is not released yet','Check the repository again, or ask the authors for a timeline. <code>“'+esc(r.hard.code_not_released)+'”</code>']);
 if(r.hard.closed_api_target)o.push(['The method under test is a hosted closed model','You need paid API access to the same model, and the model may have changed since the paper.']);
 if(r.soft.gated_access)o.push(['Data or weights need an access request','Sign in and accept the terms, or fill in the form, on day one: approval can take days. <code>'+esc(r.soft.gated_access.evidence)+'</code>']);
 if(r.soft.partial_release_pending)o.push(['Part of the release is still pending','Read the README to see which part; it may not be the part your claim needs. <code>'+esc(r.soft.partial_release_pending.evidence)+'</code>']);
 if(r.soft.google_drive_download||r.soft.baidu_pan_download)o.push(['Downloads go through Google Drive'+(r.soft.baidu_pan_download?' / Baidu Pan':''),'Budget time for manual or rate-limited downloads'+(r.soft.baidu_pan_download&&!r.soft.google_drive_download?'; Baidu Pan needs an account and is slow outside China':'')+'.']);
 if(r.soft.native_build)o.push(['A C++/CUDA extension has to be built','Compare your CUDA and compiler versions with the repository\'s before starting; a container with the authors\' versions is usually the quickest fix.']);
 if(r.soft['yt-dlp']||r.soft['youtube.com'])o.push(['Data comes from video sites','Check the videos still exist before committing; some will have been removed.']);
 if(r.hosted&&!r.hard.closed_api_target)o.push(['The text names a hosted model (GPT, Claude, Gemini…)','Find out whether it is the method under test (API access needed), a judge (an open model can stand in, at the cost of comparability) or only a baseline you can skip.']);
 if(r.capp)o.push(['The claimed result is in the appendix','The setup details are usually there too; read that section, not just the main text.']);
 if(r.loc&&!r.loc.startsWith('found on the stated page')&&!r.loc.startsWith('not checked'))o.push(['The claimed table or figure was not on the stated page of the arXiv PDF','The arXiv version may differ from the one the organisers used. Go by the picture in the official record.']);
 return o}
const SIMNOTE='sim = cosine similarity of the text embeddings of title + claim. Two random papers here score about 0.68; nearest neighbours 0.80–0.90. It means similar wording and subject, not shared code, data or authors.';
function closeCard(){S.sel=null;$('#card').style.display='none';render()}
function select(id,scroll){S.sel=id;const r=D[IDX[id]];V.cx=r.x;V.cy=r.y;if(!scroll)V.k=Math.max(V.k,3);$('#cmp').style.display='none';
 const q=encodeURIComponent('"'+r.t+'"'),other=r.gh.filter(g=>g[0].toLowerCase()!==(r.repo||'').toLowerCase()),am=r.appm||{},I=issues(r),okLoc=r.loc.startsWith('found on the stated page'),scanned=r.rs==='cloned';
 const tile=(cls,big,small)=>`<div class="tile ${cls}"><b>${big}</b><span>${small}</span></div>`;
 const tiles=[
  scanned?tile(NK(r.hard)?'warn':r.tk>=7?'ok':'',NK(r.hard)?'Blocked':`Turnkey ${r.tk}/10`,'code repository found'):tile('warn',r.rs==='dead'?'Dead link':'No repo found','code repository'),
  tile(I.length?'warn':'ok',I.length?`${I.length} thing${I.length>1?'s':''}`:'Nothing flagged','in your way'),
  tile('',`${r.ck}/10`,'claim is checkable'),
  tile('',r.mins!=null?`~${Math.round(r.mins)} min`:'?',`to read · ${r.pages??'?'} pages`),
  tile(r.capp?'warn':'',`Page ${r.cpg??'?'}`,r.capp?'claim is in the appendix':'where the claim is'),
  tile('',`${r.score}`,`my score · #${r.rank}`)].join('');
 const repoLine=r.repo?`<div class="kv"><a href="${esc(r.repo)}" target="_blank">${esc(r.repo.replace('https://',''))}</a><br><span class="mut">found via ${esc(r.rvia==='pdf'?'the PDF':'the project page')} · confidence ${esc(r.rconf||'?')} · ${scanned?`read at commit ${r.rsha}, last commit ${r.rdate}`:r.rs==='dead'?'<span class="bad">link returns 404</span>':esc(nice(r.rs))}</span></div>`:'<div class="kv mut">None found in the PDF or on its project page.</div>';
 const nLinks=other.length+r.hf.length+r.pp.length+r.anon.length+(r.repo?1:0);
 $('#card').style.display='block';$('#card').scrollTop=0;$('#card').innerHTML=`<button class="x" data-close="1">✕ back to list</button>
 <h3>${esc(r.t)}</h3><div class="mut" style="font-size:12px">${esc(r.au.join(', '))}</div>
 <div style="margin:6px 0 2px">${badges(r)} ${autos(r)}</div>
 <div class="row"><button data-short="${r.id}" class="${S.short.includes(r.id)?'on':''}">${S.short.includes(r.id)?'✓ on my shortlist':'+ add to my shortlist'}</button><button data-like="${r.id}" class="${S.likes.includes(r.id)?'on':''}">${S.likes.includes(r.id)?'★ ranking similar papers higher':'☆ show me more like this'}</button>
  <a class="pill" href="https://arxiv.org/abs/${r.id}" target="_blank">arXiv</a><a class="pill" href="https://arxiv.org/pdf/${r.id}" target="_blank">PDF</a>${r.repo?`<a class="pill" href="${esc(r.repo)}" target="_blank">Code</a>`:''}${r.orv?`<a class="pill" href="${esc(r.orv)}" target="_blank" title="reviews, and where to ask the authors">OpenReview</a>`:''}<a class="pill" href="${esc(r.rec)}" target="_blank" title="this paper's record in the organisers' dataset, with their picture of the claimed table or figure">Official record</a></div>
 <div class="tiles">${tiles}</div>
 <h4>In your way${I.length?` (${I.length})`:''}</h4>${I.length?I.map(([t,w])=>`<div class="issue"><b>${t}</b><div>${w}</div></div>`).join(''):'<div class="issue ok"><b>Nothing was flagged</b><div>Not a guarantee: only the checks described in the README were made.</div></div>'}
 <h4>The claim <span class="mut" style="text-transform:none;letter-spacing:0">— copied word for word from the organisers' dataset</span></h4>
 <dl class="dl"><dt>Where</dt><dd><b>${esc(r.src)}</b>${okLoc||r.loc.startsWith('not checked')?'':` <span class="bad">— ${esc(r.loc)}; go by the picture in the official record</span>`}</dd>
  <dt>Claim</dt><dd>${esc(r.claim)}</dd><dt>Metric</dt><dd>${esc(r.metric)||'–'}</dd><dt>Compared against</dt><dd>${esc(r.cmp)||'–'}</dd><dt>Falsified if</dt><dd>${esc(r.crit)||'–'}</dd>
  <dt>Checkable</dt><dd>${r.ck}/10 <span class="mut">${esc(r.ckw.join(' · '))||'no adjustments'}</span></dd></dl>
 <details class="sec"${scanned?' open':''}><summary>Code and links found in the paper (${nLinks})</summary>
  <div class="sub">Code repository</div>${repoLine}
  ${other.length?`<div class="sub">Other GitHub links in the PDF</div>${lnk(other,'')}`:''}
  <div class="sub">Hugging Face links (models, datasets, demos)</div>${lnk(r.hf,'none found')}
  ${r.pp.length||r.anon.length?`<div class="sub">Project pages / anonymous code</div>${lnk(r.pp.concat(r.anon),'')}`:''}
  <div class="sub">Search elsewhere</div><div class="kv"><a class="pill" href="https://github.com/search?q=${q}&type=repositories" target="_blank">GitHub by title</a><a class="pill" href="https://huggingface.co/search/full-text?q=${q}" target="_blank">Hugging Face by title</a>${r.pwc?`<a class="pill" href="${esc(r.pwc)}" target="_blank">Papers with Code</a>`:''}</div></details>
 ${scanned?`<details class="sec"><summary>Repository scan — turnkey ${r.tk}/10</summary>
  <dl class="dl"><dt>Files</dt><dd>${r.npy} Python files · dependency files: ${esc(r.deps.join(', ')||'none')}</dd><dt>Licence</dt><dd>${esc(r.lic||'none found')}</dd><dt>Tests</dt><dd>${r.tests?'yes':'none found'}</dd></dl>
  <div class="sub">Good signs (+1 each)</div>${ev(r.boost,'good')}<div class="sub">Friction</div>${ev(r.soft,'')}<div class="sub">Blockers (score set to 0)</div>${ev(r.hard,'bad')}
  <div class="sub">Building blocks for award entries (not scored)</div>${ev(r.facts,'sem')}${NK(r.ign)?`<div class="sub">Keyword matches judged harmless and ignored</div>${ev(r.ign,'mut')}`:''}</details>`:''}
 <details class="sec"><summary>Paper details</summary>
  <dl class="dl"><dt>Length</dt><dd>${r.pages??'?'} pages (${r.mb??'?'} main body)${r.refs!=null?` · about ${r.refs} references`:''}</dd>
  <dt>Reading time</dt><dd>${r.mins!=null?'~'+Math.round(r.mins)+' min for abstract + method + setup':'unknown'}</dd>
  <dt>Setup in appendix</dt><dd>mentions hyperparameters ${am.hyperparameter||0}×, prompt templates ${am.prompt_template||0}×, seeds ${am.seeds||0}× <span class="mut">(more usually means the setup is written down)</span></dd>
  <dt>Hardware</dt><dd>${r.hw.length?r.hw.map(h=>`<code>…${esc(h)}…</code>`).join('<br>'):'<span class="mut">not stated in the paper (not the same as cheap)</span>'}</dd></dl></details>
 <h4>Most similar papers <span class="mut" style="text-transform:none;letter-spacing:0;cursor:help" title="${SIMNOTE}">— click to jump · what is “sim”? ⓘ</span></h4><ul>${r.nb.map(([n,sm])=>{const m=D[IDX[n]];return `<li data-go="${n}">${esc(m.t)} <span class="mut">sim ${sm} · my #${m.rank}</span></li>`}).join('')}</ul>`;
 render();if(scroll){const tr=document.querySelector(`tr[data-id="${id}"]`);tr&&tr.scrollIntoView({block:'center',inline:'nearest'})}}
// ---- compare (up to 5)
function cosP(i,j){let s=0;for(let k=0;k<DIM;k++)s+=E[i*DIM+k]*E[j*DIM+k];return s/16129}
function compare(){const P=S.short.map(id=>D[IDX[id]]);$('#card').style.display='none';const c=$('#cmp');c.style.display='block';c.scrollTop=0;const ORG=' <span class="mut" style="font-size:10px">(organisers)</span>';
 if(P.length<2){c.innerHTML='<button class="x" data-closecmp="1">✕ back to list</button><h3>Compare papers</h3><p class="mut">Add at least two papers to your shortlist with the + button, then compare them here. Up to five — the hackathon allows five papers per person.</p>';return}
 const keys=o=>NK(o)?Object.keys(o).map(nice).join(', '):'<span class="mut">none found</span>',ns='<span class="mut">no repository read</span>',sc=r=>r.rs==='cloned';
 const ROWS=[['Official labels',r=>badges(r)],['Automatic topics / areas',r=>autos(r)],
  ['Claim to verify',r=>`<b>${esc(r.src)}</b><br>${esc(r.claim)}<br>${claimLinks(r)}`],['Metric',r=>esc(r.metric)],['Compared against',r=>esc(r.cmp)||'–'],['Counts as falsified if',r=>esc(r.crit)||'–'],
  ['Can I tell if I matched it?',r=>`Checkable ${r.ck}/10<br><span class="mut">${esc(r.ckw.join(' · '))}</span>`],
  ['Is there code?',r=>r.repo?`<a href="${esc(r.repo)}" target="_blank">repo</a> — ${r.rs==='cloned'?'turnkey '+r.tk+'/10, last commit '+r.rdate:'<span class="bad">'+esc(nice(r.rs))+'</span>'}`:'<span class="bad">no repository found</span> — see links in details'],
  ['What helps',r=>sc(r)?`<span class="good">${keys(r.boost)}</span>`:ns],['What slows you down',r=>sc(r)?keys(r.soft):ns],['What stops you',r=>sc(r)?`<span class="bad">${keys(r.hard)}</span>`:ns],
  ['What stands in your way',r=>{const I=issues(r);return I.length?'<ul style="margin:0;padding-left:15px">'+I.map(i=>`<li style="cursor:default">${i[0]}</li>`).join('')+'</ul>':'<span class="good">nothing flagged</span>'}],
  ['Hardware the paper mentions',r=>r.hw.length?`<code>…${esc(r.hw[0].slice(0,150))}…</code>`:'<span class="mut">not stated</span>'],
  ['Reading load',r=>`${r.pages??'?'} pages · ~${r.mins!=null?Math.round(r.mins):'?'} min · about ${r.refs??'?'} references${r.capp?' · <span class="bad">claim is in the appendix</span>':''}`],
  ['Setup written down?',r=>{const a=r.appm||{};return `appendix mentions hyperparameters ${a.hyperparameter||0}×, prompts ${a.prompt_template||0}×, seeds ${a.seeds||0}×`}],
  ['Released weights / data on Hugging Face',r=>r.hf.length?r.hf.length+' link(s) in the PDF':'<span class="mut">none found</span>'],
  ['Building blocks for award entries',r=>sc(r)?[r.facts.demo_code&&'demo code (Gradio / Streamlit)',r.facts.environment_code&&'environment code (Gym-style)',r.facts.notebook_or_colab&&'Colab notebook',r.lic&&'licence: '+r.lic,r.tests&&'has tests'].filter(Boolean).join('; ')||'<span class="mut">none found</span>':ns],
  ['Overlap with the rest of your shortlist',(r,i)=>{let b=-1,bs=-1;P.forEach((o,j)=>{if(j!==i){const s=cosP(IDX[r.id],IDX[o.id]);if(s>bs){bs=s;b=j}}});return `closest: #${b+1} (sim ${bs.toFixed(2)})${bs>.85?' <span class="bad">— very similar topic</span>':bs<.72?' <span class="good">— different area</span>':''}`}],
  ['My score',r=>`${bar(r)}<b>${r.score}</b> (#${r.rank})`]];
 c.innerHTML=`<button class="x" data-closecmp="1">✕ back to list</button><h3>Compare ${P.length} shortlisted papers</h3><div class="mut">Rows are facts to weigh, not a verdict. Nothing is highlighted as "best": which row matters depends on your time, hardware and the award you are aiming for. ${SIMNOTE}</div>
 <div style="overflow:auto;margin-top:6px"><table id="cmpt"><tr><th></th>${P.map((r,i)=>`<th>#${i+1} <a href="https://arxiv.org/abs/${r.id}" target="_blank">${r.id}</a><br>${esc(r.t)}<br><button data-open="${r.id}">details</button> <button data-short="${r.id}">remove</button></th>`).join('')}</tr>
 ${ROWS.map(([n,f])=>`<tr><td>${n}${['Official labels','Claim to verify','Metric','Compared against','Counts as falsified if'].includes(n)?ORG:''}</td>${P.map((r,i)=>`<td>${f(r,i)}</td>`).join('')}</tr>`).join('')}</table></div>`}
// ---- controls
$('#cons').innerHTML=CONS.map(c=>`<label><input type="checkbox" id="c_${c[0]}"> ${c[1]} <span class="mut">(${D.filter(c[2]).length})</span></label>`).join('');CONS.forEach(c=>$('#c_'+c[0]).onchange=e=>{S.cons[c[0]]=e.target.checked;render()});
$('#sl').innerHTML=W.map(([k,l,col])=>`<label><span><span class="sw" style="background:${col}"></span>${l}</span><span><input type="range" min="0" max="10" step="1" id="w_${k}"> <b id="v_${k}"></b></span></label>`).join('');
W.forEach(([k],i)=>$('#w_'+k).oninput=e=>{S.w[i]=+e.target.value;render()});
document.querySelectorAll('[data-pre]').forEach(b=>b.onclick=()=>{S.w=PRE[b.dataset.pre].slice();render()});
['fTop','fTrk'].forEach(i=>$('#'+i).onchange=render);$('#q').oninput=render;$('#mfirst').onchange=render;$('#c_match').onchange=render;
let tmr;$('#int').oninput=()=>{typing();clearTimeout(tmr);tmr=setTimeout(render,160)};$('#int').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();render();$('#tw').scrollTop=0}};
$('#fCol').onchange=()=>{S.sem=null;render()};$('#bLinks').onclick=()=>{S.links=!S.links;$('#bLinks').classList.toggle('on');draw()};
$('#hd').onclick=e=>{const k=e.target.dataset.k;if(!k||k==='pin'||k==='labels')return;S.dir=S.sort===k?-S.dir:(k==='rank'||k==='t'||k==='mins'?1:-1);S.sort=k;render()};
function toggleShort(id){if(S.short.includes(id))S.short=S.short.filter(x=>x!==id);else if(S.short.length<5)S.short.push(id);else{say('shortlist is full (5) — remove one first');return}if(!S.short.length)S.only=false}
function toggleLike(id){S.likes.includes(id)?S.likes=S.likes.filter(x=>x!==id):S.likes.length<12&&S.likes.push(id)}
document.addEventListener('click',e=>{const t=e.target,d=t.dataset||{};
 if(d.rm){toggleShort(d.rm);render();if($('#cmp').style.display==='block')compare();e.stopPropagation();return}
 if(d.short){toggleShort(d.short);if($('#cmp').style.display==='block'){render();compare()}else if($('#card').style.display==='block'&&S.sel)select(S.sel,false);else render();return}
 if(d.like){toggleLike(d.like);if($('#card').style.display==='block'&&S.sel)select(S.sel,false);else render();return}
 if(d.close){closeCard();return}if(d.closecmp){$('#cmp').style.display='none';return}
 if(d.comp){setLast(d.comp);return}if(d.add){$('#int').value=$('#int').value.replace(/[,\s]*$/,'')+', '+d.add;render();return}
 const op=t.closest('[data-open]');if(op){select(op.dataset.open,false);return}const go=t.closest('[data-go]');if(go){select(go.dataset.go,false);return}
 const tr=t.closest('#tb tr[data-id]');if(tr&&t.tagName!=='A'&&t.tagName!=='BUTTON')select(tr.dataset.id,false)});
const say=t=>{$('#msg').textContent=t;const o=$('#mstat').innerHTML;$('#mstat').innerHTML='<span class="bad">'+esc(t)+'</span>';setTimeout(()=>{$('#msg').textContent='';render()},2200)};
$('#bCmp').onclick=compare;$('#bOnly').onclick=()=>{if(!S.short.length){say('shortlist is empty');return}S.only=!S.only;render()};
$('#share').onclick=()=>{save();(navigator.clipboard?navigator.clipboard.writeText(location.href):Promise.reject()).then(()=>say('link copied'),()=>say('copy the address bar'))};
$('#reset').onclick=()=>{S.w=PRE.bal.slice();S.likes=[];S.short=[];S.cons={};S.only=false;$('#int').value='';$('#q').value='';$('#c_match').checked=false;$('#fTop').value='';$('#fTrk').value='';$('#sugg').innerHTML='';render()};
$('#csv').onclick=()=>{const q=v=>'"'+String(v??'').replace(/"/g,'""')+'"',rows=D.filter(r=>pass(r)).sort((a,b)=>a.rank-b.rank);const t=['my_rank,my_score,arxiv_id,title,presentation,track,difficulty,auto_topics,turnkey,checkability,read_mins,interest_match,on_shortlist,repo'].concat(rows.map(r=>[r.rank,r.score,r.id,q(r.t),r.pres,r.trk,r.diff,q(r.tops.join('; ')),r.tk,r.ck,r.mins,r.match==null?'':r.match.toFixed(2),S.short.includes(r.id),r.repo||''].join(','))).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{type:'text/csv'}));a.download='neurips2026_my_ranking.csv';a.click()};
if(window.innerWidth<=820)$('#legend').classList.add('min');
load();render();
</script></body></html>"""
OUT = ROOT / "index.html"
html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
            .replace("__VOC__", json.dumps(VOC, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
            .replace("__ES__", b64(ES)).replace("__TV__", b64(TV)).replace("__DIM__", str(ES.shape[1])))
OUT.write_text(html)
print(f"wrote {OUT.name}: {len(data)} papers, {len(VOC)} search terms, {OUT.stat().st_size/1e6:.2f} MB")
