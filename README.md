# NeurIPS 2026 Reproduction Paper Map

An evidence map of the papers in the [NeurIPS 2026 ML Intern reproduction hackathon claim list](https://huggingface.co/datasets/Neurips-2026-MLIntern-Repro/papers), built to answer one question before you commit a week to a paper: **what am I stepping into?**

For each of the 1,046 papers it shows what the paper links to, what its repository contains, and what is likely to get in your way — with the file and line behind every finding.

- **Open the map:** https://huggingface.co/spaces/gitgithan/neurips2026-repro-map (works in any browser; no account, no install, nothing is sent anywhere)
- **Or offline:** download `index.html` from this repository and open it.
- **Or as data:** the [evidence dataset on the Hugging Face Hub](https://huggingface.co/datasets/gitgithan/neurips2026-repro-map-evidence) (browse it as a table, or `load_dataset("gitgithan/neurips2026-repro-map-evidence")`). The same files are in this repository as `data/papers_evidence.csv` and `data/papers_evidence.json`.

## What this is, and what it is not

- It **reads**; it does not run. Scripts read each paper's PDF and a shallow copy of its GitHub repository. No code from any paper was executed to produce these numbers.
- It **does not rank papers for you**. The page opens with equal weights. You set the weights, type what you are interested in, and star papers you like; the order is recomputed in your browser.
- It **does not predict reproducibility**. Nobody has measured how well these scores predict whether a paper reproduces. Treat a score as a pointer to evidence worth reading, not a verdict. A low score often means "expect setup work", and someone with the hardware or the patience may be exactly the right person for that paper.

## Using the page

The page is laid out as four steps.

| Step | What you do | What happens |
|---|---|---|
| 1 · Your interests | Type words or phrases separated by commas (`healthcare, brain decoding, LoRA`), or a whole sentence (`I want something on tutoring systems that trains on one small GPU`). Suggestions appear as you type | The list re-orders live. A line under the box says, for each word, how many papers match **exactly** and how many are **related by meaning** |
| 2 · Hide papers where… | Tick the things you cannot work around: no repository, a blocker, gated access, a hosted model, a native build, manual downloads, "harder than most", claim in the appendix, posters | Those papers disappear. Each switch shows how many papers it hides. These are yes/no facts, not preferences |
| 3 · Read the evidence | Click a dot or a row | A panel lists every link, every scan finding with its file and line, hardware sentences, and the most similar papers |
| 4 · Shortlist and compare | Press **+** on up to five papers (the hackathon limit), then "Compare side by side" | A table with one column per paper and one row per question you would ask before committing |
| 5 · Ask ML Intern | Press **Ask ML Intern ↗** next to the shortlist | A question is written from your interests exactly as you typed them, the hide-switches you ticked, and your shortlisted papers (your top five if the shortlist is empty): each paper's claim, falsification criterion, repository and flagged issues. It is copied to your clipboard and [ML Intern in HuggingChat](https://huggingface.co/chat/?mode=ml-intern) opens in a new tab, under your own Hugging Face account. If the chat box is empty, paste. Details: [Ask ML Intern](#ask-ml-intern) |

Also on the page:

- **Similarity lines** are off by default; the button in the header turns them on.
- **Weights for "my score"** (collapsed by default): six sliders. The coloured bar next to each score shows how much each one contributed. There is no built-in ranking; the default is equal weights.
- **☆ "more like this"**: ranks papers with similar text higher. Separate from the shortlist.
- **Suggestions**: "Close to what you asked for", and "Something different" — runnable papers from topics your interests do not touch, for people who want to explore.
- **Map**: scroll to zoom, drag to pan, hover for a title, double-click to reset. Papers matching your interests get a white ring; shortlisted papers a square. Click a legend entry to isolate a group.
- **Resizing**: drag the bar between the map and the right-hand side, or the bar above the table.
- **Sharing**: "Copy link to these settings" stores interests, switches, weights, stars and shortlist in the URL.
- It works on a phone, with the map above the list.

### Where each label comes from

| Label | Who assigned it | How it reached the page |
|---|---|---|
| Oral / spotlight / poster, track | The NeurIPS 2026 programme committee | Copied unchanged from the hackathon's official claim list |
| Standard / "harder than most", the claim to verify, table-or-figure, metric | The hackathon organisers | Copied unchanged from the official claim list (hosted on the Hugging Face Hub) |
| Topics and areas (shown dashed and italic, prefixed "Auto") | Nobody — computed by this tool. Topics come from clustering the papers' text; areas from a hand-written list of field names in `search_glossary.toml` | Automatic and rough. **A paper can carry several topics and areas, or no area at all.** Details under "Where the topics and areas come from" |
| Turnkey, checkable, read time, match, my score | Nobody — heuristics of this tool | Rules are below |

Nothing comes from the authors' own keywords (the claim list has none) or from Papers with Code. That site has its own categories for NeurIPS 2026 papers; they are not used here because the site does not permit automated collection. Only the link to each paper's page is shown.

## What you get for each paper

Every paper has one record in `data/papers_evidence.json`. The page's detail panel shows the same content.

| Section | Fields | Where it comes from |
|---|---|---|
| `labels` | oral / spotlight / poster, track, difficulty, table-or-figure claim | The official claim list, unchanged |
| `claim_source` | Where the organisers say the claim is, e.g. "Table 2, page 6" | The organisers' dataset |
| `official` | Authors, background and method summaries, the claim to verify and its caption, the metric, what it is compared against, and the falsification criterion (what result would count as overturning the claim), with the dataset revision they were copied from | The organisers' dataset, unchanged |
| `links.repo` | The repository judged to be the paper's own, how it was found (PDF or project page), confidence, whether it is alive, the commit that was scanned | PDF link extraction + an HTTP check + a shallow clone |
| `links.github_in_pdf`, `huggingface_in_pdf`, `project_pages`, `anonymous_code` | Every other link of that kind in the PDF, with page numbers, surrounding text, and whether it looks like the authors' own or cited third-party work | PDF link extraction |
| `links.openreview`, `arxiv`, `papers_with_code` | Where to read reviews and reach the authors | The official claim list |
| `paper` | Page counts, the page the claim is on, whether it is in the appendix, estimated reading time, sentences that mention hardware, whether the text names a hosted model | PDF text |
| `repo_scan` | File counts, dependency files, and three lists — good signs, friction, blockers — each entry with the `file:line: text` that triggered it; plus keyword matches that were judged harmless | Reading the cloned files |
| `scores` | `turnkey`, `checkability`, `readability`, each with its basis | Heuristics below |
| `auto_labels` | `topics` (one to three) and `areas` (zero to three), with a `source` note | Computed by this tool: text clustering, and closeness to the glossary descriptions |
| `map` | 2D position, main topic, 4 most similar papers with their similarity | Text embedding of title + claim |

**When a paper shows "no repo found"** (474 papers), it means no GitHub link was found in the PDF or on its project page. The code may still exist. The detail panel gives you the Hugging Face links and project pages we did see, one-click searches for the title on GitHub and Hugging Face, and the OpenReview page, where authors can be asked directly. Author e-mail addresses are deliberately not collected.

## The two data files

`data/papers_evidence.json` and `data/papers_evidence.csv` describe the same 1,046 papers and cannot disagree.

- **The JSON is the full record**: every link with its page and context, every scan finding with the quoted `file:line`, the map position and neighbours.
- **The CSV is a flat view of the JSON.** Each of its 47 columns is computed from one JSON record by a fixed rule (a value, a count, or a list joined with `;`). The rules are listed in `data/csv_columns.json` as `csv_column → json_path`.
- **This is checked on every build.** After writing both files, the pipeline reads them back and compares every cell of the CSV with the value computed from the JSON; the build fails if any differ.
- **Columns starting with `official_` are the organisers', not ours.** Authors, the background and method summaries, the claim and its caption, the metric, the comparison and the falsification criterion (the `official` block in the JSON) are copied unchanged from the organisers' dataset, at the revision recorded in `official.dataset_revision`. With them, the files hold every text column of the organisers' dataset plus what this project adds, so one file is enough. Only the organisers' picture of each claimed table or figure is not copied; `official_claim_record` links to it. The organisers' dataset is the authority: if it is corrected later, these copies are out of date until the pipeline is re-run.

The page (`index.html`) is built from the JSON alone and contains the same text, since a page can only show what it holds.

## Reading the table: every column, every blank

`data/papers_evidence.csv` has one row per paper. This section says what each column means, how it was measured, why it can be empty, and what you can do about it. The JSON file holds the same information in full, with the quoted evidence (see "The two data files").

**A blank or a low number is a statement about what a script could see, not about what exists.** Many gaps can be closed by a person: following a project page, signing in to a gated dataset, or asking the authors.

### From the official claim list (never blank)

| Column | Meaning | How to use it |
|---|---|---|
| `arxiv_id`, `title` | The paper | |
| `claim_source` | Where the claim is in the paper, e.g. "Table 2, page 6" | Open that page first |
| `official_authors` | The paper's authors | Recognise a group whose code you trust; know who to ask |
| `official_background`, `official_method` | The organisers' plain-language summary of the problem and of what the paper does | The quickest way to decide whether a paper is in your area before opening the PDF |
| `official_claim` | The result you are asked to reproduce, in the organisers' words | This, not the abstract, is what your verdict is about |
| `official_claim_caption` | The caption of the claimed table or figure | Confirms you are looking at the right table |
| `official_metric` | What is measured and how it is reported (seeds, spread) | Tells you what number to produce and what tolerance is fair |
| `official_comparison` | What the method is compared against | Tells you which baselines you would have to run as well |
| `official_falsification_criterion` | What result would count as overturning the claim | Read it before choosing: it often reveals the real cost, such as "after retraining" |
| `presentation` | `oral`, `spotlight` or `poster` | The conference's own selection; it says nothing about how easy the paper is to reproduce |
| `track` | `main`, `evaluations_datasets`, or a journal track | Benchmark and dataset papers often need data access more than GPUs |
| `difficulty` | `standard` or `harder` — the organisers' label for claims that are harder than most to reproduce | Not a reason to avoid a paper if you have the resources; a reason to read its evidence first |
| `claim_kind` | `table` or `figure` | See `checkability` |

### Did we find the code?

| Column | Meaning | Blank / special values | What you can do |
|---|---|---|---|
| `repo_url` | The GitHub repository judged to be the paper's own | **Blank for 474 papers**: no GitHub link was found in the PDF or on a `github.io` project page | The code may still exist. Check `project_page`, `huggingface_links_in_pdf` and `other_github_links_in_pdf` (149 of these papers have a Hugging Face link or project page; 152 link some other GitHub repository). Then search the title on GitHub and Hugging Face, look at the OpenReview page, or ask the authors there |
| `repo_status` | `cloned` (555): alive and scanned. `dead` (17): the link in the PDF returns 404. `no_link_found` (474) | Never blank | For `dead`: the repository may have been renamed, moved to an organisation, or made private. Search for the name, or ask the authors |
| `other_github_links_in_pdf` | How many other GitHub repositories the PDF links | 0 = none | Usually baselines and tools the paper depends on. The JSON lists each with its page and surrounding sentence; sometimes the paper's real code is one of them |
| `huggingface_links_in_pdf` | How many Hugging Face links (models, datasets, demos) the PDF contains | 0 = none | Released weights or data often live here even when there is no code repository |
| `project_page` | The paper's project page, if the PDF links one | Blank for 839 papers: none linked | Project pages often carry the newest links to code, data and demos |
| `openreview` | The paper's OpenReview page | Blank for 12 papers (journal-track papers without one) | Reviews often discuss reproducibility; signed-in users can post a comment to reach the authors |

### What the repository looks like

All of these are `not scanned` (or blank, for `turnkey`) when there was no live repository to read. That means "unknown", not "bad".

| Column | Meaning and measurement | Blank means | How to decide with it |
|---|---|---|---|
| `turnkey` | 0–10: how runnable the repository looks from its files. Starts at 3, +1 per good sign, −1 or −2 per friction; 0 if there is a blocker. Full rules under "The heuristics" | No repository was scanned (491 papers). On the page these count as 0 in "my score"; use the "Has repo" filter if you do not want them mixed in | 7 or more: the entry point, the environment and the downloads are probably all spelled out. 3–6: expect to write some glue. 0: read `blockers` — one specific thing stands in the way |
| `turnkey_basis` | `repo scan`, `no repo link found`, or `dead` | Never blank | Tells you whether `turnkey` is a measurement or an absence |
| `good_signs` | Which positive signals were found (for example `load_dataset(`, `from_pretrained(`, `Dockerfile`, `eval.py`, `reproduce.sh`, `uv.lock`) | Blank = scanned, none found. `not scanned` = no repository | None found does not mean broken: plenty of working research code has none of these. It means you will be reading the README to find the entry point |
| `friction` | Things that cost time but can be worked through: `google_drive_download`, `baidu_pan_download`, `native_build`, `gated_access`, `partial_release_pending`, `yt-dlp`, `youtube.com`, `ffmpeg` | Blank = scanned, none found | `gated_access`: you can usually get in by signing in and accepting terms, or by filling in a form — do it on day one, approval can take days. `google_drive_download` / `baidu_pan_download`: budget time for manual downloads. `native_build`: check your CUDA and compiler versions against the repository's first. `partial_release_pending`: read the README to see which part is missing — it may not be the part your claim needs. Video tooling: check the videos still exist |
| `licence` | Licence detected from a top-level `LICENSE` file (`MIT`, `Apache-2.0`, `GPL`, `CC BY-NC`, …). `none found` = scanned, no licence file | No licence means you cannot assume you may redistribute a modified copy — relevant if you plan to publish a derived tool, model or dataset. Ask the authors |
| `has_tests`, `demo_code`, `environment_code` | Whether the repository has a tests folder, imports Gradio or Streamlit, or imports a Gym-style environment library | Not scored. They are building blocks: an existing demo shortens the way to an app or an explainer; environment code shortens the way to an RL environment |
| `repo_last_commit` | Date of the commit that was read | An old date on a 2026 paper can mean the code predates the final paper |
| `blockers` | The reason `turnkey` is 0: `no_code_files` (the repository is a placeholder), `code_not_released` (the README says the code is coming), `closed_api_target` (the method under test is a hosted closed model) | Blank = scanned, no blocker | `no_code_files` / `code_not_released`: check the repository again — it was read at one commit and may have been filled in since — or ask the authors for a timeline. `closed_api_target`: reproducible only with paid API access to the same model version, which may have changed |

### How hard is the claim to check and to read?

| Column | Meaning and measurement | How to decide with it |
|---|---|---|
| `checkability` | 0–10: how cleanly you can tell whether your result matches. Starts at 4; +2 if the claim is a table; +2 if the metric reports a spread or a number of runs; +1 if the claim sentence states numbers; −2 if it aggregates over five or more benchmarks; −1 if the result is in the appendix. Never blank | High: you will get a clear yes or no. Low: decide beforehand what "matched" will mean — which benchmarks you will run, and what tolerance you accept |
| `checkability_why` | The adjustments that were applied, in words | Blank = none applied (score 4) |
| `readability` | 0–10. Starts at 10 and falls with reading time (5 minutes costs nothing, 45 minutes costs 6 points); −0.75 for two or more theorems in the main body, −1.5 for five or more; −1 if the claim sits deep in the appendix (past page 20); +0.5 or +1 when a long appendix documents hyperparameters or prompts. Never blank | A rough guide to how long it takes before you can start running anything. Low is not bad: theory-heavy papers score low and may still have trivial experiments |
| `read_mins` | Estimated minutes to read the abstract, method and experimental setup: words ÷ 200 per minute, up to +50% for equation-dense papers, clamped to 5–45. For 20 papers whose sections could not be detected it is estimated from the main-body length | Use it to budget your first sitting with a paper |
| `pages` | Total PDF pages, appendix included | |
| `code_statements` | Which kinds of statement about code the paper makes: `in_supplementary_material`, `promised_for_later`, `on_request`, `on_openreview`. Blank = none found. The quoted sentences are in the JSON | See "Blockers, and ways round them". Most useful for papers with no repository |
| `claim_location_check` | Whether the table or figure label named by the official list was found on the stated page of the arXiv PDF (see "Where the claimed table or figure is in the paper") | If it is not "found on the stated page", open `official_claim_record` and go by the picture there |
| `official_claim_record` | Link to this paper's row in the official claim list, with the picture of the claim | The authority for what you must reproduce |
| `claim_page`, `claim_in_appendix` | Where the claimed table or figure is (225 claims are in an appendix) | An appendix claim usually means the setup details are in the appendix too — read that section, not just the main text |
| `names_hosted_model` | `True` (204 papers) if the claim text names a hosted model such as GPT, Claude or Gemini | Check whether it is the method under test (you will need API access), only a judge (you may be able to substitute an open model, at the cost of comparability), or just a baseline you can skip |
| `hardware_mentions` | How many sentences in the PDF mention GPUs or GPU-hours (0 for 305 papers) | 0 means the paper does not say, not that it is cheap. The sentences are quoted in the JSON and on the page; read them — they may describe a baseline rather than the method |
| `reference_count` | Approximate number of references: the highest `[n]` in a numbered list, otherwise the number of reference lines ending in a year. Blank when the reference list was not found | A weak signal. A long list often means a benchmark or survey-style paper with many baselines to understand; it says nothing about whether the code runs |
| `auto_topics`, `auto_areas` | Automatic labels, several per paper possible. Topics come from clustering all papers; areas are glossary terms whose description is unusually close to the paper. `auto_areas` is blank when nothing was close enough | Rough. Use them to browse, and the interest box to search |

## How it was built

Four steps, plain Python, no API keys and no language-model calls.

1. **`pipeline/1_fetch_claims.py`** downloads the claim list (pinned to one dataset revision).
2. **`pipeline/2_scan_pdfs_and_repos.py`** runs in stages:
   - `download`: fetch each arXiv PDF, slowly (about one request every 3 seconds).
   - `parse`: extract page structure, section word counts, and every link with its page and surrounding text.
   - `pages`: for papers with a project page but no GitHub link, read the project page for a repository link by the same owner.
   - `head`: check each repository link is alive.
   - `clone`: shallow partial clone — latest commit only, files over 2 MB skipped.
   - `scan`: match the patterns in `signals.toml` against the cloned files. Read-only.
   - `merge`: one row per paper.
3. **`pipeline/3_build_evidence.py`** assembles the evidence records, computes the three scores, embeds title + claim, lays out the map, assigns automatic topics and areas, and builds the search vocabulary.
4. **`pipeline/4_build_map.py`** writes `index.html`.

Coverage of this snapshot: 1,046 papers; a repository link was found for 572; 555 were alive and scanned; 17 links were dead; 474 had no link.

## The heuristics

All patterns and weights are in [`signals.toml`](signals.toml).

### Choosing the paper's own repository

A PDF usually links several GitHub repositories, most of them baselines and tools. Each link is scored on where it appears (first page, abstract or a "code is available at" sentence score high; the reference list scores low), whether the repository name resembles the paper's title, and whether the owner is a well-known infrastructure organisation. The top link is used only above a threshold and is reported with a confidence. All the others stay visible under "Other GitHub links in the PDF".

*When it is wrong:* a paper that releases code under an unrelated repository name, or links it only from a project page that is not on `github.io`, shows as "no repo found".

### Turnkey (0–10): how runnable the repository looks

Starts at **3**.

| Signal | Effect | Why |
|---|---|---|
| `load_dataset(`, `from_pretrained(`, a `huggingface.co/` link in code or docs | +1 each | Data and weights download themselves |
| `uv.lock`, a Dockerfile, an eval script, a `reproduce.sh`-style script | +1 each | The environment or the entry point is pinned down |
| Google Drive download | −1 | Manual or rate-limited downloads |
| Baidu Pan download | −1 to −2 | Hard to use outside China; −1 if a Drive alternative exists |
| C++/CUDA extension to build | −1 to −2 | Compiler and CUDA version friction; −1 if only in a nested folder |
| "request access" / form / agreement in top-level docs | −2 | A person has to approve you |
| Video scraping tools (`yt-dlp`, YouTube links, `ffmpeg`) | −1 each | Data may have disappeared |
| No dependency file; README under 500 characters; closed-API key referenced | −1 each | |

The score is set to **0** in only three cases: the repository has no code files; its main README says the code is not released yet; or the method under test is itself a hosted closed model.

### How a keyword becomes a blocker (and mostly does not)

A keyword match on its own is weak evidence, and the scan is built around that:

- **"Coming soon" rarely means the code is missing.** In this corpus the phrase matched in 44 repositories. In 8 of them the main README says the code itself is not released, and those repositories contain no Python files. In the rest it refers to something else: a future feature of a bundled library, a paper link, a citation, one optional checkpoint. So the rule fires only when the phrase sits next to "code" or "implementation" in the top-level README *and* the repository has almost no code. A pending partial release (weights, training code) costs 1 point instead.
- **Nested folders are usually someone else's code.** Repositories often bundle whole copies of other projects, and their documentation matches everything. Access requirements and build steps count fully only at the top level.
- **"TODO:" in documentation says nothing about runnability** and is ignored.
- **A download link is friction, not a wall.** A Google Drive link costs a point; it does not zero the score.

Every match that was judged harmless is still listed in the record under `ignored_matches`, so you can disagree.

### Checkable (0–10): can you tell whether you matched the claim?

Starts at 4. +2 if the claim is a table (exact numbers rather than a curve to read off). +2 if the metric text reports a spread or a number of runs (you know the tolerance). +1 if the claim sentence states numbers. −2 if the claim aggregates over five or more benchmarks. −1 if the claimed result is in the appendix. The reasons are stored in `checkability_why`.

### Reading time and readability

Reading time = words in the abstract, method and experimental-setup sections ÷ 200 words per minute, increased by up to 50% for equation-dense papers, clamped to 5–45 minutes. When sections cannot be detected, the main-body word count is scaled by the corpus-median ratio, and `read_basis` says so. Readability (0–10) falls with reading time and with theorem count, and rises slightly when a long appendix documents hyperparameters or prompts.

### Hardware mentions

There is no compute score. Sentences from the PDF that mention GPUs, GPU-hours or specific accelerators are shown verbatim, because a keyword-based "how expensive is this" number was not reliable enough to publish, and because what counts as expensive depends on what you have.

### Interest match

Each word or phrase you type is matched in two ways.

- **Exact**: the word appears at the start of a word in the title (100%) or in the claim, method or background text (70%); several hits add a little.
- **Related by meaning** (up to 60%): the paper's text is unusually close to your word in the embedding space — more than 2.5 standard deviations above the average paper for that word. This runs without any model in your browser: a vector for each of about 6,000 search terms was computed when the page was built. Phrases that are not in that vocabulary are composed from their words.

**Sentences.** An entry of four or more words is treated as a sentence, not one phrase. Filler words (`I`, `want`, `something`, `that`…) are dropped, known two-word phrases are kept together, and any remaining word that appears in more than 30 papers (`small`, `systems`, `weights`) is ignored because it says little about what you want — unless it is a glossary term. The status line under the box shows exactly which words were used. The word or known phrase straight after `not`, `no`, `without` or `except` is left out (`brain decoding but not vision` searches for brain decoding only). No language model reads the sentence, so anything subtler — a condition such as `on one GPU`, or "something like X but cheaper" — is not understood: use the hide-switches and weights for hard limits, and the Ask ML Intern button, which passes your sentence on word for word.

Field names and jargon that papers rarely use literally (`edtech`, `healthcare`, `bci`, `rag`…) are listed in [`search_glossary.toml`](search_glossary.toml) with a plain-language description, and are searched by that description. Add your own terms there.

*When it is wrong:* "related" is approximate. If the collection has few papers on a subject, the nearest ones may be only loosely connected; their percentages will be low. The status line always tells you how many matches were exact.

With "list matching papers first" ticked, exact matches come first, then related ones (closest first), then everything else; within a group the order is your score.

### "My score"

The weighted average of the six components you gave weight to, scaled to 0–100. Papers with no scanned repository count as 0 on turnkey. The coloured bar shows each component's share.

### Similarity

The "sim" numbers, the lines on the map and "more like this" all use the cosine similarity between text embeddings of two papers (title + claim). In this collection two random papers score about 0.68, and a paper's four nearest neighbours typically 0.80–0.90. High similarity means similar wording and subject. It does not mean shared code, data or authors. In the compare view, a shortlist whose papers are all very similar (above 0.85) covers one area several times.

### Suggestions

"Close to what you asked for" lists the best matches not yet on your shortlist. "Something different" takes topics that none of your top matches belong to and offers, from each, the highest-scoring paper (ignoring the interest weight) that has a scanned repository with no blocker.

### Ask ML Intern

The button opens ML Intern, Hugging Face's research agent, which now lives inside HuggingChat: https://huggingface.co/chat/?mode=ml-intern . The map does not run ML Intern and has no account or key of its own; you use ML Intern under your own Hugging Face account, on whatever terms it offers you. Nothing leaves the page until you press the button. When you do, the question is copied to your clipboard and also placed in the link that opens, so HuggingChat receives it.

What goes into the question:

| What you did on the map | What the question contains |
|---|---|
| Typed in the interests box | Your text, word for word (`My interests: …`). Left out if the box is empty |
| Ticked hide-switches | One line listing them (`I have ruled out papers where: …`) |
| Shortlisted papers with **+** | Exactly those papers, up to five |
| Shortlisted nothing | The five highest papers by your score that pass your hide-switches and filters. This is the score order, even if you re-sorted the table by another column |
| Moved weights, chose a preset, starred papers | Not written into the question, but they decide which five papers are highest |

For each paper the question gives: the title and arXiv link; the claim to verify and where it is in the paper (first 260 characters); the organisers' falsification criterion (first 200 characters); the repository link with its turnkey score, or "no repository found"; and up to four flagged issues. It ends by asking which paper to reproduce first and why, the first three steps, what is most likely to block you, and what reduced-scale version of the claim would still be a fair test.

Not included: the weight values, the map view, the compare table, and each paper's metric, comparison baseline, hardware sentences and reading time. Add those by hand if they matter to your question.

*When it is wrong:* the notes in the question are this map's scan results, with the same limits as the rest of the page. ML Intern's answer is its own; check it against the paper and the repository.

### Blockers, and ways round them

Each paper's panel starts with "What stands in your way, and a way round it". These are the things it can list, how each is detected, and how many papers it affects in this snapshot.

| What stands in the way | How it is detected | Papers | A way round it |
|---|---|---|---|
| No code repository found | No GitHub link in the PDF or on its `github.io` project page | 474 | Look for a supplementary file on the paper's OpenReview page; search the title on GitHub and Hugging Face; check the paper's other links; ask the authors |
| Code is in the supplementary material | The paper says so in a sentence (quoted on the page), and no repository was found | 13 | Download the supplementary file from OpenReview; a free OpenReview account may be needed |
| Code was promised for later | The paper says "will be released" or "upon acceptance", and no repository was found | 36 | It may be out now: search by title, check the project page, ask the authors |
| Something is available only on request | The paper says "upon request" about code, data or weights | 2 | Write to the authors early |
| Repository link is dead | The link returns 404 | 17 | It may have been renamed or moved; search for it, or ask |
| Repository has no code, or the README says code is coming | No code files; or "code coming soon" in the main README | 29 + 8 | Check it again — it was read at one commit — or ask for a timeline |
| The method under test is a hosted closed model | Closed-API keys in code, no local model code, and the paper names a hosted model | 14 | Paid API access to the same model version |
| Gated data or weights | "request access", a form or an agreement in top-level docs | 13 | Sign in and accept the terms, or apply, on day one |
| Manual downloads | Google Drive or Baidu Pan links | 66 | Budget time; Baidu Pan needs an account |
| Native build | C++/CUDA extension in the repository | 26 | Match CUDA and compiler versions first, or use a container |
| A hosted model is named in the text | GPT, Claude, Gemini… in the claim, method or metric text | 204 | Work out whether it is the method, a judge or a baseline |
| Claim is in the appendix | Stated page is past the start of the appendix | 225 | Read the appendix section for the setup |

**About OpenReview.** OpenReview does not allow automated checks, so this tool cannot tell which papers have a supplementary file there, or whether downloading it needs a login. The only signal is what the paper itself says. For any paper with no repository, treat the OpenReview page as the first place to look, and expect that you may have to sign in.

### How the claim is kept faithful to the official list

The claim, its source ("Table 2, page 6"), the metric, the comparison and the falsification criterion are **copied, not rewritten**. The panel shows all of them, because together they say what you must reproduce, against what, and what result would count as overturning it. They come from the official claim list at one pinned revision (`8187c351…`), and a check confirms that all 1,046 records are identical, field by field, to that revision. The page shows the full text. If the organisers update the list, this snapshot will be behind until step 1 is re-run with the new revision.

Every paper carries two links:

- **official record** — that paper's row in the official claim list on the Hugging Face Hub, which includes the organisers' picture of the claimed table or figure. This is the authority; if anything here disagrees with it, it is right.
- **claim it** — the hackathon's claim page, where you register that you are taking a paper. That page has no per-paper address, so the link copies the arXiv id for you to paste into its search. It needs a hackathon login.

### Where the claimed table or figure is in the paper

This tool does **not** locate the table or figure itself. The official list states it ("Table 2, page 6") and the page number is read from that text. Two things are then derived:

- **In the appendix?** The stated page is compared with the page where the appendix starts in the arXiv PDF.
- **Location check.** The stated page of the arXiv PDF is searched for the stated label ("Table 2"). The result is in `claim_location_check`:

| Result | Papers | What it means for you |
|---|---|---|
| found on the stated page | 1,017 | The arXiv PDF agrees with the official list |
| found one page away | 18 | Usually a different arXiv version or a float that moved; look at the neighbouring page |
| found on a different page | 7 | The arXiv PDF differs more from the version the organisers used; go by the picture in the official record |
| label not found in the PDF text | 1 | The label may be drawn as an image, or numbered differently |
| not checked | 3 | The official source names no table or figure number |

The PDFs are the current arXiv versions, which may not be the version the organisers read. When the check does not say "found on the stated page", trust the picture in the official record over the page number.

### Where the topics and areas come from

The claim list has no subject categories, so the two kinds of category on the page are **made by this tool**. They are not NeurIPS subject areas, not author keywords, not Papers with Code categories and not Hugging Face tags.

| Category | Source | How a paper gets it | How many per paper |
|---|---|---|---|
| **Topic** (for example "quantization / decoding / accuracy") | Computed from the papers themselves. Nobody wrote the list | Title + claim of every paper is embedded with `BAAI/bge-small-en-v1.5`, and the embeddings are grouped into 14 clusters with k-means. Each cluster is named by its three most distinctive words (TF-IDF), which is why the names read like word lists | One main topic, plus a second or third when the paper sits almost as close to that cluster's centre (within 0.02 cosine similarity). 644 of 1046 papers have more than one |
| **Area** (for example "healthcare", "robotics", "bci") | A hand-written list of 76 field names in [`search_glossary.toml`](search_glossary.toml), each with a one-line plain-language description. The list was written for this project and can be edited | The paper's text (title + claim + method summary) is compared with each description; an area is attached when the paper is unusually close to it (2.5 standard deviations above the average paper for that area) | Zero to three. 571 papers have none, because nothing in the list was close enough; 208 have two or three |

Both are rough. These are all choices, and different choices give different categories: the number of topics (14) is set in `signals.toml`, the list of areas in `search_glossary.toml`, and the two closeness thresholds in `pipeline/3_build_evidence.py`. On the page they are drawn with a dashed outline and the prefix "Auto" so they cannot be mistaken for official labels.

### The map

Each paper's position comes from the same title + claim embedding, projected to 2D with UMAP. Position reflects text similarity only. Two papers being close does not mean they share code, data or authors.

## Known limits

- The scan sees the default branch at one commit, without files over 2 MB. A repository may have changed since.
- A README can promise more than the code delivers. A high turnkey score means the right files and calls are present, not that they work.
- Papers without a found repository have no turnkey value at all. On the page they count as 0 in "my score"; use the "Has repo" filter to keep the two groups apart.
- Hardware sentences are picked by keyword and can refer to a baseline rather than the paper's own method.
- The rules were tuned by reading matches in this corpus. They will misjudge repositories that are unusual in ways this corpus is not.

## Changing the heuristics

Edit `signals.toml` — every pattern and weight is there — or `search_glossary.toml` for search terms, then re-run from the `scan` stage (for a glossary change, only the last two commands):

```bash
uv run python pipeline/2_scan_pdfs_and_repos.py scan --force
uv run python pipeline/2_scan_pdfs_and_repos.py merge
uv run --group embed python pipeline/3_build_evidence.py --no-embed
uv run python pipeline/4_build_map.py
```

To change only how the page scores or displays, edit `pipeline/4_build_map.py` and run the last command. To change the weights for yourself, you do not need any of this: use the sliders.

## Running the pipeline yourself

**You do not need to do this to use the map** — the hosted page and `index.html` already contain everything. Run it if you want to change the heuristics, add search terms, or check the work.

### Set up

You need [uv](https://docs.astral.sh/uv/) and git. No GPU, no API keys, no accounts.

```bash
git clone https://github.com/gitgithan/neurips2026-repro-map
cd neurips2026-repro-map
uv sync --group embed      # creates .venv from uv.lock (Python 3.11+); the "embed" group adds the text-embedding libraries
```

`uv run …` uses that environment automatically; there is nothing to activate.

### The cheap path: change something and rebuild the page (about 20 seconds)

The repository ships everything the page builder needs, so this works straight after cloning:

```bash
uv run python pipeline/4_build_map.py        # rebuilds index.html from data/ (about 1 second)
```

Changing `search_glossary.toml`, the checkability weights or anything else in step 3 needs step 2's intermediate files (`data/audit/`), which are not shipped because they are large. For that, run the full path once.

### The full path: rebuild everything

> **Size and time warning.** A full rebuild downloads about **6.6 GB of PDFs** and **4.5 GB of repository clones**, and the Python environment with the embedding libraries is about **6 GB** — roughly **17 GB** in total — and takes about **70 minutes**, almost all of it the PDF download. Use `--only <arxiv_id>,<arxiv_id>` on the step-2 stages to try a handful of papers first. `data/pdfs/` and `data/repos/` can be deleted once `merge` has run.

```bash
uv run python pipeline/1_fetch_claims.py
for stage in download parse pages head clone scan merge; do
  uv run python pipeline/2_scan_pdfs_and_repos.py $stage
done
uv run --group embed python pipeline/3_build_evidence.py
uv run python pipeline/4_build_map.py        # writes index.html
```

Measured on a 6-core cloud machine with a fast connection, for all 1,046 papers:

| Step | Stage | What it does | Time |
|---|---|---|---|
| 1 | | Download the claim list | about 5 s |
| 2 | `download` | Fetch the PDFs from arXiv, one every ~3 s on purpose | **about 55 min** — the slowest part by far |
| 2 | `parse` | Read 1,046 PDFs | about 1 min |
| 2 | `pages` | Read project pages for repository links | about 2 min |
| 2 | `head` | Check repository links are alive | about 2 min |
| 2 | `clone` | Shallow-clone 555 repositories | about 2 min |
| 2 | `scan` | Match the patterns against the cloned files | about 2 min |
| 2 | `merge` | One row per paper | 1 s |
| 3 | | Embed papers and search terms on the CPU, lay out the map | about 2 min (15 s with `--no-embed` once embeddings exist) |
| 4 | | Write `index.html` | 1 s |

Every stage is resumable: re-running it skips what is already done, and `--force` redoes it. After editing `signals.toml` you only need `scan --force`, `merge`, step 3 with `--no-embed`, and step 4 — about 3 minutes. PDFs and cloned repositories are not redistributed here; the scripts fetch them.

## Contributing

Corrections and improvements are welcome, especially:

- a paper whose repository was missed or chosen wrongly (open an issue with the arXiv id and the right link);
- a rule in `signals.toml` that misjudges a repository, with the example that shows it;
- fixes to the page.

This is a spare-time project reviewed on a best-effort basis, so small, focused changes with a concrete example are the ones most likely to be merged. Large automated rewrites are unlikely to be reviewed. If you want to take it in a different direction, fork it: it is MIT-licensed so that you do not need permission, and `signals.toml` plus the sliders are designed for exactly that.

## Sources and licence

**What this project publishes**

| What | Where | Licence |
|---|---|---|
| The interactive map | https://huggingface.co/spaces/gitgithan/neurips2026-repro-map | MIT |
| The evidence data (CSV + JSON, one row per paper) | https://huggingface.co/datasets/gitgithan/neurips2026-repro-map-evidence | MIT for the columns this project adds; `official_` columns remain under the organisers' terms |
| Code, rules and this documentation | https://github.com/gitgithan/neurips2026-repro-map | MIT |

The three are published together from one build, so they always describe the same snapshot.

**What it is built from**

| Source | What is taken from it | How |
|---|---|---|
| The organisers' claim dataset, https://huggingface.co/datasets/Neurips-2026-MLIntern-Repro/papers (revision `8187c351d8201824c2e606d9fb8e299b4709a98c`) | The list of papers, the four official labels, authors, summaries, and the claim, metric, comparison and falsification criterion | Copied unchanged into the `official_` columns; every row links back to its record there |
| arXiv, https://arxiv.org/abs/<id> | Each paper's PDF | Read by script for links, page structure, hardware sentences and statements about code. PDFs are not redistributed |
| The authors' public GitHub repositories | File lists and text files of each repository at one commit (recorded per paper) | Read by script, never executed. Repositories are not redistributed; only short quoted lines are kept as evidence |
| `BAAI/bge-small-en-v1.5` (text-embedding model) | Nothing from the papers; it is the tool that turns text into vectors | Used for the map layout, topics, areas and search by meaning |

Nothing comes from Papers with Code or OpenReview beyond the links to each paper's page there.
