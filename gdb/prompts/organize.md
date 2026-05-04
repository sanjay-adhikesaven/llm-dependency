# Organize, Resolve, and Describe Names

> **Goal: read every name, group surface variants into family-
> structured items, find the canonical URL for each item, write a
> target-independent description, and drop items that can't be
> unambiguously pointed at.** Every emitted item must resolve to
> a real artifact you can put a clickable URL on; no invented
> nodes, no orphan placeholders.

Read `{{names_path}}` and write the artifact to
`{{artifact_path}}`.

## Inputs

- `{{names_path}}`: JSON `{"names": [{"type": "model"|"dataset", "name": "..."}, ...]}`.
  Already deduped on `(type, name)`. Surface variants of the
  same artifact (case / separator / accent / HF-org-prefix /
  parenthetical differences) are NOT deduped — that's your job.

## Filesystem and tool scope

Read `{{names_path}}` and `{{input_path}}` (same file). Write
`{{artifact_path}}`. WebSearch, Bash (curl / wget), and WebFetch
are all available — use them as needed to find canonical URLs,
HEAD-check candidates, and read the artifact's own card / paper
when writing descriptions.

## Valid node forms (load-bearing — drop if unresolvable)

Every item in the output lattice MUST resolve to ONE of three
forms:

1. **Open-source model or dataset** — exact HuggingFace
   identifier `<org>/<repo>` whose page returns 200, OR an
   official GitHub repo whose page returns 200. The
   `formal_name` is `<org>/<repo>` (lowercase HF form preferred);
   primary link is the `hf_model` / `hf_dataset` / `github` URL.
2. **Closed-source model** — clickable official URL (vendor docs
   page, API model page, or release-blog landing). The
   `formal_name` is `<vendor>/<identifier>`; primary link is
   `vendor_docs`.
3. **Paper-anchored release** — exact paper URL (arXiv abstract,
   ACL anthology, venue page) when no HF or GitHub release
   exists but the artifact is described in a published paper.
   The `formal_name` is a stable paper-derived label; primary
   link is `paper`.

If web search cannot resolve a name to one of these forms,
**DROP the item** to `dropped[]` with a one-line reason. The
lattice's value comes from each node being unambiguously
pointable; an item with no canonical anchor is noise.

### Drop subset / config names — they're added later

If a name looks like a subset or config of a parent dataset
(e.g., a sub-corpus of a training mix, an HF dataset config
slug, an eval-harness reformulation, a quality-tier filter)
rather than a standalone artifact, **drop it**. A separate
Python pass (`gdb.subsets`) reads each kept dataset's HF
README and populates a `subsets[]` field with its
configs / components / mix-table entries. After organize
completes, audit's pre-pass cross-checks every dropped name
against every kept dataset's `subsets[]` and restores hits
as child items with `formal_name = <parent>/<subset_slug>`.

You don't need to recover subsets yourself. Drop them with
reason like "subset of <parent>" or "config of HF dataset"
and let the Python pass handle restoration. The cost of
emitting a subset as a top-level item is wrong identity
keys, wrong link target, and a node that doesn't fold under
its parent.

Cues that a name is a subset / config:
- It uses an HF config-suffix syntax (`finemath-3plus`,
  `bbh:cot`, `mmlu:mc`).
- It uses an eval-harness reformulation suffix
  (`::cot::xxx`, `_rc_5shot`, `_Gen2MC`).
- It's named in the input pile alongside its parent (both
  `HuggingFaceTB/finemath` and `finemath-3plus` appear).
- It's a known component of a larger mix referenced in the
  input pile.

### Family-concept exception

A family-concept item (`Qwen3` the family vs. specific
`Qwen/Qwen3-4B`) is a valid SEPARATE node ONLY when it has its
own HF collection URL (`huggingface.co/collections/<org>/...`)
or its own dedicated paper. Otherwise, family-concept mentions
collapse as `aliases` of the most-likely specific item — usually
the chat / instruct variant when the source uses a bare size
form.

## What you decide

For each input name:

1. **Family membership** — which other names refer to the same
   family of artifacts.
2. **Family name** — a short, recognizable label.
3. **Identity keys for the family** — the dimensions that vary
   inside it. Open vocabulary.
4. **Surface collapse** — names that differ only in case /
   separator / accent / HF-org prefix / trailing parenthetical
   merge into ONE item with multiple `aliases`. Names that
   differ in any identity dimension (size, stage, date,
   quantization) stay separate.
5. **Per item: `formal_name`, `identity` dict, `aliases` list,
   `kind`, `links`, `description`** — see schema below.
6. **Resolution** — web-search every clustered item to find its
   canonical URL. HEAD-check before adding. Drop items that
   can't be resolved.

Every kept item must trace back to at least one real input name.
Do not invent items.

### Hard rule: every item MUST have ≥1 alias from the input pile

If after clustering an item would have `aliases: []` (no input
surface form resolved to it), the item is INVENTED — drop it.
HF org enumeration is for finding canonical URLs of names the
input mentioned, NOT for adding releases the input never named.
Phantom items the input pile didn't name are noise; the lattice
models what the sources SAID, not the entire HF catalog.

**Exception (narrow):** a family-concept root whose `identity`
carries only broad keys (e.g., `org` + `collection` only; no
`size` / `stage` / `date`) MAY have empty `aliases` when it
serves as the partial-order anchor for items below it AND has
its own HF collection URL or paper.

## Bucketing for parallelism

This is a hint for splitting the input across subagents — NOT a
definition of family membership.

For each name, take the substring before the first `/` or `-`
(whichever appears first):
- `Qwen/Qwen3-4B` → `Qwen`
- `Qwen3-7B-Instruct` → `Qwen3`
- `OLMo-3-1025-7B` → `OLMo`
- `MMLU-Pro` → `MMLU`

Group names whose prefix-tokens share ≥3 consecutive identical
characters into the same bucket. Each bucket goes to one
subagent. Right-size buckets to **30-100 names** (smaller than
naive parallelism because each item now requires a web call).

The 3-char rule is approximate. Two names in different buckets
may turn out to belong to the same family; the planner reviews
subagent outputs and merges where needed before writing the
final artifact.

## Disambiguating-facets rule (load-bearing)

Within a family, every item's `identity` dict MUST distinguish
it from every other item in the same family. If two siblings
would share the same `identity` dict, you MUST add a facet that
separates them.

Bad — siblings collapse to the same identity:

```json
{"formal_name": "Qwen/Qwen3-4B",
 "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B"}}
{"formal_name": "Qwen/Qwen3-4B-Base",
 "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B"}}
```

Good — add `stage`:

```json
{"formal_name": "Qwen/Qwen3-4B",
 "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "chat"}}
{"formal_name": "Qwen/Qwen3-4B-Base",
 "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "Base"}}
```

Choose facet keys from the family's `identity_keys` list; if no
existing key disambiguates, extend `identity_keys` with a new
one. Common disambiguating facets:

- `stage` — Base / chat / Instruct / Think / SFT / DPO
- `variant` — thinking / no-thinking
- `quantization` — FP8 / AWQ / GPTQ
- `date` — API snapshot date

Don't force one schema across unrelated families. A family of
benchmark variants will look nothing like a family of model
checkpoints.

## Resolution: find the canonical anchor

For each clustered item, find a URL that resolves AND describes
the named artifact. You have these tools available — use
whichever fits each case:

- **WebSearch** — free-form queries (e.g., `"<name>" hugging face`,
  `"<name>" github`) to find candidates
- **Bash** — `curl` for HEAD-checks and HF/GitHub API calls
- **WebFetch** — read a candidate page's content to verify it
  matches context

The work has three checks; the order and method are up to you.

### Check 1 — does the URL resolve?

HEAD-check before adding any URL to `links`. The HTTP code
maps to an outcome:

| code | meaning | action |
|---|---|---|
| **200** | exists, accessible | adopt as the primary `links[0]` |
| **401 / 403** | exists, gated | **promote to top-level `gated[]`**, NOT `dropped[]` (the artifact IS real, just inaccessible) |
| **404** | not at this URL | try a different strategy below — don't drop yet |
| **5xx / timeout** | transient | retry once before giving up |

### Check 2 — does the page describe the right artifact?

A HEAD-200 URL is necessary but NOT sufficient. Before adopting
any URL as `links[0]`, fetch the card / abstract / page and
confirm the content matches the input context — at least one of:

- The page's first prose paragraph mentions the input pile's
  organization, the family the input name belongs to, or the
  broader topic (training data, pretraining mix, eval benchmark,
  language model, classifier, etc.)
- The page's `pipeline_tag` / `task_categories` / paper abstract
  is consistent with the input name's role

If neither holds, the candidate is a **name-collision** — drop
the item with `signal: "misidentified"` and the wrong URL listed
in `attempted_canonicals[]` so audit doesn't retry it.

### Check 3 — exhaust strategies before dropping

A drop is the last resort, not the first. If your initial URL
guess returns 404, try:

- HF org enumeration (list all artifacts under the named org and
  fuzzy-match)
- HF name search (search by name fragment)
- **The other kind** — if you tried the model URL and got 404,
  try the dataset URL (and vice versa). Extract may have guessed
  the kind wrong; the right move is to fix the kind, not drop.
- GitHub (if HF has nothing)
- Vendor docs (closed-source models)
- Paper search (arXiv, ACL anthology)

Drop only after multiple strategies fail.

### Picking and writing the canonical form

- HF identifier wins when it exists. Lowercase the HF repo path
  (`Qwen/Qwen3-32B` → `Qwen/Qwen3-32B` is fine; the case-
  sensitive HF form is canonical). Surface variants the source
  used go in `aliases`.
- API-only artifacts (OpenAI, Anthropic, Google) use
  `<vendor>/<identifier>` with the dated snapshot dropped from
  the canonical (`OpenAI/gpt-4.1`, not
  `OpenAI/gpt-4.1-2025-04-14`); dated snapshots go in `aliases`.
- For HF artifacts whose canonical repo has multiple revisions
  via git branches (`<repo>@<branch>`), the canonical formal_name
  is the repo path WITHOUT the branch suffix; branches go in
  aliases. Branches are revisions of one artifact, not separate
  artifacts.

## Link kinds (closed vocabulary)

| kind | what it points at |
|---|---|
| `hf_model` | HF model repo page |
| `hf_dataset` | HF dataset repo page |
| `hf_collection` | HF collection page (family-level grouping) |
| `github` | official GitHub repo |
| `paper` | arXiv abstract page or other paper landing |
| `blog` | official release blog post |
| `vendor_docs` | API model docs |

Use these strings verbatim. Don't invent new kinds.

### Priority for the FIRST link

Order the `links` array by descending canonicity. The first
entry is the most-specific official URL the item has:

1. The matching HF kind (`hf_model` / `hf_dataset` /
   `hf_collection`).
2. `github` (when no HF page exists).
3. `paper` (when no HF or GitHub).
4. `vendor_docs` (API-only).
5. `blog` (when nothing more canonical exists).

After the primary, append every additional official link the
item has. A HF model that also has a paper AND a GitHub repo
gets all three — find them all; don't truncate.

### Specificity for family vs. leaf

A link must resolve to *exactly* what the item represents:

- **Specific artifact** — `identity` carries `size`, `stage`,
  `date`, or `quantization`. Link to the specific repo.
- **Family-level** — `identity` only carries broad keys (`org`
  + `collection`). The item represents "the family", not any
  one checkpoint. Link to family-level resources only: HF
  collection page, paper, official creator GitHub repo,
  release blog. Do NOT point a family-level item at a specific
  checkpoint.

## Description writing

For every item with at least one verified link, write a
`description` field — 1 to 3 sentences, comprehensive —
**grounded in the card / repo / paper itself**, not in how the
target consumes it. The description should read the same
whether you ran the pipeline against the target or against
some other model that happens to use the same upstream.

Sources of description content (in priority order):

1. **The HF model / dataset card** — first prose paragraph plus
   `pipeline_tag` / `task_categories` from the YAML frontmatter.
   Multiple ways to get it; use whichever works:
   - `curl` the raw README at `<page-url>/raw/main/README.md`
     (fastest when it works);
   - **WebFetch the page URL itself** when the raw README returns
     401 / 403. Many real model cards (notably the Meta Llama
     family, some Mistral / Google releases) gate the raw file
     behind a license click but the rendered page still exposes
     the card text to anonymous viewers — WebFetch sees what a
     browser sees and gets the description body even though
     `curl /raw/main/README.md` returned 401;
   - HF API (`huggingface.co/api/models/<owner>/<repo>`) for
     metadata fields (`pipeline_tag`, `library_name`, `tags`,
     `cardData`) when neither raw nor page work.
2. **GitHub README first paragraph** — if no HF card or the HF
   card returned no usable prose. WebFetch on the GitHub repo
   page renders the README the same way browsers do.
3. **arXiv abstract first sentence** — if the item is
   paper-anchored, fetch the abstract page (or read the PDF if
   already in the workspace).
4. **WebSearch** as last resort — when none of the above
   produces a usable seed, search for `"<formal_name>"` and
   pull a description from a release blog, vendor docs, or
   reputable third-party (HF blog post, paper that introduces
   the artifact). Keep the description grounded in something
   *citable*; don't invent.

What the description must NOT say:

- "Used by <target> to ..." — target-dependent, banned. Frame
  the artifact as a standalone thing.
- "We use ..." / "We trained ..." — first-person framing comes
  from the target's authors writing about the target itself;
  rewrite to third-person.
- "This dataset was used to train ..." — relationship to a
  consumer is captured by relate edges, not in the description.

If after exhausting the above you still have nothing citable,
leave `description: null` rather than guessing. A null
description is a known gap; a fabricated one is misinformation.

## Kind correction (cross-kind mistags)

The lattice item carries a `kind` field (`"model"` or
`"dataset"`) inherited from extract. Sometimes extract gets the
kind wrong — typically when the same surface name was emitted as
both kinds. The HF URL is the source of truth: a URL under
`https://huggingface.co/<owner>/<repo>` is a **model**, while a
URL under `https://huggingface.co/datasets/<owner>/<repo>` is a
**dataset**.

When the canonical link you found resolves to a different kind
than the lattice item's `kind` field, **fix the field**. Note
in the top-level `notes` which items you re-typed.

Don't re-type based on aliases or names alone — only when an HF
URL you HEAD-verified disagrees with the field.

## Per-item schema

```json
{
  "kind": "model",
  "formal_name": "Qwen/Qwen3-4B",
  "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "chat"},
  "aliases": ["Qwen3-4B", "qwen3-4b", "Qwen 3 4B"],
  "links": [
    {"kind": "hf_model", "url": "https://huggingface.co/Qwen/Qwen3-4B"},
    {"kind": "paper",    "url": "https://arxiv.org/abs/2509.18888"},
    {"kind": "github",   "url": "https://github.com/QwenLM/Qwen3"}
  ],
  "description": "A 4-billion-parameter open language model from Alibaba's Qwen team..."
}
```

For dataset items, also include a `subsets` field (emit `[]`):

```json
{
  "kind": "dataset",
  "formal_name": "allenai/dolma3_dolmino_mix-100B-1025",
  "identity": {...},
  "aliases": [...],
  "links": [...],
  "description": "...",
  "subsets": []
}
```

- `kind`: `"model"` or `"dataset"`. May be re-typed during URL
  resolution per "Kind correction".
- `formal_name`: the HEAD-verified canonical identifier.
- `identity`: dict whose keys are the family's `identity_keys`.
  Must distinguish this item from every family sibling.
- `aliases`: deduped list of every original input surface form
  that collapsed to this item. The `formal_name` itself goes
  in `aliases` only if a source emitted it verbatim.
- `links`: ordered list with primary canonical URL first.
  Empty list means audit will revisit.
- `description`: comprehensive, neutral, target-independent. May
  be `null` if no card / paper could be fetched.
- `subsets` (datasets only): emit as `[]`. The Python pass
  (`gdb.subsets`) populates it post-organize from the HF
  README's `configs:` and components / mix tables.

## Per-family schema

```json
{
  "family": "Qwen3",
  "identity_keys": ["org", "collection", "size", "stage"],
  "items": [ ... ]
}
```

- `family`: a short label you choose. Pick the most recognizable
  short form (the collection name in most cases).
- `identity_keys`: the dimensions that vary across this family's
  items. Pick from open vocabulary; common keys include `org`,
  `collection`, `version`, `size`, `stage`, `date`,
  `quantization`, `vendor`, `family`, `variant`. Don't force one
  schema across unrelated families.

## Output

```json
{
  "groups": [
    {
      "family": "Qwen3",
      "identity_keys": ["org", "collection", "size", "stage"],
      "items": [
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3-4B",
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "chat"},
          "aliases": ["Qwen3-4B", "Qwen 3 4B"],
          "links": [
            {"kind": "hf_model", "url": "https://huggingface.co/Qwen/Qwen3-4B"}
          ],
          "description": "..."
        }
      ]
    }
  ],
  "gated": [
    {
      "name": "<name from input>",
      "kind": "model" | "dataset",
      "reason": "<one-line reason>",
      "gated_url": "<URL that returned 401 / 403>",
      "attempted_canonicals": ["..."]
    }
  ],
  "dropped": [
    {
      "name": "<name from input>",
      "kind": "model" | "dataset",
      "reason": "<one-line reason>",
      "attempted_canonicals": ["<URL or path tried>", "..."],
      "signal": "404" | "no-search-result" | "misidentified"
              | "subset-of-parent" | "ambiguous"
              | "rate-limit" | "timeout"
    }
  ],
  "notes": "<short summary of family merges, kind re-types, drops>"
}
```

### `dropped[]` schema (load-bearing — audit reads this)

`dropped[]` is for items where **no resolvable URL exists** for
any kind / variant tried. Each entry MUST include:

- `name` — the original surface form from extract (verbatim).
- `kind` — `"model"` or `"dataset"`.
- `reason` — one-line drop reason.

Optional but RECOMMENDED:

- `attempted_canonicals` — the candidate URLs you tried before
  giving up. Audit uses these to avoid re-trying paths you
  already proved don't exist.
- `signal` — what the failed lookups returned. Closed vocab:
  `"404"`, `"no-search-result"`, `"misidentified"` (you found a
  different artifact under the same name), `"subset-of-parent"`
  (the name is a subset / config of a parent dataset; the
  Python pass will restore it), `"ambiguous"`, `"rate-limit"`,
  `"timeout"`.

`401` and `403` are NOT in the `dropped[]` signal vocabulary.
Items returning 401/403 go in `gated[]` instead — see below.

### `gated[]` schema (load-bearing — for HEAD-401/403 items)

When a candidate URL returns 401 / 403, the artifact exists but
is gated (private / requires access). It is NOT a drop — the
lattice still benefits from knowing the artifact exists, and
relate may reference it via free-text `object` mentions.

Each `gated[]` entry has:

- `name` — original surface form from extract (verbatim).
- `kind` — `"model"` or `"dataset"`.
- `reason` — one-line note (e.g., "Gated HF dataset").
- `gated_url` — the verified-but-gated URL.
- `attempted_canonicals` (optional) — other URLs tried.

## Family-split sanity check (run after merging)

Walk every family with ≥ 2 items. For each, verify:

1. **Same provider / org.** If items in the family span two
   different `identity.org` (or `identity.vendor`) values that
   aren't aliases of each other, split into two families.
   Cross-org clustering is almost always a substring false
   positive.

2. **Same artifact kind.** If items in the family span both
   `model` and `dataset` kinds AND don't share a release event,
   split.

3. **Same product line.** If items name semantically distinct
   artifacts (a benchmark vs. a corpus, a pretraining mix vs.
   an instruction dataset), they're different families even if
   their names share substrings.

For each split / merge, document briefly in `notes`.

## Subagent dispatch

The Task tool is available — subagents run as
`{{subagent_model}}`. One subagent per bucket. Right-size to
30-100 names per bucket.

When dispatching, transcribe verbatim into each subagent's brief:
- The "Valid node forms" section (the 3-form criterion + drop-
  subsets rule).
- The "Disambiguating-facets rule" section.
- The "Web search and HEAD-verification" section (especially
  step 3, content verification).
- The "Link kinds" closed-vocabulary table.
- The "Description writing" section.

Subagents have none of your context — without these
transcriptions they will silently revert to old behavior
(cluster but not resolve / describe / drop / verify content).

After all subagents return, review for cross-bucket merges (two
seed buckets that turned out to hold one family) and final URL
spot-checks. Apply the family-split sanity check before writing
the final artifact.

### Rate-limit handling

If a Task dispatch fails with a rate-limit / 429 / overloaded
error, sleep and retry the dispatch — do NOT do the subagent's
work inline. Inline-fallback runs every web search and
HEAD-check at planner cost instead of subagent cost.

Recommended retry pattern:
- First retry: sleep 30 seconds, dispatch again.
- Second retry: sleep 90 seconds, dispatch again.
- Third retry: sleep 5 minutes, dispatch again.
- Still failing: only then consider merging two buckets into one
  larger Task before inline-fallback.

The pipeline also retries hard spawn failures at the dispatch
level with exponential backoff; you only need to handle the
Task-tool subagent dispatches that happen inside your turn.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
