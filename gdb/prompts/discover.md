# Discover

> **Goal: COVERAGE.** Fetch every source that documents
> `{{target}}`'s provenance. A miss creates a permanent lineage
> hole that nothing downstream can recover — extract reads only
> what discover wrote to disk, so a source you skip here is a
> source the lattice will never know about.

Fetch sources, organize them into batches, write a discovery
artifact to `{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `target` and `workspace_dir`.
- `{{workspace_dir}}`: writable directory for fetched files.

## Filesystem scope

Read `{{input_path}}`. Write fetched files into
`{{workspace_dir}}` (relative paths recorded in the discovery
artifact). Write the discovery artifact to `{{artifact_path}}`.
Network fetches land in `{{workspace_dir}}` — never to the
project tree, never to /tmp, never anywhere else.

## What is in scope

A source is in scope if and only if:

- It **is** the target: model card, dataset card, tech report,
  release blog, or target-specific code repo.
- It **is** the target's direct training upstream: the specific
  dataset consumed or the base model fine-tuned from.
- It **is** a target-specific recipe / config / training script
  living in the target creator's own repo.
- It **is** a filter classifier or synthetic-data generator
  used to build a direct upstream (e.g., `fineweb-edu-classifier`
  used to filter `fineweb-edu`, or `Mixtral-8x7B-Instruct` used
  to generate `cosmopedia`). Filter-classifier provenance is
  load-bearing — without it, the lineage truncates one hop
  short of the actual quality gate.

**In scope but light-touch (fetch the source page itself, not
its further dependencies):**

- Evaluation benchmark cards / papers (MMLU, GSM8K, IFEval,
  etc.) — they're datasets in the new model.
- Judge / teacher / reward models named explicitly as
  participants in the target's pipeline.
- Comparison / baseline models the source uses for context.

**Out of scope (do not fetch):**

- Distant sibling releases — earlier major versions of the
  same family that aren't a direct training input.
- Cited-but-not-used prior work (a one-line "tokenization
  follows X" reference where X didn't actually participate).
- Software / training frameworks / inference engines (these
  are not models or datasets; the storage layer rejects them).

**Parallel-pattern coverage.** If the target uses multiple
parallel artifacts of the same shape — sibling synthetic-data
variants, parallel filter classifiers (one per language, one
per cut), per-stage preference mixes — enumerate every
variant. Asymmetric coverage downstream (one variant linked,
the sibling unlinked) is harder to fix than over-fetching
here.

## Where to look

First-party sources (HF cards, paper PDFs, official creator
blogs, the creator's own GitHub repo, peer-reviewed venue
PDFs) before wrappers and summaries. Discover siblings via HF
and GitHub org APIs. Do not fetch tutorials, third-party
summaries, community wrappers, deprecated repos, or
tokenizer-only repos.

## Fetching conventions

> **Save raw response bytes. Do NOT summarize, paraphrase, or
> distill what you fetch.** Downstream extract reads the raw
> file; a summary IS a permanent lineage hole. The model card
> for `Foo` filtered into `Foo-Edu` is the ONLY place that
> names `Foo-Edu`'s base classifier — if you summarize "the
> card describes Foo's filtering", that name is lost forever.

Concretely:

- **Prefer Bash with `curl` / `wget` / `git clone` over the
  WebFetch tool.** WebFetch returns a model-summarized digest,
  not the response body — it is the wrong tool for primary-
  source capture. WebFetch is acceptable ONLY for
  orientation (deciding whether a URL is in scope), not for
  the saved file itself.
- HF model card raw README:
  `curl -sL https://huggingface.co/<repo>/raw/main/README.md -o <out>.md`
- HF dataset card raw README:
  `curl -sL https://huggingface.co/datasets/<repo>/raw/main/README.md -o <out>.md`
- HF API for commit SHA, base_model, datasets list:
  `curl -sL https://huggingface.co/api/models/<repo>/revision/main`
- arXiv: fetch BOTH the abstract HTML (orientation) AND the
  full PDF (the bytes extract reads). The full HTML page
  (`/html/<id>v<n>`) is also raw text — fetch it too if
  available; it's much easier to grep than the PDF.
  `curl -sL https://arxiv.org/abs/<id> -o paper-abs.html`
  `curl -sL https://arxiv.org/pdf/<id> -o paper.pdf`
  `curl -sL https://arxiv.org/html/<id>v1 -o paper.html`
- GitHub README and target-specific files:
  `curl -sL https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md`
  Or `git clone --depth 1` if you need many files from one
  repo.
- Pin commit_sha for HF and GitHub sources where possible
  (use HF API for the SHA; `git rev-parse HEAD` for cloned
  repos). arXiv PDFs need none.

### File-naming hygiene

Use lowercase, hyphenated filenames matching the artifact id
(`smollm2-1.7b-instruct.md`, not `SmolLM2-1.7B-Instruct.md`)
to keep the workspace listing scannable for downstream agents.

## Post-fetch verification

Before writing the manifest, read the first ~50 lines of every
captured file from disk and verify:

1. The file contains real source content (not a JS-shell
   wrapper, a 401/404 page, or a "Sign in to continue" stub).
2. For paper / blog / card sources: the target's name appears
   multiple times AND the source is **about** the target, not
   merely mentioning it in passing.
3. Files under 1 KB for paper/blog/card URLs are almost
   certainly stripped — re-fetch with `curl -L` and explicit
   `-A` user-agent if needed.

## Coverage self-check (mandatory)

After the initial fan-out, before writing the manifest:

1. Enumerate ground truth via HF and GitHub org APIs for the
   target's namespace (`huggingface.co/api/models?author=<org>`
   and similar for datasets / GitHub).
2. Filter to in-scope repos by family-prefix match (e.g.,
   `^smollm2-` for the SmolLM2 target).
3. **Cross-reference harvest**: grep every captured file for
   `huggingface.co/...`, `github.com/...`, `arxiv.org/...`
   patterns, plus bare names matching common HF repo shapes.
   Collect the URL set.
4. Diff (1)+(3) against what you've already fetched. Dispatch
   catch-up fetches for the missing ones (they're often
   filter classifiers, sibling stage-chain checkpoints, and
   synthetic-data generator models named only inside other
   cards — the highest-value lineage holes).
5. Iterate until coverage stops improving, with a small bound
   (≤3 rounds) to avoid runaway loops. If the bound trips,
   add a `coverage_warnings` field listing what remained
   unfetched.

## Output

```json
{
  "batches": [
    {
      "label": "target cards",
      "summary": "why these sources belong together",
      "sources": [
        {"path": "relative/to/workspace.md", "url": "https://...", "commit_sha": null}
      ]
    }
  ]
}
```

Each `sources[i].path` must be a path inside `{{workspace_dir}}`
that you actually wrote. The pipeline derives source `title`
from filename and content; you do not need to set it.

## Batching rules

A batch is the unit of downstream extract work. The principle:
one batch should be a topically coherent set that a careful
reader treats as a single pass — neither so narrow that you
fragment cross-references across batches, nor so wide that
the batch covers unrelated families.

1. **Stage chains stay together.** Base + SFT/DPO/Instruct for
   one size go in one batch.
2. **Dataset families stay together; different families
   split.** SFT mixes, pretraining corpora, math mixes, code
   mixes, DPO data are distinct families and split.
3. **Tech report is its own batch** (PDF + HTML + optional
   blog).
4. **Big sources go in exactly one batch. Small sources may
   be reused** if they belong with two families.
5. **Filter classifiers** group with the dataset family they
   filter (`fineweb-edu-classifier` belongs with the
   FineWeb-Edu / SmolLM-Corpus pretraining batch, not in its
   own batch).

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An
empty `batches[]` list usually signals a misread or a fetch
failure — surface the failure mode rather than emitting
nothing.

You are running as `{{planner_model}}`. Use subagents for
independent fetch packets. Subagents run as `{{subagent_model}}`.

{{subagent_prompt}}
