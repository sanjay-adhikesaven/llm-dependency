# Link, Type-Correct, and Describe Lattice Items

> **Goal: for every item in the lattice, attach the official
> URL(s) that resolve to that exact item, fix the `kind` if
> the URL reveals a cross-kind mistag, and write a neutral
> target-independent description.** Most specific link first,
> every link HEAD-verified, no padding.

Read `{{lattice_path}}` and write the linked artifact to
`{{artifact_path}}`.

## Filesystem scope

Read `{{lattice_path}}` and `{{input_path}}` (same file).
Write `{{artifact_path}}`. HEAD-check every link with
`curl -sL -o /dev/null -w '%{http_code}' <url>`. Web search
is permitted to find canonical paper / blog URLs.

## Specificity rule (load-bearing)

A link must resolve to *exactly* what the item represents.
Read each item's `identity` to decide which links apply:

- **Specific artifact** — `identity` carries `size`, `stage`,
  `date`, or `quantization` (something pinning it to a real
  checkpoint or dataset config). Link to the specific repo:
  - `Qwen/Qwen3-4B` → `https://huggingface.co/Qwen/Qwen3-4B`
  - `allenai/Olmo-3-1025-7B` → `https://huggingface.co/allenai/Olmo-3-1025-7B`
  - `HuggingFaceTB/finemath::finemath-3plus` → the dataset
    config viewer URL (or the parent dataset URL).

- **Family-level** — `identity` only carries `org` +
  `collection` (or similarly broad keys). The item represents
  "the family", not any one checkpoint. Link to family-level
  resources only:
  - HF collection page (`https://huggingface.co/collections/<org>/...`)
  - tech report / paper (arXiv)
  - official creator GitHub repo
  - official release blog

  Do **NOT** point a family-level item at a specific
  checkpoint. `Qwen/Qwen3` is not `Qwen/Qwen3-4B`.

API-only artifacts (no HF/GitHub repo, e.g. OpenAI /
Anthropic models) link to their `vendor_docs` page.

## Link kinds (closed vocabulary)

| kind | what it points at |
|---|---|
| `hf_model` | HF model repo page |
| `hf_dataset` | HF dataset repo page |
| `hf_dataset_config` | HF dataset viewer for one config |
| `hf_collection` | HF collection page (family-level grouping) |
| `github` | official GitHub repo |
| `paper` | arXiv abstract page or other paper landing |
| `blog` | official release blog post |
| `vendor_docs` | API model docs (OpenAI, Anthropic, …) |

Use these strings verbatim. Don't invent new kinds.

## Priority for the FIRST link

Order the `links` array by descending canonicity. The first
entry is the most-specific official URL the item has:

1. The matching HF kind (`hf_model` / `hf_dataset` /
   `hf_dataset_config` / `hf_collection`).
2. `github` (when no HF page exists).
3. `paper` (when no HF or GitHub).
4. `vendor_docs` (API-only).
5. `blog` (when nothing more canonical exists).

After the primary, append every *additional* official link the
item has. A HF model that also has a paper AND a GitHub repo
gets all three — primary HF first, then paper / GitHub /
release blog as available. Find them all; don't truncate.

## Output schema

Same shape as the input lattice plus three new optional fields
on each item: `links` (always — possibly empty), `description`
(when at least one link is verified), and the existing `kind`
which you may have re-typed.

Items with re-typed kind get a top-level `notes` mention at
the end. Other fields (`formal_name`, `identity`, `aliases`)
pass through unchanged. The top-level `groups` list and
per-group `family` / `identity_keys` are unchanged.

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
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B"},
          "aliases": ["Qwen3-4B"],
          "links": [
            {"kind": "hf_model",
             "url": "https://huggingface.co/Qwen/Qwen3-4B"},
            {"kind": "github",
             "url": "https://github.com/QwenLM/Qwen3"},
            {"kind": "paper",
             "url": "https://arxiv.org/abs/2509.18888"}
          ],
          "description": "A 4-billion-parameter open language model from Alibaba's Qwen team, released September 2025 as part of the Qwen3 family. Trained on a multi-stage pipeline with extended context support."
        },
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3",
          "identity": {"org": "Qwen", "collection": "Qwen3"},
          "aliases": ["Qwen3", "Qwen 3"],
          "links": [
            {"kind": "hf_collection",
             "url": "https://huggingface.co/collections/Qwen/qwen3-..."},
            {"kind": "paper",
             "url": "https://arxiv.org/abs/2509.18888"},
            {"kind": "github",
             "url": "https://github.com/QwenLM/Qwen3"}
          ],
          "description": "Open-weight language-model collection from Alibaba's Qwen team, spanning sizes from 0.5B to 235B parameters with thinking and non-thinking variants."
        },
        {
          "kind": "model",
          "formal_name": "allenai/dolma3-fasttext-quality-classifier",
          "identity": {"org": "allenai", "collection": "dolma3"},
          "aliases": ["dolma3-fasttext-quality-classifier"],
          "links": [
            {"kind": "hf_model",
             "url": "https://huggingface.co/allenai/dolma3-fasttext-quality-classifier"}
          ],
          "description": "FastText-based quality classifier released by AI2 alongside Dolma 3, scoring web documents for inclusion in the pretraining mixture."
        }
      ]
    }
  ],
  "notes": "Re-typed allenai/dolma3-fasttext-quality-classifier from dataset to model (HF URL is a model repo)."
}
```

If an item has no resolvable official URL (rare; mostly
obscure one-off datasets), emit `links: []`. Don't guess.

## Kind correction (cross-kind mistags)

The lattice item carries a `kind` field (`"model"` or
`"dataset"`) inherited from extract. Sometimes extract or
organize gets the kind wrong — typically when the same surface
name was emitted as both kinds and the wrong one survived
clustering. The HF URL is the source of truth: a URL under
`https://huggingface.co/<owner>/<repo>` is a **model**, while
a URL under `https://huggingface.co/datasets/<owner>/<repo>`
is a **dataset**.

When the canonical link you found resolves to a different
kind than the lattice item's `kind` field, **fix the field**.
Example: `allenai/dolma3-fasttext-quality-classifier` may
arrive as `kind: "dataset"`, but
`https://huggingface.co/allenai/dolma3-fasttext-quality-classifier`
is a model repo — flip `kind` to `"model"`. Note in the
`notes` field which items you re-typed.

Don't re-type based on aliases or names alone — only when an
HF URL you HEAD-verified disagrees with the field.

## Neutral, target-independent description

For every item that has at least one official link, write a
short `description` field — 1 to 3 sentences, ≤500 chars —
**grounded in the card / repo / paper itself**, not in how
the target consumes it. The description should read the same
whether you ran the pipeline against the target or against
some other model that happens to use the same upstream.

Sources of description content (in priority order):

1. **HF README YAML frontmatter and the first prose
   paragraph** — fetch the card via
   `curl -sL https://huggingface.co/<repo>/raw/main/README.md`
   (or `.../datasets/<repo>/raw/main/README.md` for datasets)
   and use the first paragraph + the YAML's `pipeline_tag` /
   `task_categories` as the seed.
2. **GitHub README first paragraph** — if no HF card.
3. **arXiv abstract first sentence** — if the item is paper-
   only.

What the description must NOT say:

- "Used by Olmo-3 to ..." — target-dependent, banned. Frame
  the artifact as a standalone thing.
- "We use ..." / "We trained ..." — first-person framing
  comes from the target's authors writing about it; rewrite
  to third-person.
- "This dataset was used to train ..." — relationship to a
  consumer is captured by relate edges, not in the
  description.

Good descriptions:

- `"A 7-billion-parameter open language model from AI2,
  released October 2025 as part of the OLMo-3 family.
  Pretrained on Dolma 3 with extended context to 65k tokens."`
- `"A 102k-prompt RL dataset bundling math, code, and
  instruction-following sources, released by AI2 alongside
  the Olmo-3 RL recipe."`
- `"A neural OCR model fine-tuned for converting academic
  PDFs into clean text, released by AI2 in February 2025
  (Poznanski et al., 2025)."`

If you can't fetch a card / paper, leave `description: null`
rather than guessing.

## Subagent dispatch

The Task tool is available — subagents run as
`{{subagent_model}}`. Bucket the families and dispatch one
subagent per bucket (30-100 items per bucket is the sweet
spot). Each subagent finds and HEAD-verifies links for the
items in its bucket and returns its slice of the lattice.
Aggregate before writing.

Transcribe the specificity rule, link-kinds vocabulary, and
priority order verbatim into each subagent's brief — they have
none of your context.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
