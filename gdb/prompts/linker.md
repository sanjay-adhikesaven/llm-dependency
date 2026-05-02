# Link Lattice Items to Official URLs

> **Goal: for every item in the lattice, attach the official
> URL(s) that resolve to that exact item.** Most specific
> link first, every link HEAD-verified, no padding.

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

Same shape as the input lattice. Each item gets a new `links`
array. Other fields (`kind`, `formal_name`, `identity`,
`aliases`) pass through unchanged. The top-level `groups`
list and per-group `family` / `identity_keys` are unchanged.

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
          ]
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
          ]
        }
      ]
    }
  ]
}
```

If an item has no resolvable official URL (rare; mostly
obscure one-off datasets), emit `links: []`. Don't guess.

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
