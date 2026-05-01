# graph/

`graph/` is a standalone prototype that walks from a target model or
dataset to a forest of lattice nodes: discover sources, extract
mentions, dedup-cluster them, audit each cluster's identity and
exact public link, verify the links, build the lattice, and describe
each entity leaf.

It does not migrate from or write to `trace/` or `prov/`.

## Quick start

```bash
cd graph
python -m pip install -e .
gdb init
gdb run discover --target HuggingFaceTB/SmolLM2-1.7B
gdb run extract
gdb run check
gdb run audit
gdb run verify-links
gdb run build-lattice
gdb run describe
```

The deterministic stages can ingest existing JSON artifacts directly:

```bash
gdb run extract --batch-id <batch> --artifact mentions.json
gdb run check  --artifact mentions.json
gdb run audit  --artifact audit.json
gdb run describe --artifact describe.json
```

Runtime paths are controlled by `GDB_STORAGE` and `GDB_PATH`.

## Pipeline

| Stage | Runtime | Job |
|---|---|---|
| `discover` | one Claude planner | fetch sources for the target into batches |
| `extract` | one Claude planner per batch (Python parallel) | per source: surface, kind, atoms, inline links, source-side anchors |
| `check` | Python | dedup mentions into clusters, detect 7 conflict codes |
| `audit` | one Claude planner → fans out subagents | per cluster: identity, aux, aliases, concept_path, confirmed link, conflict resolution |
| `verify-links` | Python | HEAD-check every typed link |
| `build-lattice` | Python | concept + entity nodes, edges, forest manifest, lattice audit |
| `describe` | one Claude planner → fans out subagents | per entity-leaf description, with HF README / API fetch inline |

Two parallelism shapes:

- **Python-level** (`extract` only) — batches are independent;
  `ThreadPoolExecutor` spawns one Claude planner per batch.
- **Planner-level** (`audit`, `describe`) — the stage needs a global
  view, so Python spawns ONE Claude planner. The planner buckets the
  work and dispatches subagents (e.g., Codex / Sonnet / Haiku for
  cost) via the Task tool, then aggregates and writes one artifact.

## Terminology

A mention has three layers:

- **identity** — lattice-axis facets (`family`, optional `size` and
  `stage`, plus `extra` for date snapshots and other axis-relevant
  bits).
- **aux** — lossless facets that should match across mentions of the
  same concept but don't add a lattice axis (release `date`,
  `mix_size`, `context_length`, `version`, source-local labels).
- **aliases** — surface variants of the same referent. Each alias
  carries `descriptors` (per-variant facets like `quantization`,
  `precision`, `format`, `namespace`) and may carry its own typed
  link list when the variant has its own public release.

Identifiers and citations are split:

- **link** — a URL-resolvable typed identifier. One of `hf_model`,
  `hf_dataset`, `hf_dataset_config`, `github_repo`, `github_ref`,
  `api_model_id`, `official_release_url`, `paper_release`.
- **anchor** — a source-side citation: the file path, location, and
  verbatim excerpt that grounds the mention to a real spot in the
  source corpus.

Concrete entity leaves are keyed by their exact link. Concept nodes
are reviewed concept-path prefixes and can share a display name with
an entity leaf — for example, an abstract `Qwen3-4B` concept and the
concrete `Qwen/Qwen3-4B` HF model. The lattice distinguishes them by
`node_type`.

## Tests

```bash
python -m pytest tests/ -q
```
