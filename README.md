# graph/

`graph/` is a standalone prototype for the redesigned first phase:
discover sources, extract model/dataset names, review them into
concept paths and exact entity anchors, check and repair consistency,
verify obvious links, describe concrete entities, and emit a forest of
lattice nodes.

It intentionally does not migrate from or write to `trace/` or `prov/`.

## Quick Start

```bash
cd graph
python -m pip install -e .
gdb init
gdb run discover --target HuggingFaceTB/SmolLM2-1.7B
gdb run extract-mentions
gdb run check-mentions
gdb run repair-mentions
gdb run investigate-hf
gdb run review-entities
gdb run verify-links
gdb run link-unresolved
gdb run build-lattice
gdb run build-relationships
gdb run describe-entities
```

The deterministic stages can also ingest artifacts directly:

```bash
gdb run extract-mentions --batch-id <batch> --artifact mentions.json
gdb run repair-mentions --artifact repair.json
gdb run review-entities --artifact review.json
gdb run link-unresolved --artifact links.json
```

Runtime paths are controlled by `GDB_STORAGE` and `GDB_PATH`.

Concrete entity leaves are keyed by exact anchors such as HF repos,
HF dataset configs, GitHub refs, API model ids, official release URLs,
or exact paper-only release records. Concept nodes are reviewed path
prefixes and may share display names with entity leaves, e.g. an
abstract `Qwen3-4B` concept and the concrete `Qwen/Qwen3-4B` HF model.

`investigate-hf` fetches HF README front matter and Hub API metadata,
validates dataset configs when visible, records collection candidates,
and materializes relationship hints such as `base_model`.
