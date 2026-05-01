# graph/

`graph/` is a standalone prototype for the redesigned first phase:
discover sources, extract model/dataset mentions with structured
identity metadata, check and repair consistency, verify obvious links,
and emit a forest of lattice nodes.

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
gdb run verify-links
gdb run link-unresolved
gdb run build-lattice
```

The deterministic stages can also ingest artifacts directly:

```bash
gdb run extract-mentions --batch-id <batch> --artifact mentions.json
gdb run repair-mentions --artifact repair.json
```

Runtime paths are controlled by `GDB_STORAGE` and `GDB_PATH`.
