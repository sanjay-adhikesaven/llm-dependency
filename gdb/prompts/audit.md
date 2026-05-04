# Audit and Revise

> **Goal: read the whole lattice, find what's weird, fix it.** You're
> a senior reviewer doing a second pass over organize's work. There
> is no fixed list of issues to look for — survey the structure,
> identify cases that don't make sense, and decide on appropriate
> edits. Same shape in, same shape out.

Read `{{organize_path}}` and write the revised artifact to
`{{artifact_path}}`.

## Filesystem scope

Read `{{organize_path}}` and `{{input_path}}` (same file). Write
`{{artifact_path}}`. Web search and HF / GitHub URL lookups are
permitted; you'll use them more than organize did because the
recheck pile demands fresh investigation.

## What runs before you (your input is pre-processed)

A pure-Python pass (`gdb.subsets.populate_then_flag`) runs BEFORE
you. It is **purely additive** — never moves items, never
restores drops, never renames anything. It does two things:

1. **Populates `subsets[]` on every dataset node** by fetching
   the HF README and parsing its YAML `configs:` field plus
   markdown composition / components / mix tables. Each subset
   slug is lowercase-kebab.

2. **Surfaces suspicious cases as a top-level `audit_hints[]`
   array.** Each hint flags a case that likely warrants your
   attention but doesn't prescribe an action. Hint kinds:

   - `item_matches_parent_subset` — an existing item's name slug
     appears in some other kept item's `subsets[]`. The
     `item_role` field labels what kind of item it is
     (`canonical` / `soft-anchored` / `concept` / `unanchored`).
     Typical action: reshape soft-anchored sub-components under
     the parent as `<parent>/<slug>` child items; keep
     foundational concepts (`item_role: concept`) standalone.
   - `dropped_matches_parent_subset` — a dropped name's slug
     appears in some kept item's `subsets[]`. Typical action:
     restore as `<parent>/<slug>` child item with identity
     inheriting parent + `subset: <slug>`, then remove from
     `dropped[]`.
   - `sibling_identity_collision` — two items in the same family
     carry identical identity dicts. You MUST resolve.
   - `cross_org_family` — a family spans multiple `identity.org`
     / `identity.vendor` values. Your judgment: substring false
     positive (split) or legitimate product-line grouping (keep,
     possibly add a discriminating facet).
   - `formal_name_vs_canonical_url_mismatch` — the formal_name
     doesn't match the canonical path inside its primary HF URL.
     Typical action: rename to canonical and carry the old form
     as alias (cosmetic, do in batch via Bash, not subagent).
   - `phantom_item` — empty aliases AND identity isn't a
     family-concept root. Drop or fix.

   The hints are **suggestions, not commands**. A hint exists
   because Python found a pattern; you decide what's right
   given the broader context. A hint doesn't apply when the
   item is foundational, when a duplicate name is coincidental,
   or when the cross-org grouping captures a real product-line
   relationship.

So when you read the input artifact: every dataset has populated
`subsets[]`, the lattice structure is exactly as organize left
it (no items moved or restored), and `audit_hints[]` lists what
Python found suspicious. **Your edits are the only changes that
happen to the lattice.**

## How to think about this pass

You are a careful reviewer. The lattice in front of you was
produced by a planner working under time pressure across many
buckets in parallel. It will have inconsistencies. Your job is
to spot them and fix them. There is no exhaustive checklist
because the failure modes are open-ended; instead, hold these
questions in mind as you walk the lattice.

### The one universal rule: verify before acting

Before you rename or drop any item — including items whose names
look "weird," items whose URLs look unfamiliar, items that look
"invented" — **HEAD-check the URL the item points at.** If it
returns 200 and the page describes the named artifact, the item
is real, no matter how unusual the name or URL looks.

```
curl -sL -o /dev/null -w '%{http_code}' <url>
```

A real artifact's name and anchor do not have to look like what
you'd expect. Suffixes the planner thinks are "synthetic" are
sometimes the actual canonical names the artifact's authors
chose. Anchors that aren't HF or GitHub are sometimes the only
canonical home an artifact has (a foundational data project on
its own .org domain; a creator's HF org page or HF buckets
page; a benchmark released as a paper with no separate repo).
Pattern-matching on name shape or URL kind without checking
ground truth is the most expensive mistake you can make in this
pass — it deletes real items that already cost organize money
to find.

### Questions to walk the lattice with

- **Does each item correspond to exactly one real artifact?**
  An item is fine if (a) its primary URL HEAD-checks 200 AND
  (b) the page at that URL describes the named artifact. An
  item is broken when either fails: rename to its canonical
  identifier (carry the old name as alias), or drop with a
  recorded reason, or promote to `gated[]` if the URL is
  401/403, or split if the item absorbs two distinct artifacts.

  Valid anchors include all of these — none are second-class:
  - HF model / dataset / collection / org / buckets pages
  - GitHub org / repo
  - Paper (arXiv, ACL, venue page)
  - Vendor docs (OpenAI / Anthropic / Google API model pages)
  - Official project homepage on its own `.org` / `.com`
    domain (foundational data resources like web crawls,
    archives, encyclopedias, and forums often use this)
  - Release blog post

- **Are family groupings cohesive?** A family should hold items
  that share a real product line. Lineage cohesion matters
  more than substring matching — items related by release
  history (an extension / variant / converted form of a base
  artifact) belong together; items that only share a name
  fragment do not. The auditor's judgment here is what
  matters; document each split / merge in `notes`.

- **Within each family, do siblings actually differ?** If two
  items in the same family carry identical `identity` dicts,
  the lattice can't tell them apart. Either they're the same
  artifact (merge) or they need a discriminating facet
  (extend `identity_keys`).

- **Do `formal_name`s match the artifact's canonical anchor?**
  After verifying the URL is real, check that the formal_name
  matches the canonical identifier of the resolved page: HF
  artifacts use the lowercase HF `<owner>/<repo>` path that
  appears in the URL; closed-source models use
  `<vendor>/<slug>` from the vendor docs URL; family-concept
  roots use a clean family name (no `(collection)` /
  `(legacy family)` parentheticals — the cleaner form is just
  `Qwen3` / `Qwen3-Coder`). Rename when there's a clear
  mismatch (e.g., a fabricated slug whose URL doesn't 200).
  **Never rename based on suffix shape alone — only after
  proving the original 404s.**

- **Don't synthesize items to absorb scaling-config or
  factory-function aliases.** When you see dropped[] entries
  like `olmo2_14M`, `llama_like`, `gemma3_27B` — Python factory
  functions or scaling-experiment configs from a primary code
  repo — they are CODE, not artifacts. Don't create a synthetic
  family-concept item like `meta-llama/Llama-2-architecture` or
  `allenai/OLMo-2-architecture` to absorb them. The
  architecture relationship those configs reference (e.g.,
  "OLMo 2 uses a Llama-style architecture") is captured at
  relate stage as an `inspired_by` edge between two real model
  nodes, NOT as a separate node here. Let those drops stay
  dropped.

- **Does the lattice over-decompose?** Surface variants of one
  artifact should be one item with multiple aliases.
  Eval-harness reformulations of one benchmark should fold
  into the canonical with the harness as a facet. Date-snapshot
  variants of an API model should fold into the family
  canonical with the date as a facet.

- **Does the dropped pile contain false negatives?** Some
  dropped items will look like real artifacts the organize
  planner missed. Independently verify before accepting any
  drop reason that names an unrelated project, different
  community, or library namespace — those reasons are common
  cover for the planner not having found the right anchor.

- **Are there items with `description: null` that have a
  fetchable source?** Organize sometimes leaves a description
  empty when its first attempt failed (e.g., raw README returned
  401/403 because the model is gated, planner ran out of budget
  for that bucket, or transient network error). Try again — the
  source ladder is the same as organize's:
  1. `curl` the raw README at `<page-url>/raw/main/README.md`;
  2. **WebFetch the page URL** when raw returns 401/403
     (rendered page exposes the card text even for gated repos
     like the Meta Llama family);
  3. HF API (`huggingface.co/api/models/<owner>/<repo>`) for
     `pipeline_tag` / `library_name` / `tags` / `cardData`;
  4. GitHub README first paragraph (if no HF card);
  5. arXiv abstract first sentence (if paper-anchored);
  6. WebSearch on `"<formal_name>"` for a release blog,
     vendor docs, or HF blog post.

  Same writing rules: target-independent, third-person, ≤3
  sentences. Leave `description: null` only after the full
  ladder fails.

These questions overlap. Walk the artifact looking through all
of them; many edits will address several at once. The universal
rule (verify before acting) trumps any pattern-match.

## Edits available to you

Anything that produces a cleaner lattice in the same schema:

- **Rename** an item's `formal_name` to its canonical form when
  the current name is fabricated, mistyped, or doesn't follow
  the convention for its kind. Carry the old form into
  `aliases`.
- **Split** an item that absorbs two distinct artifacts
  (different release events, different weights, different
  papers).
- **Merge** two items that are really the same artifact under
  different surface forms.
- **Move** an item from one family to another when it was
  bucketed by substring rather than by lineage.
- **Split a family** that spans multiple orgs / artifact kinds /
  product lines. Document each split briefly in `notes`.
- **Merge two families** that turn out to refer to the same
  product line.
- **Add a discriminating facet** when siblings carry identical
  identity. Extend the family's `identity_keys` if needed.
- **Drop an item** that shouldn't exist (invented placeholder,
  unverifiable name, item the lattice has no canonical anchor
  for).
- **Restore a dropped name** as a new item if you find a real
  anchor matching the 3-form criterion (open HF/GitHub release;
  vendor docs page; paper anchor).
- **Promote a dropped name to `gated[]`** if you confirm the
  artifact exists at a 401/403 HF URL.
- **Augment a dropped reason** with `[audit-confirmed]` plus
  the evidence you collected, when recheck confirms no anchor
  exists.
- **Fill a missing description** when an item has
  `description: null` AND a fetchable source exists. Apply the
  source ladder above (raw README → page WebFetch → HF API →
  GitHub README → arXiv abstract → WebSearch). Keep the same
  rules organize uses: target-independent, third-person, ≤3
  sentences, no fabrication. Skip silently when the ladder
  exhausts (description stays null).

When you make any non-trivial edit, briefly note it in the
top-level `notes` field — what changed and why. The notes are
what an operator scans to understand what the audit pass did.

## When uncertain

Dispatch a sonnet subagent to investigate one specific case
(e.g., "is `<name>` a real release? what's its canonical URL?
what role does it play?"). Subagents have none of your
context — transcribe the relevant rule fragments verbatim.
Cap total dispatched investigations at ~10 per pass — this is
a check-and-fix pass, not a full re-organize. For broader
sweeps (e.g., "rename every bare-benchmark name to its HF
canonical path"), apply the rule yourself in batch using
`Bash` rather than dispatching one subagent per item.

## Recheck investigations — what to actually do

For each `dropped[]` entry you decide to RECHECK:

1. If the entry has `attempted_canonicals[]`, do NOT re-try
   those exact paths — organize already verified they don't
   exist. Search adjacent variants instead.

2. Standard search patterns:
   ```
   curl -sL https://huggingface.co/api/models?author=<org>
   curl -sL https://huggingface.co/api/datasets?author=<org>
   curl -sL 'https://huggingface.co/api/models?search=<name>&limit=20'
   curl -sL 'https://huggingface.co/api/datasets?search=<name>&limit=20'
   ```

3. If the dropped name might be an internal codename
   referenced in a primary repo's README, grep the
   already-fetched primary repo for the slug.

4. For API-only models, check the vendor's docs page directly.

5. After HEAD-200 on any candidate, fetch the card and verify
   the content matches the input context — same
   "verify-before-adopting" discipline organize was supposed
   to apply.

## Output schema

Same shape as the organize artifact, plus an OPTIONAL top-level
`gated[]` array. Required fields unchanged from organize:
`groups[]` with `family`, `identity_keys`, `items[]`; each item
with `kind`, `formal_name`, `identity`, `aliases`, `links`,
`description`. The `subsets[]` field on dataset items is carried
through from the Python pre-pass — leave it alone unless you
have a specific correction.

Top-level optional fields:
- `notes` — brief summary of what you changed, with counts
  ("recheck: restored N, promoted M to gated, kept K dropped;
  split L families; renamed P formal_names; ...").
- `dropped` — the union of organize-dropped entries you didn't
  restore or promote, plus any new drops you make. Same schema
  as organize's `dropped[]`.
- `gated` — entries promoted from `dropped[]` for which you
  confirmed a 401/403 gated URL. Each entry: `{name, kind,
  reason, gated_url, attempted_canonicals?}`.

```json
{
  "groups": [
    {
      "family": "...",
      "identity_keys": [...],
      "items": [
        {"kind": "...", "formal_name": "...", "identity": {...},
         "aliases": [...], "links": [...], "description": "..."}
      ]
    }
  ],
  "gated": [
    {"name": "...", "kind": "...", "reason": "...",
     "gated_url": "...", "attempted_canonicals": []}
  ],
  "dropped": [...],
  "notes": "..."
}
```

The output is the WHOLE revised lattice, not a diff — every
family that should remain in the lattice must appear in the
output, even if unchanged.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
