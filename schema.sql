PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  id             TEXT PRIMARY KEY,
  stage          TEXT NOT NULL,
  seed           TEXT,
  parent_run_id  TEXT REFERENCES runs(id) ON DELETE SET NULL,
  label          TEXT,
  attrs          TEXT NOT NULL DEFAULT '{}',
  started_at     TEXT NOT NULL,
  ended_at       TEXT
);
CREATE INDEX IF NOT EXISTS runs_stage_idx ON runs(stage);
CREATE INDEX IF NOT EXISTS runs_seed_idx ON runs(seed);

CREATE TABLE IF NOT EXISTS sources (
  id             TEXT PRIMARY KEY,
  content_hash   TEXT NOT NULL UNIQUE,
  content_type   TEXT,
  storage_ref    TEXT,
  title          TEXT,
  canonical_url  TEXT,
  remote_url     TEXT,
  commit_sha     TEXT,
  attrs          TEXT NOT NULL DEFAULT '{}',
  fetched_at     TEXT,
  created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_remote_sha_idx ON sources(remote_url, commit_sha);

CREATE TABLE IF NOT EXISTS source_urls (
  source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  url            TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  PRIMARY KEY(source_id, url)
);
CREATE INDEX IF NOT EXISTS source_urls_url_idx ON source_urls(url);

CREATE TABLE IF NOT EXISTS source_commits (
  source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  commit_sha     TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  PRIMARY KEY(source_id, commit_sha)
);
CREATE INDEX IF NOT EXISTS source_commits_sha_idx ON source_commits(commit_sha);

CREATE TABLE IF NOT EXISTS batches (
  id                   TEXT PRIMARY KEY,
  label                TEXT,
  summary              TEXT,
  content_fingerprint  TEXT NOT NULL UNIQUE,
  attrs                TEXT NOT NULL DEFAULT '{}',
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_sources (
  batch_id  TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  ordinal   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (batch_id, source_id)
);
CREATE INDEX IF NOT EXISTS batch_sources_source_idx ON batch_sources(source_id);

CREATE TABLE IF NOT EXISTS batch_artifacts (
  batch_id       TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  stage          TEXT NOT NULL,
  artifact_path  TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','complete','failed','superseded')),
  run_id         TEXT REFERENCES runs(id) ON DELETE SET NULL,
  attrs          TEXT NOT NULL DEFAULT '{}',
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  PRIMARY KEY (batch_id, stage)
);
CREATE INDEX IF NOT EXISTS batch_artifacts_stage_idx ON batch_artifacts(stage, status);

CREATE TABLE IF NOT EXISTS mentions (
  id                  TEXT PRIMARY KEY,
  batch_id            TEXT REFERENCES batches(id) ON DELETE SET NULL,
  source_id           TEXT REFERENCES sources(id) ON DELETE SET NULL,
  kind                TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  surface             TEXT NOT NULL,
  surface_key         TEXT NOT NULL,
  identity_json       TEXT NOT NULL,
  identity_key        TEXT NOT NULL,
  descriptors_json    TEXT NOT NULL DEFAULT '{}',
  aliases_json        TEXT NOT NULL DEFAULT '[]',
  subsets_json        TEXT NOT NULL DEFAULT '[]',
  context_roles_json  TEXT NOT NULL DEFAULT '[]',
  atoms_json          TEXT NOT NULL DEFAULT '[]',
  referent_scope      TEXT NOT NULL DEFAULT 'ambiguous'
                        CHECK (referent_scope IN ('entity','concept','ambiguous')),
  links_json          TEXT NOT NULL DEFAULT '[]',
  concept_path_json   TEXT NOT NULL DEFAULT '[]',
  aux_json            TEXT NOT NULL DEFAULT '{}',
  relationships_json  TEXT NOT NULL DEFAULT '[]',
  anchors_json        TEXT NOT NULL DEFAULT '[]',
  description         TEXT,
  notes               TEXT,
  attrs               TEXT NOT NULL DEFAULT '{}',
  status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active','dropped','repaired')),
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS mentions_identity_idx ON mentions(kind, identity_key);
CREATE INDEX IF NOT EXISTS mentions_surface_idx ON mentions(surface_key);
CREATE INDEX IF NOT EXISTS mentions_batch_idx ON mentions(batch_id);

CREATE TABLE IF NOT EXISTS mention_violations (
  id            TEXT PRIMARY KEY,
  code          TEXT NOT NULL,
  severity      TEXT NOT NULL DEFAULT 'error'
                  CHECK (severity IN ('error','warning')),
  subject_key   TEXT,
  details_json  TEXT NOT NULL DEFAULT '{}',
  status        TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open','resolved','ignored')),
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS mention_violations_code_idx ON mention_violations(code, status);

-- Mentions that failed extract-time validation in ways audit cannot
-- repair (no anchors, malformed shape, missing surface, invalid kind).
-- Sibling mentions in the same batch still commit; these land here for
-- later review. Storage layer schema constraints (e.g., kind IN
-- ('model','dataset')) make this routing necessary; soft errors like
-- bad link shape get repaired in-place by commit_mentions and never
-- reach this table.
CREATE TABLE IF NOT EXISTS rejected_mentions (
  id                 TEXT PRIMARY KEY,
  batch_id           TEXT,
  run_id             TEXT,
  artifact_index     INTEGER,
  surface            TEXT,
  reason_codes_json  TEXT NOT NULL DEFAULT '[]',
  errors_json        TEXT NOT NULL DEFAULT '[]',
  raw_json           TEXT NOT NULL,
  status             TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending','reviewed','reingested','dismissed')),
  created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rejected_mentions_batch_idx ON rejected_mentions(batch_id, status);
CREATE INDEX IF NOT EXISTS rejected_mentions_status_idx ON rejected_mentions(status);

CREATE TABLE IF NOT EXISTS link_checks (
  id            TEXT PRIMARY KEY,
  cluster_key   TEXT NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  link_kind     TEXT NOT NULL,
  link_value    TEXT NOT NULL,
  url           TEXT NOT NULL,
  ok            INTEGER NOT NULL DEFAULT 0,
  status_code   INTEGER,
  error         TEXT,
  checked_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS link_checks_cluster_idx ON link_checks(cluster_key);
CREATE INDEX IF NOT EXISTS link_checks_url_idx ON link_checks(url);

CREATE TABLE IF NOT EXISTS entity_descriptions (
  entity_key      TEXT PRIMARY KEY,
  kind            TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  display_name    TEXT NOT NULL,
  links_json      TEXT NOT NULL DEFAULT '[]',
  description     TEXT NOT NULL DEFAULT '',
  metadata_json   TEXT NOT NULL DEFAULT '{}',
  source_json     TEXT NOT NULL DEFAULT '{}',
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entity_descriptions_kind_idx ON entity_descriptions(kind);

CREATE TABLE IF NOT EXISTS hf_metadata (
  link_key         TEXT PRIMARY KEY,
  link_type        TEXT NOT NULL,
  link_value       TEXT NOT NULL,
  kind             TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  ok               INTEGER NOT NULL DEFAULT 0,
  repo_url         TEXT,
  readme_url       TEXT,
  api_url          TEXT,
  metadata_json    TEXT NOT NULL DEFAULT '{}',
  card_data_json   TEXT NOT NULL DEFAULT '{}',
  configs_json     TEXT NOT NULL DEFAULT '[]',
  collections_json TEXT NOT NULL DEFAULT '[]',
  relationships_json TEXT NOT NULL DEFAULT '[]',
  description      TEXT NOT NULL DEFAULT '',
  error            TEXT,
  fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS hf_metadata_link_idx ON hf_metadata(link_type, link_value);

CREATE TABLE IF NOT EXISTS family_policies (
  id              TEXT PRIMARY KEY,
  kind            TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  root            TEXT NOT NULL,
  policy_json     TEXT NOT NULL DEFAULT '{}',
  anchors_json    TEXT NOT NULL DEFAULT '[]',
  source          TEXT NOT NULL DEFAULT 'review',
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS family_policies_root_idx ON family_policies(kind, root);

CREATE TABLE IF NOT EXISTS entity_relationships (
  id                TEXT PRIMARY KEY,
  source_entity_key TEXT,
  source_link_json  TEXT NOT NULL DEFAULT '{}',
  relation          TEXT NOT NULL,
  target_link_json  TEXT NOT NULL DEFAULT '{}',
  target_name       TEXT,
  anchors_json      TEXT NOT NULL DEFAULT '[]',
  metadata_json     TEXT NOT NULL DEFAULT '{}',
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entity_relationships_source_idx ON entity_relationships(source_entity_key);

CREATE TABLE IF NOT EXISTS lattice_nodes (
  id                   TEXT PRIMARY KEY,
  node_key             TEXT NOT NULL UNIQUE,
  kind                 TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  node_type            TEXT NOT NULL DEFAULT 'concept'
                         CHECK (node_type IN ('concept','entity')),
  identity_json        TEXT NOT NULL,
  concept_path_json    TEXT NOT NULL DEFAULT '[]',
  display_name         TEXT NOT NULL,
  aliases_json         TEXT NOT NULL DEFAULT '[]',
  descriptors_json     TEXT NOT NULL DEFAULT '{}',
  links_json           TEXT NOT NULL DEFAULT '[]',
  verified_links_json  TEXT NOT NULL DEFAULT '[]',
  aux_json             TEXT NOT NULL DEFAULT '{}',
  description          TEXT,
  occurrence_count     INTEGER NOT NULL DEFAULT 0,
  projection           INTEGER NOT NULL DEFAULT 0,
  flags_json           TEXT NOT NULL DEFAULT '[]',
  created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lattice_nodes_kind_idx ON lattice_nodes(kind);

CREATE TABLE IF NOT EXISTS lattice_edges (
  parent_node_key TEXT NOT NULL REFERENCES lattice_nodes(node_key) ON DELETE CASCADE,
  child_node_key  TEXT NOT NULL REFERENCES lattice_nodes(node_key) ON DELETE CASCADE,
  rationale       TEXT,
  PRIMARY KEY(parent_node_key, child_node_key),
  CHECK(parent_node_key <> child_node_key)
);
CREATE INDEX IF NOT EXISTS lattice_edges_child_idx ON lattice_edges(child_node_key);
