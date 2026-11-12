# Private configuration and data

API keys, database credentials, organization schemas, example records, embeddings,
uploads, chat/session stores, generated results, logs, and Python bytecode are not
distributed in this repository. Sensitive files were also removed from Git history.

## Configuration

Set `OPENAI_API_KEY` in the process environment or inject it from a secret manager.
There is no hardcoded API-key fallback. `.env.example` lists configuration names;
copy it to an ignored `.env` for local use and load it into the process environment
before launching the application. Only entry points that explicitly load dotenv
read that file automatically.

Provide authorized database configuration and application data separately, outside
version control. `config.examples/ORACLE_INFO.example.json` is a format example only.
Replace its placeholder fields locally and point `ORACLE_JSON_PATH` at the private
file. Use `DB_LIST_PATH`, `TABLES_JSON_PATH`, `METADATA_DB_PATH`, and
`FAISS_INDEX_PATH` where supported to select private data locations. The table
description agent also accepts `--oracle-info`, `--table-names`, `--time-column`,
`--comments-db`, and `--context` for externally supplied files.

Configure model and service URLs with `EMBEDDING_BASE_URL`, `RERANK_URL`, and
`OPENAI_BASE_URL` where supported. Defaults refer to local services. If Oracle
Thick mode is needed, configure `ORACLE_CLIENT_LIB_DIR` for that machine.
Set database/Redis passwords through environment variables; no shared default
password is provided.

## Generated artifacts

Applications that need schemas, FAISS indices, SQLite files, or uploaded data
require authorized private inputs before starting. No original dataset is bundled.
Regenerate vector indices from those inputs and rebuild frontend assets from
`frontend/` using `npm ci` and `npm run build` where that frontend is present.
Do not force-add ignored files or commit credentials, caches, logs, or datasets.

## Existing clones

Clone this repository again after the history rewrite. Merging old branches can
reintroduce removed data. Rotate any credentials that were previously committed;
rewriting history does not revoke a credential or erase other people's copies.
