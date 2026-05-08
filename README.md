# MTG Agentic Card Search (Python)

## 1) Backend setup (local dev)

Use an **editable** install so the standalone backend CLI (`mtg-card-search-backend`) always runs your current working tree instead of a one-off wheel copy.

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -e ./backend
```

### Create database from Scryfall bulk JSON

Download a Scryfall bulk export (JSON, top-level array of card objects) from **[Bulk Data · API Documentation](https://scryfall.com/docs/api/bulk-data)**. For this project, use the **All Cards** file (every language; largest), or **Default Cards** if you only need English or a smaller download. URLs on that page change daily; you can also discover the current `download_uri` via `GET https://api.scryfall.com/bulk-data` and fetch the **all_cards** (or **default_cards**) entry.

Run from the **repository root** with **`source backend/.venv/bin/activate`**.

Examples:
```bash
python3 -m backend.database.bootstrap_db /path/to/all-cards.json
python3 -m backend.database.bootstrap_db /path/to/all-cards.json --limit 200
python3 -m backend.database.bootstrap_db /path/to/all-cards.json --skip-embeddings
python3 -m backend.database.bootstrap_db /path/to/all-cards.json --db backend/database/all-cards.db
```

| Argument | Description |
|----------|-------------|
| `json_path` (positional) | Path to the Scryfall bulk **all-cards** JSON (top-level array). Required. |
| `--db PATH` | SQLite output file. Default: `backend/database/all-cards.db`. |
| `--limit N` | Import only the first **N** card objects; omit for the full bulk file. |
| `--no-progress` | Disable the tqdm progress bar (e.g. CI or plain logs). |
| `--skip-embeddings` | Do not call the embeddings API or fill `card_embeddings` (no `OPENROUTER_API_KEY` needed). Schema still includes the table. |
| `--embedding-batch-size N` | Card texts per embeddings API request. Default: **64**. |
| `--embedding-model NAME` | Embedding model id. Default: **`text-embedding-3-small`**. |
| `--embedding-dimensions D` | Vector size for the model. Default: **1536**. |

This creates (by default):

```text
backend/database/all-cards.db
```

Bootstrap also builds **`cards_fts`** (FTS5) and **`cards_vec`** (`vec0`) when possible. If you imported data earlier, rebuild indices:

```bash
python3 -m backend.database.build_search_indices
```

Optional index-refresh overrides:

- `--embedding-model` and `--embedding-dimensions` take precedence over built-in defaults.
- `--embedding-batch-size` controls API batch size for refresh runs.

## 2) Frontend setup (local dev)

```bash
python3 -m venv frontend/.venv
source frontend/.venv/bin/activate
pip install --upgrade pip
pip install -r frontend/requirements.txt
pip install -e ./frontend
```

## 3) Configure environment

Backend env:

```bash
cp backend/.env.example backend/.env
```

Frontend env:

```bash
cp frontend/.env.example frontend/.env
```

Then **edit** those files and replace any placeholders. **Mandatory** values:

| File | Variable | Notes |
|------|----------|--------|
| `backend/.env` | **`OPENROUTER_API_KEY`** | Required to start the API (agent + embedding-related tooling). The process exits if this is missing or left as the example placeholder. |
| `frontend/.env` | **`API_BASE_URL`** | Must match where the backend listens (default `http://127.0.0.1:8000` is correct for local dev in section 4). Update if the API uses another host or port. |

Everything else in the `.env.example` files is optional; see inline comments. Optional frontend overrides include `API_URL`, `FRONTEND_HOST`, and `FRONTEND_PORT`.

## 4) Run backend and frontend (dev from repo root only)

Always run commands from the **repository root**. Use **two terminals** (or run the API in the background): activate **`backend/.venv`** for the API and **`frontend/.venv`** for the UI.

### Backend (FastAPI)

```bash
source backend/.venv/bin/activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```


#### Backend CLI (`mtg-card-search-backend`)

After `pip install -e ./backend` (section 1), you can use the installed CLI instead of raw uvicorn:

```bash
source backend/.venv/bin/activate
mtg-card-search-backend --help
```

| Flag | Effect |
|------|--------|
| `--profiling-enabled` | Sets `PROFILING_ENABLED=1`: per-request timings on stdout (`[profile]…`). |
| `--log-file PATH` | Tee **stdout and stderr** to `PATH` (append, UTF-8); console output unchanged. |

Example:

```bash
source backend/.venv/bin/activate
mtg-card-search-backend --profiling-enabled --log-file ./backend.log
```

With **raw uvicorn**, there is no `--log-file` on the server: use shell redirection or set env vars yourself:

```bash
source backend/.venv/bin/activate
PROFILING_ENABLED=1 uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 \
  2>&1 | tee -a backend.log
```

### Frontend (Gradio)

In a **second** terminal:

```bash
source frontend/.venv/bin/activate
uvicorn frontend.main:app --reload --reload-dir frontend --host 127.0.0.1 --port 7860
```

Open the UI at the printed local URL (usually `http://127.0.0.1:7860`).

#### Frontend CLI (`mtg-card-search-frontend`)

After `pip install -e ./frontend` (section 2), you can launch the same Gradio UI without uvicorn. Bind address and port come from **`frontend/.env`** (`FRONTEND_HOST`, `FRONTEND_PORT`; defaults `127.0.0.1` and `7860`). **`API_BASE_URL`** must point at your running backend (section 3).

```bash
source frontend/.venv/bin/activate
mtg-card-search-frontend
```

Listen on all interfaces (e.g. access from another machine on your LAN):

```bash
source frontend/.venv/bin/activate
FRONTEND_HOST=0.0.0.0 mtg-card-search-frontend
```

Point at a backend on another host or port for one session (overrides `frontend/.env` for this process):

```bash
source frontend/.venv/bin/activate
API_BASE_URL=http://192.168.1.10:8000 mtg-card-search-frontend
```

## 5) Build wheel packages for remote deployment

Build **installable wheels** on your development machine, copy them to a remote host, and `pip install` there (dependencies are pulled from PyPI unless you vendor them; see below).

### Build (repository root)

Install the build frontend once (use the same environment you use for packaging, e.g. `source backend/.venv/bin/activate`):

```bash
pip install build
```

Produce wheels (outputs go to `backend/dist/` and `frontend/dist/`):

```bash
python3 -m build backend
python3 -m build frontend
```

You should see files such as:

```text
backend/dist/mtg_card_search_backend-<version>-py3-none-any.whl
frontend/dist/mtg_agentic_frontend-<version>-py3-none-any.whl
```

### Deploy on a remote machine

If you have a **full clone** on the server, use the **repository root** and the same venv layout as sections 1–2; only the install step becomes `pip install ./path/to/*.whl` instead of `pip install -e ./backend`. If you copied **wheels only**, run `pip install` from any convenient directory.

1. Copy the `.whl` files (and your `backend/.env` / `frontend/.env` or equivalent secrets) to the server.
2. Ensure the backend has **`all-cards.db`** (or set **`DATABASE_PATH`**) and a Python/SQLite stack that supports **loadable extensions** if you use **sqlite-vec** (same constraints as local dev).
3. Install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install /path/to/mtg_card_search_backend-*-py3-none-any.whl
pip install /path/to/mtg_agentic_frontend-*-py3-none-any.whl
```

Then use **`mtg-card-search-backend`** and **`mtg-card-search-frontend`** as usual, or run **`uvicorn backend.main:app`** / **`uvicorn frontend.main:app`** with the same environment variables as in section 4.

### Offline / vendor dependencies (optional)

To install on a host **without** PyPI access, on the build machine collect the project wheel **and** all dependencies:

```bash
mkdir -p dist/vendor
pip wheel ./backend -w dist/vendor
pip wheel ./frontend -w dist/vendor
```

Copy the whole `dist/vendor/` directory to the remote host, then:

```bash
pip install --no-index --find-links /path/to/vendor \
  /path/to/vendor/mtg_card_search_backend-*-py3-none-any.whl
pip install --no-index --find-links /path/to/vendor \
  /path/to/vendor/mtg_agentic_frontend-*-py3-none-any.whl
```

Resolve any platform-specific wheels (e.g. native SQLite/extension builds) on an environment similar to production if needed.
