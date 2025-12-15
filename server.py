"""
Caia Model Registry

A simple model registry for tracking checkpoints, configs, and metrics.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8001

v2 notes:
  - No checkpoint loading/unpickling in the API server (no torch.load()).
  - API key auth required for all endpoints except / and /health.
"""

from __future__ import annotations

import hmac
import os
import sqlite3
import json
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterator, Literal

from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Config
# =============================================================================

ModelStatus = Literal["experimental", "staging", "production", "archived"]

DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200

DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000
DEFAULT_MAX_HASH_BYTES = 256 * 1024 * 1024  # 256 MiB
DEFAULT_MAX_METRICS_BODY_BYTES = 64 * 1024
DEFAULT_MAX_EVENT_PAYLOAD_BYTES = 64 * 1024
DEFAULT_MAX_WRITE_BODY_BYTES = 256 * 1024


def _env_csv(name: str) -> List[str]:
    value = os.getenv(name)
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_db_path() -> Path:
    raw = os.getenv("CAIA_REGISTRY_DB_PATH")
    if raw:
        return Path(raw)
    return Path(__file__).parent / "registry.db"


def _get_allowed_cors_origins() -> List[str]:
    configured = os.getenv("CAIA_REGISTRY_CORS_ORIGINS")
    if configured is None:
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
        ]
    return _env_csv("CAIA_REGISTRY_CORS_ORIGINS")


def _get_api_keys() -> List[str]:
    keys: List[str] = []
    keys.extend(_env_csv("CAIA_REGISTRY_API_KEYS"))
    single = os.getenv("CAIA_REGISTRY_API_KEY")
    if single and single.strip():
        keys.append(single.strip())
    return [k for k in keys if k]


DB_PATH = _get_db_path()
ALLOW_LOCAL_PATHS = _env_bool("CAIA_REGISTRY_ALLOW_LOCAL_PATHS", default=False)
EXPOSE_LOCAL_PATHS = _env_bool("CAIA_REGISTRY_EXPOSE_LOCAL_PATHS", default=False)
TRUST_ACTOR_HEADER = _env_bool("CAIA_REGISTRY_TRUST_ACTOR_HEADER", default=False)
SQLITE_BUSY_TIMEOUT_MS = _env_int("CAIA_REGISTRY_SQLITE_BUSY_TIMEOUT_MS", DEFAULT_SQLITE_BUSY_TIMEOUT_MS)
MAX_HASH_BYTES = _env_int("CAIA_REGISTRY_MAX_HASH_BYTES", DEFAULT_MAX_HASH_BYTES)
MAX_METRICS_BODY_BYTES = _env_int("CAIA_REGISTRY_MAX_METRICS_BODY_BYTES", DEFAULT_MAX_METRICS_BODY_BYTES)
MAX_EVENT_PAYLOAD_BYTES = _env_int("CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES", DEFAULT_MAX_EVENT_PAYLOAD_BYTES)
MAX_WRITE_BODY_BYTES = _env_int("CAIA_REGISTRY_MAX_WRITE_BODY_BYTES", DEFAULT_MAX_WRITE_BODY_BYTES)
PROMOTION_REQUIRED_SUITE = (os.getenv("CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE") or "").strip() or None

if MAX_METRICS_BODY_BYTES <= 0:
    MAX_METRICS_BODY_BYTES = DEFAULT_MAX_METRICS_BODY_BYTES
if MAX_EVENT_PAYLOAD_BYTES <= 0:
    MAX_EVENT_PAYLOAD_BYTES = DEFAULT_MAX_EVENT_PAYLOAD_BYTES
if MAX_WRITE_BODY_BYTES <= 0:
    MAX_WRITE_BODY_BYTES = DEFAULT_MAX_WRITE_BODY_BYTES


# =============================================================================
# Auth
# =============================================================================

@dataclass(frozen=True)
class AuthContext:
    actor: str


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def _default_actor_from_key(api_key: str) -> str:
    prefix = api_key[:4]
    return f"key:{prefix}…" if prefix else "key"


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_actor: Optional[str] = Header(default=None, alias="X-Actor"),
) -> AuthContext:
    configured = _get_api_keys()
    if not configured:
        raise HTTPException(status_code=503, detail="API key auth not configured (set CAIA_REGISTRY_API_KEY(S))")

    provided = (x_api_key or "").strip() or _extract_bearer_token(authorization)
    if not provided:
        raise HTTPException(status_code=401, detail="Missing API key")

    if not any(hmac.compare_digest(provided, k) for k in configured):
        raise HTTPException(status_code=401, detail="Invalid API key")

    actor_header = (x_actor or "").strip()
    actor = actor_header if (TRUST_ACTOR_HEADER and actor_header) else _default_actor_from_key(provided)
    actor = actor[:200]
    request.state.actor = actor
    return AuthContext(actor=actor)


# =============================================================================
# Database
# =============================================================================

def _configure_sqlite_connection(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
    _configure_sqlite_connection(conn)
    return conn


def get_db() -> Iterator[sqlite3.Connection]:
    conn = _connect_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def _ensure_models_columns(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "models")
    additions: List[tuple[str, str]] = [
        ("artifact_uri", "TEXT"),
        ("checkpoint_sha256", "TEXT"),
        ("checkpoint_size_bytes", "INTEGER"),
        ("config_sha256", "TEXT"),
        ("config_size_bytes", "INTEGER"),
        ("run_id", "TEXT"),
        ("git_commit", "TEXT"),
        ("created_by", "TEXT"),
        ("source_host", "TEXT"),
        ("frozen", "INTEGER NOT NULL DEFAULT 0"),
    ]
    for name, decl in additions:
        if name in cols:
            continue
        conn.execute(f"ALTER TABLE models ADD COLUMN {name} {decl};")

    cols = _table_columns(conn, "models")
    if "artifact_uri" in cols:
        conn.execute(
            """
            UPDATE models
            SET artifact_uri = COALESCE(NULLIF(artifact_uri, ''), 'legacy:' || id)
            WHERE artifact_uri IS NULL OR artifact_uri = ''
            """
        )
    if "frozen" in cols:
        conn.execute("UPDATE models SET frozen = 0 WHERE frozen IS NULL")


def _ensure_model_events_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT,
            actor TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_model_events_model_id ON model_events(model_id);
        CREATE INDEX IF NOT EXISTS idx_model_events_created_at ON model_events(created_at);
        """
    )


def _ensure_indexes_and_triggers(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
        CREATE INDEX IF NOT EXISTS idx_models_updated_at ON models(updated_at);
        CREATE INDEX IF NOT EXISTS idx_models_run_id ON models(run_id);

        DROP TRIGGER IF EXISTS trg_models_set_updated_at;
        CREATE TRIGGER trg_models_set_updated_at
        AFTER UPDATE ON models
        FOR EACH ROW
        WHEN NEW.updated_at = OLD.updated_at
        BEGIN
            UPDATE models SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'now') WHERE id = OLD.id;
        END;
        """
    )


def init_db():
    """Initialize database schema"""
    conn = _connect_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'experimental',

            -- Artifact identity (preferred) and legacy local paths (dev only)
            artifact_uri TEXT,
            checkpoint_path TEXT NOT NULL,
            config_path TEXT,

            checkpoint_sha256 TEXT,
            checkpoint_size_bytes INTEGER,
            config_sha256 TEXT,
            config_size_bytes INTEGER,

            -- Provenance
            run_id TEXT,
            git_commit TEXT,
            created_by TEXT,
            source_host TEXT,

            frozen INTEGER NOT NULL DEFAULT 0,

            -- Architecture info
            d_model INTEGER,
            n_layers INTEGER,
            n_heads INTEGER,
            vocab_size INTEGER,
            params INTEGER,

            -- Training info
            training_step INTEGER,
            training_loss REAL,
            dataset TEXT,

            -- Metadata
            description TEXT,
            tags TEXT,  -- JSON array
            metrics TEXT,  -- JSON object

            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(name, version)
        );
        """
    )
    _ensure_models_columns(conn)
    _ensure_model_events_table(conn)
    _ensure_indexes_and_triggers(conn)
    conn.commit()
    conn.close()


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str
    version: str
    status: ModelStatus = "experimental"

    artifact_uri: Optional[str] = None
    local_checkpoint_path: Optional[str] = Field(default=None, alias="checkpoint_path")
    local_config_path: Optional[str] = Field(default=None, alias="config_path")

    checkpoint_sha256: Optional[str] = None
    checkpoint_size_bytes: Optional[int] = None
    config_sha256: Optional[str] = None
    config_size_bytes: Optional[int] = None

    run_id: Optional[str] = None
    git_commit: Optional[str] = None
    created_by: Optional[str] = None
    source_host: Optional[str] = None

    frozen: bool = False

    # Architecture
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    params: Optional[int] = None

    # Training
    training_step: Optional[int] = None
    training_loss: Optional[float] = None
    dataset: Optional[str] = None

    # Metadata
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None

    config_json: Optional[Dict[str, Any]] = None


class ModelUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    status: Optional[ModelStatus] = None
    artifact_uri: Optional[str] = None

    local_checkpoint_path: Optional[str] = Field(default=None, alias="checkpoint_path")
    local_config_path: Optional[str] = Field(default=None, alias="config_path")

    checkpoint_sha256: Optional[str] = None
    checkpoint_size_bytes: Optional[int] = None
    config_sha256: Optional[str] = None
    config_size_bytes: Optional[int] = None

    run_id: Optional[str] = None
    git_commit: Optional[str] = None
    created_by: Optional[str] = None
    source_host: Optional[str] = None

    frozen: Optional[bool] = None

    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    params: Optional[int] = None

    training_step: Optional[int] = None
    training_loss: Optional[float] = None
    dataset: Optional[str] = None

    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    status: ModelStatus

    artifact_uri: str
    checkpoint_sha256: Optional[str] = None
    checkpoint_size_bytes: Optional[int] = None
    config_sha256: Optional[str] = None
    config_size_bytes: Optional[int] = None

    run_id: Optional[str] = None
    git_commit: Optional[str] = None
    created_by: Optional[str] = None
    source_host: Optional[str] = None

    frozen: bool

    local_checkpoint_path: Optional[str] = None
    local_config_path: Optional[str] = None

    d_model: Optional[int]
    n_layers: Optional[int]
    n_heads: Optional[int]
    vocab_size: Optional[int]
    params: Optional[int]

    training_step: Optional[int]
    training_loss: Optional[float]
    dataset: Optional[str]

    description: Optional[str]
    tags: Optional[List[str]]
    metrics: Optional[Dict[str, Any]]

    created_at: str
    updated_at: str


class ModelEventResponse(BaseModel):
    id: int
    model_id: int
    event_type: str
    payload: Optional[Dict[str, Any]] = None
    actor: Optional[str] = None
    created_at: str


class ModelMetricsWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str
    metrics: Dict[str, Any]

    dataset: Optional[str] = None
    eval_commit: Optional[str] = None
    eval_config: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def row_to_model(row: sqlite3.Row) -> ModelResponse:
    """Convert database row to ModelResponse"""
    local_checkpoint_path: Optional[str] = row["checkpoint_path"] if EXPOSE_LOCAL_PATHS else None
    local_config_path: Optional[str] = row["config_path"] if EXPOSE_LOCAL_PATHS else None
    artifact_uri = row["artifact_uri"] or f"legacy:{row['id']}"
    tags = _json_loads(row["tags"])
    if not isinstance(tags, list):
        tags = None
    metrics = _json_loads(row["metrics"])
    if not isinstance(metrics, dict):
        metrics = None

    return ModelResponse(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        status=row["status"],
        artifact_uri=artifact_uri,
        checkpoint_sha256=row["checkpoint_sha256"],
        checkpoint_size_bytes=row["checkpoint_size_bytes"],
        config_sha256=row["config_sha256"],
        config_size_bytes=row["config_size_bytes"],
        run_id=row["run_id"],
        git_commit=row["git_commit"],
        created_by=row["created_by"],
        source_host=row["source_host"],
        frozen=bool(row["frozen"]),
        local_checkpoint_path=local_checkpoint_path,
        local_config_path=local_config_path,
        d_model=row["d_model"],
        n_layers=row["n_layers"],
        n_heads=row["n_heads"],
        vocab_size=row["vocab_size"],
        params=row["params"],
        training_step=row["training_step"],
        training_loss=row["training_loss"],
        dataset=row["dataset"],
        description=row["description"],
        tags=tags,
        metrics=metrics,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value)


def _json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_sha256(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = value.strip().lower()
    if candidate.startswith("sha256:"):
        candidate = candidate.split(":", 1)[1].strip()
    if candidate == "":
        return None
    if len(candidate) != 64 or any(c not in "0123456789abcdef" for c in candidate):
        raise HTTPException(status_code=400, detail="Invalid sha256 (expected 64 hex characters)")
    return candidate


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config_json must be an object")
    return data


def _extract_arch_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    arch = cfg.get("architecture", cfg)
    if not isinstance(arch, dict):
        return {}
    return {
        "d_model": arch.get("d_model"),
        "n_layers": arch.get("n_layers"),
        "n_heads": arch.get("n_heads"),
        "vocab_size": arch.get("vocab_size"),
    }


def _sha256_and_size(path: Path, *, max_hash_bytes: int) -> tuple[Optional[str], Optional[int]]:
    try:
        size = path.stat().st_size
    except OSError:
        return None, None

    if size > max_hash_bytes:
        return None, size

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest(), size


def _actor_from_request(request: Request) -> str:
    return getattr(request.state, "actor", None) or "unknown"


async def enforce_metrics_body_limit(request: Request) -> None:
    raw = request.headers.get("content-length")
    if raw:
        try:
            if int(raw) > MAX_METRICS_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large (max {MAX_METRICS_BODY_BYTES} bytes)",
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > MAX_METRICS_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body too large (max {MAX_METRICS_BODY_BYTES} bytes)",
        )


async def enforce_write_body_limit(request: Request) -> None:
    raw = request.headers.get("content-length")
    if raw:
        try:
            if int(raw) > MAX_WRITE_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large (max {MAX_WRITE_BODY_BYTES} bytes)",
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > MAX_WRITE_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body too large (max {MAX_WRITE_BODY_BYTES} bytes)",
        )


def _insert_model_event(
    conn: sqlite3.Connection,
    *,
    model_id: int,
    event_type: str,
    payload: Optional[Dict[str, Any]],
    actor: str,
) -> None:
    payload_json = json.dumps(payload) if payload is not None else None
    if payload_json is not None and len(payload_json.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Event payload too large (max {MAX_EVENT_PAYLOAD_BYTES} bytes)",
        )
    conn.execute(
        """
        INSERT INTO model_events (model_id, event_type, payload_json, actor)
        VALUES (?, ?, ?, ?)
        """,
        (model_id, event_type, payload_json, actor),
    )


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    init_db()
    yield


app = FastAPI(
    title="Caia Model Registry",
    description="Track and manage model checkpoints",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    return {"service": "model-registry", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


protected = APIRouter(dependencies=[Depends(require_api_key)])


@protected.post("/models", response_model=ModelResponse)
def register_model(
    model: ModelCreate,
    request: Request,
    _body_limit: None = Depends(enforce_write_body_limit),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Register a new model"""
    actor = _actor_from_request(request)

    if (model.local_checkpoint_path or model.local_config_path) and not ALLOW_LOCAL_PATHS:
        raise HTTPException(status_code=400, detail="Local paths are disabled on this server")

    checkpoint_sha256 = _normalize_sha256(model.checkpoint_sha256)
    config_sha256 = _normalize_sha256(model.config_sha256)
    checkpoint_size = model.checkpoint_size_bytes
    config_size = model.config_size_bytes

    config_json = model.config_json
    if config_json is None and model.local_config_path:
        try:
            config_json = _read_json_file(Path(model.local_config_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Failed to read config JSON from local_config_path") from e

    if config_json:
        arch = _extract_arch_fields(config_json)
        model.d_model = model.d_model if model.d_model is not None else arch.get("d_model")
        model.n_layers = model.n_layers if model.n_layers is not None else arch.get("n_layers")
        model.n_heads = model.n_heads if model.n_heads is not None else arch.get("n_heads")
        model.vocab_size = model.vocab_size if model.vocab_size is not None else arch.get("vocab_size")

    if model.local_checkpoint_path:
        sha, size = _sha256_and_size(Path(model.local_checkpoint_path), max_hash_bytes=MAX_HASH_BYTES)
        if size is None:
            raise HTTPException(status_code=400, detail="local_checkpoint_path is not readable")
        if checkpoint_size is not None and checkpoint_size != size:
            raise HTTPException(status_code=400, detail="checkpoint_size_bytes does not match local file size")
        if checkpoint_sha256 is not None and sha is not None and checkpoint_sha256 != sha:
            raise HTTPException(status_code=400, detail="checkpoint_sha256 does not match local file hash")
        checkpoint_size = size if checkpoint_size is None else checkpoint_size
        checkpoint_sha256 = sha if checkpoint_sha256 is None else checkpoint_sha256

    if model.local_config_path:
        sha, size = _sha256_and_size(Path(model.local_config_path), max_hash_bytes=MAX_HASH_BYTES)
        if size is None:
            raise HTTPException(status_code=400, detail="local_config_path is not readable")
        if config_size is not None and config_size != size:
            raise HTTPException(status_code=400, detail="config_size_bytes does not match local file size")
        if config_sha256 is not None and sha is not None and config_sha256 != sha:
            raise HTTPException(status_code=400, detail="config_sha256 does not match local file hash")
        config_size = size if config_size is None else config_size
        config_sha256 = sha if config_sha256 is None else config_sha256

    artifact_uri = (model.artifact_uri or "").strip() or None
    if not artifact_uri:
        if checkpoint_sha256:
            artifact_uri = f"sha256:{checkpoint_sha256}"
        else:
            raise HTTPException(status_code=400, detail="artifact_uri is required unless checkpoint_sha256 is provided")

    checkpoint_path_db = model.local_checkpoint_path or artifact_uri
    config_path_db = model.local_config_path

    created_by = model.created_by or actor
    source_host = model.source_host or (request.client.host if request.client else None)

    try:
        cursor = conn.execute("""
            INSERT INTO models (
                name, version, status,
                artifact_uri, checkpoint_path, config_path,
                checkpoint_sha256, checkpoint_size_bytes,
                config_sha256, config_size_bytes,
                run_id, git_commit, created_by, source_host, frozen,
                d_model, n_layers, n_heads, vocab_size, params,
                training_step, training_loss, dataset,
                description, tags, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.name,
            model.version,
            model.status,
            artifact_uri,
            checkpoint_path_db,
            config_path_db,
            checkpoint_sha256,
            checkpoint_size,
            config_sha256,
            config_size,
            model.run_id,
            model.git_commit,
            created_by,
            source_host,
            1 if model.frozen else 0,
            model.d_model,
            model.n_layers,
            model.n_heads,
            model.vocab_size,
            model.params,
            model.training_step,
            model.training_loss,
            model.dataset,
            model.description,
            json.dumps(model.tags) if model.tags else None,
            json.dumps(model.metrics) if model.metrics else None,
        ))

        row = conn.execute("SELECT * FROM models WHERE id = ?", (cursor.lastrowid,)).fetchone()
        _insert_model_event(
            conn,
            model_id=int(cursor.lastrowid),
            event_type="create",
            payload={"name": model.name, "version": model.version, "status": model.status, "artifact_uri": artifact_uri},
            actor=actor,
        )
        return row_to_model(row)

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Model {model.name}:{model.version} already exists")


@protected.get("/models", response_model=List[ModelResponse])
def list_models(
    conn: sqlite3.Connection = Depends(get_db),
    status: Optional[ModelStatus] = Query(None, description="Filter by status"),
    name: Optional[str] = Query(None, description="Filter by name (exact match)"),
    q: Optional[str] = Query(None, description="Fuzzy search over name/version/description"),
    tag: Optional[str] = Query(None, description="Filter by tag (naive contains)"),
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(0, ge=0),
    sort: Literal["created_at", "updated_at", "training_step"] = Query("created_at"),
    order: Literal["asc", "desc"] = Query("desc"),
):
    """List all models"""
    sort_col = {"created_at": "created_at", "updated_at": "updated_at", "training_step": "training_step"}[sort]
    order_sql = "ASC" if order == "asc" else "DESC"

    query = "SELECT * FROM models WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if name:
        query += " AND name = ?"
        params.append(name)
    if q:
        like = f"%{q}%"
        query += " AND (name LIKE ? OR version LIKE ? OR description LIKE ?)"
        params.extend([like, like, like])
    if tag:
        query += " AND tags LIKE ?"
        params.append(f'%"{tag}"%')

    query += f" ORDER BY {sort_col} {order_sql}"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [row_to_model(row) for row in rows]


@protected.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, conn: sqlite3.Connection = Depends(get_db)):
    """Get a specific model by ID"""
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    return row_to_model(row)


@protected.get("/models/by-name/{name}/{version}", response_model=ModelResponse)
def get_model_by_name(name: str, version: str, conn: sqlite3.Connection = Depends(get_db)):
    """Get a specific model by name and version"""
    row = conn.execute(
        "SELECT * FROM models WHERE name = ? AND version = ?",
        (name, version)
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    return row_to_model(row)


@protected.patch("/models/{model_id}", response_model=ModelResponse)
def update_model(
    model_id: int,
    update: ModelUpdate,
    request: Request,
    _body_limit: None = Depends(enforce_write_body_limit),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Update a model"""
    actor = _actor_from_request(request)

    # Check exists
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    locked = (row["status"] == "production") or bool(row["frozen"])
    if update.frozen is False and bool(row["frozen"]):
        raise HTTPException(status_code=409, detail="Cannot unfreeze a frozen model")

    if (update.local_checkpoint_path or update.local_config_path) and not ALLOW_LOCAL_PATHS:
        raise HTTPException(status_code=400, detail="Local paths are disabled on this server")

    if update.local_checkpoint_path is not None and update.local_checkpoint_path.strip() == "":
        raise HTTPException(status_code=400, detail="local_checkpoint_path cannot be empty")
    if update.local_config_path is not None and update.local_config_path.strip() == "":
        raise HTTPException(status_code=400, detail="local_config_path cannot be empty")

    artifact_fields = {
        "artifact_uri",
        "local_checkpoint_path",
        "local_config_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "config_sha256",
        "config_size_bytes",
        "run_id",
        "git_commit",
        "created_by",
        "source_host",
        "d_model",
        "n_layers",
        "n_heads",
        "vocab_size",
        "params",
    }
    if locked:
        touched = {name for name, value in update.model_dump(exclude_unset=True, by_alias=False).items() if value is not None}
        blocked = sorted(touched.intersection(artifact_fields))
        if blocked:
            raise HTTPException(status_code=409, detail=f"Model is immutable (production/frozen); blocked fields: {blocked}")

    checkpoint_sha256 = _normalize_sha256(update.checkpoint_sha256) if update.checkpoint_sha256 is not None else None
    config_sha256 = _normalize_sha256(update.config_sha256) if update.config_sha256 is not None else None
    checkpoint_size = update.checkpoint_size_bytes
    config_size = update.config_size_bytes
    if checkpoint_size is not None and checkpoint_size < 0:
        raise HTTPException(status_code=400, detail="checkpoint_size_bytes must be >= 0")
    if config_size is not None and config_size < 0:
        raise HTTPException(status_code=400, detail="config_size_bytes must be >= 0")

    if update.local_checkpoint_path:
        sha, size = _sha256_and_size(Path(update.local_checkpoint_path), max_hash_bytes=MAX_HASH_BYTES)
        if size is None:
            raise HTTPException(status_code=400, detail="local_checkpoint_path is not readable")
        if checkpoint_size is not None and checkpoint_size != size:
            raise HTTPException(status_code=400, detail="checkpoint_size_bytes does not match local file size")
        if checkpoint_sha256 is not None and sha is not None and checkpoint_sha256 != sha:
            raise HTTPException(status_code=400, detail="checkpoint_sha256 does not match local file hash")
        checkpoint_size = size
        checkpoint_sha256 = sha if checkpoint_sha256 is None else checkpoint_sha256

    if update.local_config_path:
        sha, size = _sha256_and_size(Path(update.local_config_path), max_hash_bytes=MAX_HASH_BYTES)
        if size is None:
            raise HTTPException(status_code=400, detail="local_config_path is not readable")
        if config_size is not None and config_size != size:
            raise HTTPException(status_code=400, detail="config_size_bytes does not match local file size")
        if config_sha256 is not None and sha is not None and config_sha256 != sha:
            raise HTTPException(status_code=400, detail="config_sha256 does not match local file hash")
        config_size = size
        config_sha256 = sha if config_sha256 is None else config_sha256

    # Build update query
    updates: List[str] = []
    params: List[Any] = []
    updated_fields: List[str] = []

    def add_update(column: str, value: Any, field_name: str) -> None:
        updates.append(f"{column} = ?")
        params.append(value)
        updated_fields.append(field_name)

    if update.status is not None:
        add_update("status", update.status, "status")

    if update.artifact_uri is not None:
        artifact_uri = update.artifact_uri.strip()
        if not artifact_uri:
            raise HTTPException(status_code=400, detail="artifact_uri cannot be empty")
        add_update("artifact_uri", artifact_uri, "artifact_uri")
    if update.local_checkpoint_path is not None:
        add_update("checkpoint_path", update.local_checkpoint_path, "local_checkpoint_path")
    if update.local_config_path is not None:
        add_update("config_path", update.local_config_path, "local_config_path")

    if checkpoint_sha256 is not None or update.local_checkpoint_path is not None:
        if checkpoint_sha256 is not None:
            add_update("checkpoint_sha256", checkpoint_sha256, "checkpoint_sha256")
        elif update.local_checkpoint_path is not None:
            add_update("checkpoint_sha256", None, "checkpoint_sha256")
    if checkpoint_size is not None or update.local_checkpoint_path is not None:
        if checkpoint_size is not None:
            add_update("checkpoint_size_bytes", checkpoint_size, "checkpoint_size_bytes")
        elif update.local_checkpoint_path is not None:
            add_update("checkpoint_size_bytes", None, "checkpoint_size_bytes")

    if config_sha256 is not None or update.local_config_path is not None:
        if config_sha256 is not None:
            add_update("config_sha256", config_sha256, "config_sha256")
        elif update.local_config_path is not None:
            add_update("config_sha256", None, "config_sha256")
    if config_size is not None or update.local_config_path is not None:
        if config_size is not None:
            add_update("config_size_bytes", config_size, "config_size_bytes")
        elif update.local_config_path is not None:
            add_update("config_size_bytes", None, "config_size_bytes")

    if update.run_id is not None:
        add_update("run_id", update.run_id, "run_id")
    if update.git_commit is not None:
        add_update("git_commit", update.git_commit, "git_commit")
    if update.created_by is not None:
        add_update("created_by", update.created_by, "created_by")
    if update.source_host is not None:
        add_update("source_host", update.source_host, "source_host")

    if update.frozen is not None:
        add_update("frozen", 1 if update.frozen else 0, "frozen")

    for field in ["d_model", "n_layers", "n_heads", "vocab_size", "params"]:
        value = getattr(update, field)
        if value is not None:
            add_update(field, value, field)

    if update.training_step is not None:
        add_update("training_step", update.training_step, "training_step")
    if update.training_loss is not None:
        add_update("training_loss", update.training_loss, "training_loss")
    if update.dataset is not None:
        add_update("dataset", update.dataset, "dataset")
    if update.description is not None:
        add_update("description", update.description, "description")
    if update.tags is not None:
        add_update("tags", json.dumps(update.tags), "tags")
    if update.metrics is not None:
        add_update("metrics", json.dumps(update.metrics), "metrics")

    if updates:
        query = f"UPDATE models SET {', '.join(updates)} WHERE id = ?"
        params.append(model_id)
        conn.execute(query, params)

    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if updates:
        _insert_model_event(
            conn,
            model_id=model_id,
            event_type="update",
            payload={"updated_fields": updated_fields},
            actor=actor,
        )
    return row_to_model(row)


@protected.delete("/models/{model_id}")
def delete_model(
    model_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Delete a model"""
    actor = _actor_from_request(request)

    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    _insert_model_event(
        conn,
        model_id=model_id,
        event_type="delete",
        payload={"name": row["name"], "version": row["version"], "status": row["status"], "artifact_uri": row["artifact_uri"]},
        actor=actor,
    )
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))

    return {"status": "deleted", "id": model_id}


# =============================================================================
# Status Management
# =============================================================================

@protected.post("/models/{model_id}/promote", response_model=ModelResponse)
def promote_model(
    model_id: int,
    request: Request,
    to_status: ModelStatus = "production",
    conn: sqlite3.Connection = Depends(get_db),
):
    """Promote a model to a new status"""
    actor = _actor_from_request(request)

    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    # If promoting to production, demote any existing production models of same name
    if to_status == "production":
        if PROMOTION_REQUIRED_SUITE:
            candidate_metrics = _json_loads(row["metrics"])
            if not isinstance(candidate_metrics, dict):
                candidate_metrics = {}
            candidate_suites = candidate_metrics.get("suites")
            if not isinstance(candidate_suites, dict) or PROMOTION_REQUIRED_SUITE not in candidate_suites:
                raise HTTPException(
                    status_code=409,
                    detail=f"Promotion requires suite metrics: {PROMOTION_REQUIRED_SUITE}",
                )

            prod_row = conn.execute(
                "SELECT id, metrics FROM models WHERE name = ? AND status = 'production' AND id != ? LIMIT 1",
                (row["name"], model_id),
            ).fetchone()

            def to_score(suite_entry: Any) -> Optional[float]:
                if not isinstance(suite_entry, dict):
                    return None
                score = suite_entry.get("score")
                if isinstance(score, bool):
                    return None
                if isinstance(score, (int, float)):
                    return float(score)
                return None

            if prod_row:
                prod_metrics = _json_loads(prod_row["metrics"])
                if not isinstance(prod_metrics, dict):
                    prod_metrics = {}
                prod_suites = prod_metrics.get("suites")
                prod_suite = prod_suites.get(PROMOTION_REQUIRED_SUITE) if isinstance(prod_suites, dict) else None
                candidate_suite = candidate_suites.get(PROMOTION_REQUIRED_SUITE)
                prod_score = to_score(prod_suite)
                candidate_score = to_score(candidate_suite)
                if prod_score is not None and candidate_score is not None and candidate_score < prod_score:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Promotion blocked: {PROMOTION_REQUIRED_SUITE}.score "
                            f"({candidate_score}) < current production score ({prod_score})"
                        ),
                    )

        demoted_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM models WHERE name = ? AND status = 'production' AND id != ?",
                (row["name"], model_id),
            ).fetchall()
        ]
        conn.execute("""
            UPDATE models
            SET status = 'staging'
            WHERE name = ? AND status = 'production' AND id != ?
        """, (row["name"], model_id))
        for demoted_id in demoted_ids:
            _insert_model_event(
                conn,
                model_id=int(demoted_id),
                event_type="auto_demote",
                payload={"from": "production", "to": "staging", "reason": f"new production: {model_id}"},
                actor=actor,
            )

    conn.execute("""
        UPDATE models
        SET status = ?
        WHERE id = ?
    """, (to_status, model_id))
    _insert_model_event(
        conn,
        model_id=model_id,
        event_type="promote",
        payload={"from": row["status"], "to": to_status},
        actor=actor,
    )

    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()

    return row_to_model(row)


@protected.get("/models/status/{status}", response_model=List[ModelResponse])
def get_models_by_status(status: ModelStatus, conn: sqlite3.Connection = Depends(get_db)):
    """Get all models with a specific status"""
    rows = conn.execute(
        "SELECT * FROM models WHERE status = ? ORDER BY updated_at DESC",
        (status,)
    ).fetchall()
    return [row_to_model(row) for row in rows]


@protected.get("/production/{name}", response_model=ModelResponse)
def get_production_model(name: str, conn: sqlite3.Connection = Depends(get_db)):
    """Get the production model for a given name"""
    row = conn.execute(
        "SELECT * FROM models WHERE name = ? AND status = 'production' LIMIT 1",
        (name,)
    ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No production model found for '{name}'"
        )

    return row_to_model(row)


# =============================================================================
# Comparison & Analytics
# =============================================================================

@protected.get("/compare")
def compare_models(
    ids: str = Query(..., description="Comma-separated model IDs"),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Compare multiple models"""
    model_ids = [int(x.strip()) for x in ids.split(",")]

    placeholders = ",".join("?" * len(model_ids))
    rows = conn.execute(
        f"SELECT * FROM models WHERE id IN ({placeholders})",
        model_ids
    ).fetchall()

    models = [row_to_model(row) for row in rows]

    return {
        "models": models,
        "comparison": {
            "by_params": sorted(models, key=lambda m: m.params or 0, reverse=True),
            "by_loss": sorted(models, key=lambda m: m.training_loss or float('inf')),
            "by_step": sorted(models, key=lambda m: m.training_step or 0, reverse=True),
        }
    }


@protected.get("/stats")
def get_stats(conn: sqlite3.Connection = Depends(get_db)):
    """Get registry statistics"""
    total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    by_status_rows = conn.execute("SELECT status, COUNT(*) AS n FROM models GROUP BY status").fetchall()
    by_name_rows = conn.execute("SELECT name, COUNT(*) AS n FROM models GROUP BY name").fetchall()

    return {
        "total_models": total,
        "by_status": {r["status"]: r["n"] for r in by_status_rows},
        "by_name": {r["name"]: r["n"] for r in by_name_rows},
    }


@protected.post("/models/{model_id}/metrics", response_model=ModelResponse)
def write_model_metrics(
    model_id: int,
    body: ModelMetricsWriteRequest,
    request: Request,
    _body_limit: None = Depends(enforce_metrics_body_limit),
    if_updated_at: Optional[str] = Query(None, description="Optimistic lock: require model.updated_at to match"),
    if_updated_at_header: Optional[str] = Header(default=None, alias="If-Updated-At"),
    conn: sqlite3.Connection = Depends(get_db),
):
    actor = _actor_from_request(request)
    suite = body.suite.strip()
    if not suite:
        raise HTTPException(status_code=400, detail="suite is required")

    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    expected_updated_at = (if_updated_at_header or if_updated_at or "").strip() or None
    if expected_updated_at and row["updated_at"] != expected_updated_at:
        raise HTTPException(
            status_code=409,
            detail=f"Conflict: model.updated_at has changed (expected {expected_updated_at}, got {row['updated_at']})",
        )

    prior_metrics = _json_loads(row["metrics"])
    if not isinstance(prior_metrics, dict):
        prior_metrics = {}
    prior_metrics_sha256 = _sha256_json(prior_metrics)

    suites = prior_metrics.get("suites")
    if not isinstance(suites, dict):
        suites = {}

    suite_entry: Dict[str, Any] = {}
    existing_suite = suites.get(suite)
    if isinstance(existing_suite, dict):
        suite_entry.update(existing_suite)
    suite_entry.update(body.metrics)

    meta: Dict[str, Any] = {
        "dataset": body.dataset,
        "eval_commit": body.eval_commit,
        "eval_config": body.eval_config,
        "notes": body.notes,
        "recorded_at": _now_utc_iso(),
        "actor": actor,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    existing_meta = suite_entry.get("_meta")
    if isinstance(existing_meta, dict):
        suite_entry["_meta"] = {**existing_meta, **meta}
    else:
        suite_entry["_meta"] = meta

    suites[suite] = suite_entry
    prior_metrics["suites"] = suites

    if expected_updated_at:
        cursor = conn.execute(
            "UPDATE models SET metrics = ? WHERE id = ? AND updated_at = ?",
            (json.dumps(prior_metrics), model_id, expected_updated_at),
        )
        if cursor.rowcount == 0:
            current = conn.execute("SELECT updated_at FROM models WHERE id = ?", (model_id,)).fetchone()
            current_value = current["updated_at"] if current else None
            raise HTTPException(
                status_code=409,
                detail=f"Conflict: model.updated_at has changed (expected {expected_updated_at}, got {current_value})",
            )
    else:
        conn.execute("UPDATE models SET metrics = ? WHERE id = ?", (json.dumps(prior_metrics), model_id))
    event_payload: Dict[str, Any] = {"suite": suite, "metrics": body.metrics, "prior_metrics_sha256": prior_metrics_sha256}
    for key in ["dataset", "eval_commit", "eval_config", "notes"]:
        value = getattr(body, key)
        if value is not None:
            event_payload[key] = value
    _insert_model_event(
        conn,
        model_id=model_id,
        event_type="eval",
        payload=event_payload,
        actor=actor,
    )

    updated = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return row_to_model(updated)


@protected.get("/models/{model_id}/metrics")
def read_model_metrics(
    model_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    suite: Optional[str] = Query(None, description="If set, return only this suite"),
) -> Dict[str, Any]:
    row = conn.execute("SELECT metrics FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    metrics = _json_loads(row["metrics"])
    if not isinstance(metrics, dict):
        metrics = {}
    suites = metrics.get("suites")
    if not isinstance(suites, dict):
        suites = {}

    if suite:
        key = suite.strip()
        if key not in suites:
            raise HTTPException(status_code=404, detail="Suite not found")
        return {key: suites[key]}

    return suites


@protected.get("/models/{model_id}/events", response_model=List[ModelEventResponse])
def list_model_events(
    model_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    rows = conn.execute(
        """
        SELECT * FROM model_events
        WHERE model_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        (model_id, limit, offset),
    ).fetchall()
    return [
        ModelEventResponse(
            id=r["id"],
            model_id=r["model_id"],
            event_type=r["event_type"],
            payload=_json_loads(r["payload_json"]),
            actor=r["actor"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


app.include_router(protected)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
