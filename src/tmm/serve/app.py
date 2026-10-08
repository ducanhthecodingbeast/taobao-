"""Direction 1 — the interactive live demo (FastAPI + optional Redis/Kafka + FAISS + ONNX).

Architecture (see ``docker/docker-compose.yml`` for the full stack)
------------------------------------------------------------------
    POST /events    -> Kafka topic ``tmm.clickstream`` (append-only, at-least-once)
                    -> consumer updates the user's Redis session list (LPUSH + LTRIM 50)
    GET  /recommend -> read session from Redis -> map to rows -> user tower (ONNX)
                    -> FAISS HNSW search -> ranked item ids

Why the request path never touches Kafka
----------------------------------------
The read path is CQRS-shaped: recommendations read only from Redis, so request latency is
independent of broker health. If Redis is unreachable the service degrades to an in-process
store instead of failing, and ``/health`` reports which backend is live. That is what makes
p95 a stable number rather than a function of the broker.

Degradation contract (frozen, ``docs/BACKEND_CONTRACT.md`` §2)
--------------------------------------------------------------
* ``full``     Redis + Kafka reachable: events go to Kafka, the consumer owns session writes.
* ``degraded`` Redis reachable, Kafka not: events are written straight to Redis.
* ``minimal``  Redis unreachable: in-process sessions; ``/readyz`` returns 503 but
               ``/recommend`` keeps answering.

Memory: the demo must fit a 2 vCPU / 4 GB VPS
--------------------------------------------
The shipped embedding table is 35.46 M x 128 x 2 B = **8.45 GiB**. Loading it to serve a
10 000-item catalogue would make the "lightweight" direction a lie, so the service builds a
**reduced table** containing only the catalogue plus every item that appears in the sampled
splits (~870 k rows, ~220 MB fp16) and keeps a ``vocab index -> row`` map. Item ids cross the
API boundary as the original anonymised int64 values and are decoded on the way out.

Pitfalls handled explicitly (all real for this stack)
----------------------------------------------------
* CPU-bound inference inside ``async def`` blocks the event loop -> every data handler is a
  plain ``def`` so FastAPI runs it in its threadpool. Only the metrics middleware is async,
  and it does no CPU work.
* One pooled Redis connection, not a connect-per-request.
* Embeddings stored as raw ``int8``/binary bytes rather than JSON lists.
* Kafka is at-least-once -> session writes are idempotent per ``event_id``.
* The index is **versioned** and echoed in every response, so offline/online embedding drift
  is detectable rather than silent.
* Prometheus collectors live on a per-app :class:`~prometheus_client.CollectorRegistry`, so
  building more than one app in a test process cannot raise Duplicated timeseries.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, Field

from ..config import BUDGET, PATHS
from .events import INT64_MAX, INT64_MIN, EventPublisher, build_publisher
from .obs import (
    CONTENT_TYPE_LATEST,
    Metrics,
    StageTimer,
    build_metrics,
    configure_logging,
    log_event,
    log_warning,
)
from .ratelimit import RateLimiter, build_rate_limiter
from .resilience import STATE_CODES, CircuitBreaker, call_with_timeout
from .session import SessionStore, build_session_store
from .settings import Settings, load_settings

INDEX_VERSION = "v1"
SESSION_MAX = 50
TOWER_SEQ = 200
DEMO_COLS = ("130_1", "130_2", "130_3", "130_4", "130_5")

logger = logging.getLogger("tmm.serve.app")


# --------------------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------------------
class Recommender:
    """Loads a reduced embedding table, item vectors, an ANN index and the user tower once.

    Two load paths:

    * **artifact** (preferred, ``use_demo_artifact=True``): everything comes from the
      precomputed ``artifacts/models/demo/`` bundle -- memory-flat, no memmap over the 4.5 GB
      embedding table, no 35.46 M-entry vocabulary and **no torch**: item vectors are
      precomputed and the user tower is the bundle's ONNX graph. Measured peak RSS ~0.7 GB.
    * **legacy** (fallback): builds the reduced table from the raw feature map via
      ``load_prepared`` + ``load_embeddings_subset`` and runs the torch tower. Correct but
      peaks at ~7 GB, which is why the artifact exists.

    Either way the index and the query come from the **same** space: both from one trained
    checkpoint, or both raw SCL. Mixing them (raw SCL items, trained user tower) is what made
    an earlier version recommend at random level.
    """

    def __init__(self, n_items: int = BUDGET.demo_items, use_demo_artifact: bool = True):
        import faiss

        self.n_items = n_items
        self.artifact: Any = None
        self.row_of: np.ndarray | None = None
        self.d: Any = None
        self.vocab: Any = None
        self.cards: dict[str, int] = {}
        self.tower: Any = None
        self.onnx_session: Any = None

        from .. import demo_artifact

        artifact = None
        if use_demo_artifact and demo_artifact.exists():
            candidate = demo_artifact.load()
            if n_items <= len(candidate.item_ids):
                artifact = candidate

        if artifact is not None:
            self.artifact = artifact
            take = int(n_items)
            self.emb = artifact.emb                                   # fp16 memmap
            self.pad_row = artifact.pad_row
            self.item_ids = artifact.item_ids[:take]                  # raw ids for the API
            self.emb_mb = round(self.emb.nbytes / 2**20, 1)
            self.default_demo = artifact.default_demo
            vecs = artifact.item_vecs[:take]
            if artifact.onnx_path is not None:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = 1      # one user per request; threads only contend
                self.onnx_session = ort.InferenceSession(str(artifact.onnx_path), so,
                                                         providers=["CPUExecutionProvider"])
        else:
            vecs = self._load_legacy(n_items)
        self.dim = int(vecs.shape[1])

        faiss.omp_set_num_threads(1)
        # inner product on L2-normalised vectors = cosine, so higher score = better match
        self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.index_bytes = int(self.index.ntotal * self.dim * 4 + self.index.ntotal * 32 * 2 * 4)

        self.version = INDEX_VERSION
        self.item_pop = self._popularity()

    # -- setup helpers ----------------------------------------------------------------
    def _load_legacy(self, n_items: int) -> np.ndarray:
        import torch

        from ..export_onnx import serving_checkpoint
        from ..index import catalogue_items
        from ..models import TwoTower
        from ..train import load_prepared
        from ..vocab import load_embeddings_subset

        self.d = load_prepared("cpu")
        self.vocab, self.cards = self.d["vocab"], self.d["cards"]
        items = catalogue_items(self.vocab, n_items)                  # VOCABULARY indices
        cats = torch.from_numpy(self._item_categories(items))

        # reduced table: catalogue + anything that can appear in a session
        extra = np.unique(np.concatenate([self.d["train"]["item"], self.d["test"]["item"]]))
        keep = np.unique(np.concatenate([items, extra]))
        self.row_of = np.full(len(self.vocab.ids) + 1, -1, dtype=np.int32)
        self.row_of[keep] = np.arange(len(keep), dtype=np.int32)
        self.pad_row = len(keep)                                      # reserved all-zero row
        self.emb = torch.from_numpy(load_embeddings_subset(self.vocab, keep))
        self.emb_mb = round(self.emb.numel() * self.emb.element_size() / 2**20, 1)
        self.item_ids = self.vocab.decode(items)                      # raw ids for the API
        self.default_demo = np.asarray(
            [int(np.bincount(self.d["train"][c].astype(np.int64)).argmax()) for c in DEMO_COLS],
            dtype=np.int64)
        rows = torch.from_numpy(self.row_of[items].astype(np.int64))

        ck_path = serving_checkpoint()
        with torch.no_grad():
            if ck_path is None:                                       # raw SCL on both sides
                return torch.nn.functional.normalize(self.emb[rows].float(), dim=-1).numpy()
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            self.tower = TwoTower(self.emb, cat_card=self.cards["206"],
                                  demo_cards=[self.cards[c] for c in DEMO_COLS],
                                  dim=ck["dim"], tower_seq_len=TOWER_SEQ,
                                  residual=bool(ck.get("residual", False)))
            self.tower.load_state_dict(ck["state_dict"])
            return self.tower.eval().item_vec(rows, cats).float().numpy()

    def _item_categories(self, items: np.ndarray) -> np.ndarray:
        """Real category index for each catalogue item (never a constant feature)."""
        all_items = np.concatenate([self.d["train"]["item"], self.d["test"]["item"]])
        all_cats = np.concatenate([self.d["train"]["206"], self.d["test"]["206"]])
        uniq, first = np.unique(all_items, return_index=True)
        pos = np.clip(np.searchsorted(uniq, items), 0, len(uniq) - 1)
        hit = uniq[pos] == items
        return np.where(hit, all_cats[first[pos]], self.cards["206"] - 1).astype(np.int64)

    def _popularity(self) -> dict[int, float]:
        """Cold-start fallback keyed by RAW item id."""
        pp = PATHS.stats / "profile.json"
        if pp.exists():
            try:
                top = json.loads(pp.read_text())["popularity"]["top50"]
                return {int(i): float(50 - r) for r, (i, _) in enumerate(top)}
            except Exception:
                return {}
        return {}

    # -- inference --------------------------------------------------------------------
    def rows_for(self, item_ids: list[int]) -> np.ndarray:
        """Raw item ids -> embedding rows; unknown ids become :attr:`pad_row`."""
        if not item_ids:
            return np.empty(0, dtype=np.int64)
        if self.artifact is not None:
            # artifact path: sorted vocab_ids + searchsorted, no vocabulary object
            return self.artifact.rows_for(item_ids)
        vocab, row_of = self.vocab, self.row_of
        if vocab is None or row_of is None:  # pragma: no cover - defensive
            raise RuntimeError("Recommender has neither a demo artifact nor a vocabulary")
        idx = vocab.encode(np.asarray(item_ids, dtype=np.int64))
        rows = np.where(idx == vocab.pad_index, -1, row_of[idx])
        return np.where(rows < 0, self.pad_row, rows).astype(np.int64)

    def user_vector(self, session: list[int]) -> np.ndarray:
        """Masked mean of the session's embeddings, then the user tower.

        Unknown items are dropped (they are the padding row in training too). The mean is fed
        **un-normalised**, exactly as :meth:`tmm.models.TwoTower.user_vec` pools it.
        """
        rows = self.rows_for(session[-TOWER_SEQ:])
        rows = rows[rows != self.pad_row]
        if not len(rows):
            return np.zeros(self.dim, dtype=np.float32)
        if self.tower is not None:
            import torch

            with torch.no_grad():
                return self.tower.user_vec(torch.from_numpy(rows)[None, :],
                                           torch.from_numpy(self.default_demo)[None, :])[0].numpy()
        pooled = np.asarray(self.emb[rows], dtype=np.float32).mean(0)
        if self.onnx_session is not None:
            return self.onnx_session.run(
                ["user_vec"], {"hist_vec": pooled[None, :],
                               "demo": self.default_demo[None, :]})[0][0].astype(np.float32)
        return pooled / (np.linalg.norm(pooled) + 1e-12)

    def sample_item_ids(self, n: int = 200, seed: int = 0) -> list[int]:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(self.item_ids), size=min(n, len(self.item_ids)), replace=False)
        return [int(x) for x in self.item_ids[take]]

    def search(self, user_vec: np.ndarray, session: list[int], k: int = 10,
               exclude_seen: bool = True) -> list[dict]:
        """ANN search + cold-start fallback. Split out of :meth:`recommend` so the caller can
        time ``tower`` and ``search`` as separate stages."""
        if not np.any(user_vec):
            order = sorted(self.item_pop.items(), key=lambda kv: -kv[1])[:k]
            if order:
                return [{"item_id": int(i), "score": 0.0, "cold_start": True} for i, _ in order]
            return [{"item_id": int(i), "score": 0.0, "cold_start": True}
                    for i in self.sample_item_ids(k)]
        D, I = self.index.search(np.ascontiguousarray(user_vec[None, :].astype(np.float32)),
                                 min(k * 3 + 10, self.n_items))
        seen = set(session) if exclude_seen else set()
        out = []
        for score, j in zip(D[0], I[0]):
            if j < 0:
                continue
            iid = int(self.item_ids[j])
            if iid in seen:
                continue
            out.append({"item_id": iid, "score": round(float(score), 5), "cold_start": False})
            if len(out) >= k:
                break
        return out

    def recommend(self, session: list[int], k: int = 10, exclude_seen: bool = True) -> list[dict]:
        return self.search(self.user_vector(session), session, k=k, exclude_seen=exclude_seen)


# --------------------------------------------------------------------------------------
# Request/response models
#
# These MUST live at module level. With ``from __future__ import annotations`` every
# annotation is a string, and FastAPI resolves those strings against the module globals. A
# model defined *inside* ``create_app`` is not in the globals, so FastAPI silently degrades
# the parameter to a required **query** field and every POST returns 422 with
# ``loc: ["query", "ev"]`` -- which is exactly what happened here before the move.
# --------------------------------------------------------------------------------------
class EventIn(BaseModel):
    user_id: int = Field(ge=INT64_MIN, le=INT64_MAX)
    item_id: int = Field(ge=INT64_MIN, le=INT64_MAX)
    event_type: str = "click"
    event_id: str | None = None
    ts: str | None = None


#: Backwards-compatible alias for the pre-B1 model name.
Event = EventIn


class Rec(BaseModel):
    item_id: int
    score: float
    cold_start: bool = False


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    if client is not None and client.host:
        return client.host
    return "unknown"


def create_app(
    redis_url: str | None = None,
    n_items: int = BUDGET.demo_items,
    *,
    settings: Settings | None = None,
    store: SessionStore | None = None,
    publisher: EventPublisher | None = None,
    recommender: Any | None = None,
    rate_limiter: RateLimiter | None = None,
    registry: CollectorRegistry | None = None,
    redis_client: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Every heavy/optional component can be injected, which is what lets the unit tests run
    without Redis, Kafka or the 212 MB embedding table.
    """
    configure_logging(os.environ.get("TMM_LOG_LEVEL", "INFO"))

    base = settings or load_settings()
    cfg = dataclasses.replace(base, redis_url=redis_url) if redis_url is not None else base

    metrics: Metrics = build_metrics(registry)

    def _on_circuit(backend: str):
        def observe(state: str) -> None:
            metrics.circuit_state.labels(backend=backend).set(STATE_CODES.get(state, 0))
        return observe

    redis_breaker = CircuitBreaker(
        cfg.circuit_fail_threshold, cfg.circuit_reset_s,
        name="redis", on_state_change=_on_circuit("redis"))
    kafka_breaker = CircuitBreaker(
        cfg.circuit_fail_threshold, cfg.circuit_reset_s,
        name="kafka", on_state_change=_on_circuit("kafka"))
    metrics.circuit_state.labels(backend="redis").set(0)
    metrics.circuit_state.labels(backend="kafka").set(0)

    session_store: SessionStore = (
        store if store is not None
        else build_session_store(cfg, breaker=redis_breaker, client=redis_client))
    event_publisher: EventPublisher = (
        publisher if publisher is not None else build_publisher(cfg, metrics))
    limiter: RateLimiter = (
        rate_limiter if rate_limiter is not None else build_rate_limiter(cfg))
    rec = recommender if recommender is not None else Recommender(n_items=n_items)

    started = time.time()
    state = {"events": 0, "recommends": 0, "errors": 0, "started": started}

    if not cfg.auth_required:
        log_warning(
            logger, "auth_disabled",
            detail="TMM_API_KEYS is empty: every request is accepted. Set TMM_API_KEYS "
                   "(and optionally TMM_AUTH_REQUIRED=1) before exposing this service.",
            rate_limit_rps=cfg.rate_limit_rps)

    # -- degradation / health ----------------------------------------------------------
    def _force_probe() -> None:
        probe = getattr(session_store, "probe", None)
        if callable(probe):
            try:
                probe(force=True)
            except TypeError:  # a store whose probe() takes no arguments
                probe()

    def _kafka_ok() -> bool:
        # A publisher without a broker (kind="memory") can never make the deployment `full`.
        # Publishers that do not declare a kind are treated as Kafka, so a test double only
        # has to implement healthy()/publish() to exercise the full-mode path.
        if getattr(event_publisher, "kind", "kafka") != "kafka":
            return False
        try:
            ok = bool(event_publisher.healthy())
        except Exception:  # noqa: BLE001 - a sick publisher must not break /health
            ok = False
        if ok:
            kafka_breaker.record_success()
        return ok

    def health_snapshot(force_health: bool = False) -> dict[str, Any]:
        if force_health:
            _force_probe()
        redis_ok = session_store.backend == "redis"
        kafka_ok = _kafka_ok()
        if redis_ok and kafka_ok:
            mode = "full"
        elif redis_ok:
            mode = "degraded"
        else:
            mode = "minimal"
        return {"mode": mode, "redis": redis_ok, "kafka": kafka_ok}

    # -- auth + rate limiting ----------------------------------------------------------
    def guard(request: Request,
              x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
        """Authenticate, then charge one token. Returns the bucket key used.

        Auth runs first so a wrong key is always reported as 401 rather than being masked by
        the limiter. With ``auth_required=False`` the limiter still applies, keyed by API key
        when present and by client IP otherwise (a missing IP shares one anonymous bucket,
        so it cannot bypass the limiter).
        """
        if cfg.auth_required:
            if not x_api_key or x_api_key not in cfg.api_keys:
                raise HTTPException(
                    status_code=401, detail="invalid or missing API key",
                    headers={"WWW-Authenticate": "X-API-Key"})
            key, scope = x_api_key, "api_key"
        else:
            key = x_api_key or _client_ip(request)
            scope = "api_key" if x_api_key else "ip"
        allowed, retry_after = limiter.allow(key)
        if not allowed:
            metrics.rate_limited.labels(scope=scope).inc()
            raise HTTPException(
                status_code=429, detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, int(math.ceil(retry_after))))})
        return key

    # -- event path --------------------------------------------------------------------
    def _publish_or_raise(event: dict[str, Any]) -> bool:
        if not event_publisher.publish(event):
            raise RuntimeError("publish failed")
        return True

    def _try_publish(event: dict[str, Any]) -> bool:
        if not event_publisher.healthy():
            return False
        try:
            return bool(kafka_breaker.call(_publish_or_raise, event))
        except Exception:  # noqa: BLE001 - CircuitOpen or broker error -> direct write
            return False

    def _session_size(user_id: int) -> int:
        try:
            return len(session_store.get(user_id))
        except Exception:  # noqa: BLE001 - the size is informational only
            return 0

    def _is_seen(event_id: str) -> bool:
        """Idempotency probe. Never fatal: a broken store must not 500 an ingest request."""
        try:
            return bool(session_store.is_seen(event_id))
        except Exception:  # noqa: BLE001
            return False

    def _handle_event(ev: EventIn) -> dict[str, Any]:
        snap = health_snapshot()
        mode = snap["mode"]
        ts_bucket = int(time.time() // 60)
        event_id = ev.event_id or f"{ev.user_id}:{ev.item_id}:{ev.event_type}:{ts_bucket}"
        payload = {
            "user_id": ev.user_id,
            "item_id": ev.item_id,
            "event_type": ev.event_type,
            "event_id": event_id,
            "ts": ev.ts or datetime.now(tz=UTC).isoformat(),
        }
        # In `full` mode the consumer owns the session write, so a successful publish must
        # NOT also append here (contract §4 / acceptance criterion 7). The store is still
        # *read* to answer `duplicate`/`session_size` truthfully; any other mode, or a failed
        # publish, falls back to a direct append so no accepted click is lost.
        if mode == "full":
            if _is_seen(event_id):
                state["events"] += 1
                metrics.events_ingested.labels(result="duplicate").inc()
                return {"ok": True, "event_id": event_id, "accepted": False,
                        "session_size": _session_size(ev.user_id), "mode": mode,
                        "duplicate": True}
            if _try_publish(payload):
                accepted, size = True, _session_size(ev.user_id)
            else:
                size, accepted = _append_event(ev, event_id)
        else:
            _try_publish(payload)
            size, accepted = _append_event(ev, event_id)
        state["events"] += 1
        duplicate = not accepted
        metrics.events_ingested.labels(result="duplicate" if duplicate else "accepted").inc()
        return {"ok": True, "event_id": event_id, "accepted": accepted,
                "session_size": size, "mode": mode, "duplicate": duplicate}

    def _append_event(ev: EventIn, event_id: str) -> tuple[int, bool]:
        try:
            return session_store.append(ev.user_id, ev.item_id, event_id=event_id)
        except Exception as exc:  # noqa: BLE001
            state["errors"] += 1
            metrics.events_ingested.labels(result="rejected").inc()
            log_warning(logger, "event_store_failed", error=type(exc).__name__,
                        user_id=ev.user_id)
            raise HTTPException(status_code=503, detail="session store unavailable") from exc

    # -- lifespan ----------------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log_event(logger, "app_started", mode=health_snapshot(force_health=True)["mode"],
                  session_backend=session_store.backend, publisher=event_publisher.kind)
        try:
            yield
        finally:
            log_event(logger, "app_stopping")
            for closable in (event_publisher, session_store):
                close = getattr(closable, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass

    app = FastAPI(title="TAOBAO-MM next-item demo", version=INDEX_VERSION, lifespan=lifespan)

    # -- observability middleware ------------------------------------------------------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - t0
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or request.url.path
            metrics.http_requests.labels(endpoint=endpoint, status=str(status)).inc()
            metrics.http_duration.labels(endpoint=endpoint).observe(elapsed)

    # -- HTTP surface ------------------------------------------------------------------
    @app.get("/livez")
    def livez() -> dict[str, Any]:
        """Liveness: always 200 while the process is serving, services or not."""
        return {"status": "alive", "uptime_s": round(time.time() - started, 1)}

    @app.get("/readyz")
    def readyz(response: Response) -> dict[str, Any]:
        snap = health_snapshot(force_health=True)
        ready = snap["mode"] != "minimal"
        metrics.ready.set(1 if ready else 0)
        if not ready:
            response.status_code = 503
        return {"ready": ready, **snap}

    @app.get("/health")
    def health() -> dict[str, Any]:
        snap = health_snapshot(force_health=True)
        metrics.ready.set(1 if snap["mode"] != "minimal" else 0)
        return {
            "status": "ok" if snap["mode"] != "minimal" else "degraded",
            "mode": snap["mode"],
            "session_backend": session_store.backend,
            "index_version": getattr(rec, "version", INDEX_VERSION),
            "catalogue": getattr(rec, "n_items", 0),
            "index_mb": round(getattr(rec, "index_bytes", 0) / 2**20, 2),
            "embedding_table_mb": getattr(rec, "emb_mb", 0.0),
            "user_tower": ("onnx" if getattr(rec, "onnx_session", None) is not None
                           else ("torch" if getattr(rec, "tower", None) is not None
                                 else "identity")),
            "uptime_s": round(time.time() - state["started"], 1),
            "kafka": snap["kafka"],
            "redis": snap["redis"],
            "kafka_status": "ok" if snap["kafka"] else "unavailable",
            "redis_status": "ok" if snap["redis"] else "unavailable",
            "circuits": {"redis": redis_breaker.state, "kafka": kafka_breaker.state},
            "events": state["events"],
            "recommends": state["recommends"],
            "errors": state["errors"],
        }

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        """Prometheus text exposition (contract §3.6)."""
        metrics.ready.set(1 if health_snapshot(force_health=True)["mode"] != "minimal" else 0)
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/items/sample")
    def items_sample(n: int = Query(200, ge=1, le=2000),
                     _key: str = Depends(guard)) -> dict[str, Any]:
        """Real anonymised item ids, so a load test can post valid events."""
        return {"item_ids": rec.sample_item_ids(n)}

    @app.post("/events", status_code=202)
    def post_events(ev: EventIn, _key: str = Depends(guard)) -> dict[str, Any]:
        """Canonical ingest endpoint."""
        return _handle_event(ev)

    @app.post("/event", status_code=200)
    def post_event(ev: EventIn, _key: str = Depends(guard)) -> dict[str, Any]:
        """Legacy alias -- same handler, historical 200 status kept for compatibility."""
        return _handle_event(ev)

    def _read_session(user_id: int) -> list[int]:
        with StageTimer("session_read", metrics):
            try:
                return list(call_with_timeout(session_store.get, cfg.stage_timeout_s, user_id))
            except TimeoutError:
                redis_breaker.record_failure()
                log_warning(logger, "session_read_timeout", user_id=user_id,
                            timeout_s=cfg.stage_timeout_s)
                return []
            except Exception:  # noqa: BLE001 - degrade to an empty session, never 500
                return []

    #: The real model splits tower/search so both stages can be timed; lightweight stand-ins
    #: (tests) only need to expose ``recommend``.
    staged = callable(getattr(rec, "user_vector", None)) and callable(getattr(rec, "search", None))

    def _recommend(uid: int, kk: int) -> list[dict[str, Any]]:
        session = _read_session(uid)
        if staged:
            with StageTimer("tower", metrics):
                user_vec = rec.user_vector(session)
            with StageTimer("search", metrics):
                return rec.search(user_vec, session, k=kk)
        with StageTimer("tower", metrics), StageTimer("search", metrics):
            return rec.recommend(session, k=kk)

    @app.get("/recommend", response_model=list[Rec])
    def recommend(user_id: int = Query(..., ge=INT64_MIN, le=INT64_MAX), k: int = Query(10, ge=1, le=50),
                  _key: str = Depends(guard)) -> list[dict[str, Any]]:
        """Plain ``def`` on purpose: FastAPI runs it in the threadpool so the CPU-bound
        ONNX/FAISS work cannot block the event loop."""
        try:
            with StageTimer("total", metrics):
                with StageTimer("decode", metrics):
                    uid = int(user_id)
                    kk = int(k)
                out = _recommend(uid, kk)
            state["recommends"] += 1
            return out
        except Exception:
            state["errors"] += 1
            raise

    app.state.metrics = metrics
    app.state.settings = cfg
    app.state.session_store = session_store
    app.state.publisher = event_publisher
    app.state.rate_limiter = limiter
    app.state.recommender = rec
    app.state.redis_breaker = redis_breaker
    app.state.kafka_breaker = kafka_breaker
    return app


def _fallback_app(exc: Exception) -> FastAPI:
    """Importable app for environments without the model artifacts (docs / unit imports)."""
    metrics = build_metrics(CollectorRegistry())
    app = FastAPI(title="TAOBAO-MM next-item demo", version=INDEX_VERSION)

    @app.get("/livez")
    def livez() -> dict[str, Any]:
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz(response: Response) -> dict[str, Any]:
        response.status_code = 503
        return {"ready": False, "mode": "minimal"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "degraded", "mode": "minimal", "error": str(exc)}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    return app


# uvicorn addresses ``tmm.serve.app:app``. The module-level app is built lazily on first
# attribute access (PEP 562) so importing this module stays cheap -- tests that only need
# ``create_app`` never pay for the 212 MB embedding table. If the artifacts are missing we
# still return an importable app so the module can be inspected / documented without a
# 6 GB data download.
_APP: FastAPI | None = None


def get_app() -> FastAPI:
    global _APP
    if _APP is None:
        try:
            _APP = create_app()
        except Exception as exc:  # pragma: no cover - exercised without artifacts
            logger.warning("serving app built in degraded fallback mode: %s", exc)
            _APP = _fallback_app(exc)
    return _APP


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

