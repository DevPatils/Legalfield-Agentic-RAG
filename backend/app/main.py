"""FastAPI app: the six routes of Architecture.md §8.

The agent graph is synchronous and CPU/IO-bound in a blocking way (the SDKs are sync),
so ``stream_query`` runs in a worker thread and its output is pumped to the SSE
response through a queue. Running it inline would block the event loop and stall every
other request for the duration of a query.

The API key never reaches the browser -- every model call happens here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .agent.deps import AgentDeps, build_deps
from .agent.graph import stream_query
from .config import get_settings
from .storage.mongo_store import MongoStore

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Built once: the BM25 index scrolls every payload out of Qdrant, which is far
    # too expensive to repeat per request.
    state["deps"] = await asyncio.to_thread(build_deps, settings)
    state["mongo"] = MongoStore(settings)
    await state["mongo"].ensure_indexes()
    yield
    await state["mongo"].close()


app = FastAPI(title="Agentic RAG — Legal", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def deps() -> AgentDeps:
    if "deps" not in state:
        raise HTTPException(503, "agent not ready")
    return state["deps"]


def mongo() -> MongoStore:
    return state["mongo"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = ""
    doc_id: str | None = None


# --------------------------------------------------------------------------- query


async def run_agent(
    query: str, query_id: str, session_id: str, doc_id: str | None
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Run the blocking graph in a thread, yielding node updates as they arrive."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    SENTINEL = object()

    def worker() -> None:
        try:
            for node, partial in stream_query(
                deps(), query, query_id=query_id, session_id=session_id, doc_id=doc_id
            ):
                loop.call_soon_threadsafe(queue.put_nowait, (node, partial))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"error": str(exc)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            yield item
    finally:
        await task


def public_state(partial: dict[str, Any]) -> dict[str, Any]:
    """Strip runtime objects the browser can't use; keep the trace fields."""
    out: dict[str, Any] = {}
    for key, value in partial.items():
        if key == "context":
            out["context_size"] = len(value)
        elif hasattr(value, "to_trace"):
            out[key] = value.to_trace()
        elif isinstance(value, list):
            out[key] = [v.to_trace() if hasattr(v, "to_trace") else v for v in value]
        else:
            out[key] = value
    return out


@app.post("/api/query")
async def post_query(body: QueryRequest) -> dict[str, Any]:
    """Run the pipeline to completion and return the answer plus the full trace."""
    query_id = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())
    store = mongo()
    await store.ensure_session(session_id)
    await store.start_trace(query_id, session_id, body.query, body.doc_id)
    await store.attach_query(session_id, query_id)

    final: dict[str, Any] = {}
    async for node, partial in run_agent(body.query, query_id, session_id, body.doc_id):
        final.update(public_state(partial))
        await store.update_trace(query_id, node, partial)
    await store.finish_trace(query_id)

    return {"query_id": query_id, "session_id": session_id, **final}


@app.get("/api/query/stream")
async def stream(
    query: str = Query(min_length=1, max_length=2000),
    session_id: str = "",
    doc_id: str | None = None,
) -> EventSourceResponse:
    """SSE: emit each node's output as it completes (Architecture.md §8)."""
    query_id = str(uuid.uuid4())
    sid = session_id or str(uuid.uuid4())

    async def events() -> AsyncIterator[dict[str, str]]:
        store = mongo()
        await store.ensure_session(sid)
        await store.start_trace(query_id, sid, query, doc_id)
        await store.attach_query(sid, query_id)
        yield {
            "event": "start",
            "data": json.dumps({"query_id": query_id, "session_id": sid, "query": query}),
        }
        try:
            async for node, partial in run_agent(query, query_id, sid, doc_id):
                # Persist before emitting, so a trace fetched right after a dropped
                # connection is never behind what the client already saw.
                await store.update_trace(query_id, node, partial)
                yield {
                    "event": "node",
                    "data": json.dumps({"node": node, "state": public_state(partial)}),
                }
            await store.finish_trace(query_id)
            yield {"event": "done", "data": json.dumps({"query_id": query_id})}
        except Exception as exc:  # noqa: BLE001
            await store.fail_trace(query_id, str(exc))
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(events())


# ----------------------------------------------------------------------- documents


@app.get("/api/documents")
async def list_documents() -> list[dict[str, Any]]:
    return await asyncio.to_thread(deps().store.list_documents)


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str, section_id: str | None = None) -> dict[str, Any]:
    """Fetch a whole document's sections, or one clause for citation click-through."""

    def fetch() -> dict[str, Any]:
        payloads = [p for p in deps().store.all_chunks() if p["doc_id"] == doc_id]
        if not payloads:
            raise HTTPException(404, f"unknown document {doc_id}")
        if section_id:
            match = [p for p in payloads if p["section_id"] == section_id]
            if not match:
                raise HTTPException(404, f"{doc_id} has no section {section_id}")
            return {"doc_id": doc_id, "sections": match}
        payloads.sort(key=lambda p: p.get("section_id", ""))
        return {"doc_id": doc_id, "num_sections": len(payloads), "sections": payloads}

    return await asyncio.to_thread(fetch)


# ------------------------------------------------------------------ sessions/traces


@app.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str) -> list[dict[str, Any]]:
    return await mongo().session_history(session_id)


@app.get("/api/query/{query_id}/trace")
async def get_trace(query_id: str) -> dict[str, Any]:
    """Replay a stored trace without re-running the pipeline or spending API calls."""
    trace = await mongo().get_trace(query_id)
    if not trace:
        raise HTTPException(404, f"unknown query {query_id}")
    return trace


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if "deps" in state else "starting",
        "chunks": len(deps().bm25.chunk_ids) if "deps" in state else 0,
        "model": get_settings().llm_model,
    }
