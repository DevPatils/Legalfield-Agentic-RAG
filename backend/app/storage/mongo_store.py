"""MongoDB persistence for sessions and query traces.

Mongo holds *only* session and trace data (Architecture.md §14) -- never chunk text,
vectors, or graph edges. Those live in Qdrant.

Traces are written incrementally as each node completes rather than once at the end
(§8), which gives crash recovery and lets a partial run be inspected while it is still
going. It also means the demo can replay a stored trace instead of re-running the
pipeline and re-spending API calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from ..config import Settings, get_settings


def _serialize(value: Any) -> Any:
    """Make node output BSON-safe. RetrievedChunk objects become their trace dicts."""
    if hasattr(value, "to_trace"):
        return value.to_trace()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


class MongoStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(self.settings.mongo_uri)
        self.db = self.client[self.settings.mongo_db]

    async def ensure_indexes(self) -> None:
        await self.db.query_traces.create_index("session_id")
        await self.db.query_traces.create_index("created_at")

    # --- sessions ---

    async def ensure_session(self, session_id: str) -> None:
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$setOnInsert": {"created_at": datetime.now(UTC), "queries": []},
            },
            upsert=True,
        )

    async def attach_query(self, session_id: str, query_id: str) -> None:
        await self.db.sessions.update_one(
            {"_id": session_id}, {"$addToSet": {"queries": query_id}}
        )

    async def session_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = (
            self.db.query_traces.find(
                {"session_id": session_id},
                {"raw_query": 1, "created_at": 1, "final_answer": 1, "low_confidence": 1},
            )
            .sort("created_at", -1)
            .limit(limit)
        )
        return [{**doc, "query_id": doc.pop("_id")} async for doc in cursor]

    # --- traces ---

    async def start_trace(
        self, query_id: str, session_id: str, raw_query: str, doc_id: str | None = None
    ) -> None:
        await self.db.query_traces.update_one(
            {"_id": query_id},
            {
                "$set": {
                    "session_id": session_id,
                    "raw_query": raw_query,
                    "doc_id": doc_id,
                    "created_at": datetime.now(UTC),
                    "status": "running",
                }
            },
            upsert=True,
        )

    async def update_trace(self, query_id: str, node: str, partial: dict[str, Any]) -> None:
        """Merge one node's output into the stored trace.

        Only whitelisted fields are persisted -- the live state also carries
        RetrievedChunk objects and other run-time values that have no place in a trace.
        """
        keep = {
            "query_type",
            "needs_retrieval",
            "sub_queries",
            "planner_reasoning",
            "iteration",
            "iterations",
            "sufficient",
            "missing_info",
            "sections_needed",
            "terms_needed",
            "refine_strategy",
            "summary",
            "final_answer",
            "citations",
            "unverified_citations",
            "faithfulness",
            "low_confidence",
            "confidence",
            "latency_ms",
            "token_usage",
            "error",
        }
        update = {k: _serialize(v) for k, v in partial.items() if k in keep}
        if not update:
            return
        update["last_node"] = node
        await self.db.query_traces.update_one({"_id": query_id}, {"$set": update}, upsert=True)

    async def finish_trace(self, query_id: str) -> None:
        await self.db.query_traces.update_one(
            {"_id": query_id},
            {"$set": {"status": "complete", "completed_at": datetime.now(UTC)}},
        )

    async def fail_trace(self, query_id: str, error: str) -> None:
        await self.db.query_traces.update_one(
            {"_id": query_id}, {"$set": {"status": "failed", "error": error}}
        )

    async def get_trace(self, query_id: str) -> dict[str, Any] | None:
        doc = await self.db.query_traces.find_one({"_id": query_id})
        if doc:
            doc["query_id"] = doc.pop("_id")
        return doc

    async def close(self) -> None:
        self.client.close()
