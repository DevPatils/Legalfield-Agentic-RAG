"""Structured output schemas for every LLM call in the graph.

Each node returns a validated Pydantic model rather than free text. That is what makes
the conditional edges in Architecture.md §7 reliable: routing on ``sufficient: bool``
is only safe if the field is guaranteed to exist and be a boolean.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QueryType = Literal[
    "definitional", "cross_referential", "comparative", "overview", "out_of_scope"
]


class PlannerOutput(BaseModel):
    """Node 1: classify, gate retrieval, decompose."""

    query_type: QueryType = Field(description="The kind of question being asked.")
    needs_retrieval: bool = Field(
        description="False for greetings, meta-questions, or anything answerable "
        "without consulting the contracts."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Compound questions split into independently answerable parts. "
        "A simple question yields exactly one sub-query.",
    )
    doc_id: str | None = Field(
        default=None,
        description="Set when the question names one indexed document, e.g. 'doc_001'. "
        "Restricts retrieval to that agreement. Null when the question spans documents.",
    )
    reasoning: str = Field(default="", description="One sentence on the classification.")


class SufficiencyOutput(BaseModel):
    """Node 3: is the retrieved context actually enough to answer?"""

    sufficient: bool = Field(
        description="True only if the retrieved clauses fully support a complete answer."
    )
    missing_info: str = Field(
        default="",
        description="What is absent. Empty when sufficient.",
    )
    referenced_sections_needed: list[str] = Field(
        default_factory=list,
        description="Section numbers the retrieved text points at but which are not "
        "present, e.g. ['4.2', '1.3']. Drives graph traversal.",
    )
    defined_terms_needed: list[str] = Field(
        default_factory=list,
        description="Capitalized defined terms used but not defined in the context.",
    )


class RewrittenQuery(BaseModel):
    """Node 4a: query rewriting refinement strategy."""

    rewritten_query: str = Field(description="Expanded query with missing terms/synonyms.")
    rationale: str = Field(default="", description="Why this rewrite should retrieve better.")


class GeneratedAnswer(BaseModel):
    """Node 5: the answer, with citations required per claim."""

    answer: str = Field(
        description="The answer. Every factual claim carries a citation tag in the "
        "form [doc_id §section_id]."
    )
    citations_used: list[str] = Field(
        default_factory=list,
        description="Every section_id cited in the answer.",
    )
    confidence: Literal["high", "medium", "low"] = "medium"


class ClaimCheck(BaseModel):
    """One atomic claim checked against only its own cited chunk."""

    claim: str
    cited_section: str = ""
    supported: bool
    issue: str = Field(default="", description="Why unsupported. Empty when supported.")


class FaithfulnessOutput(BaseModel):
    """Node 6: per-citation entailment."""

    claims: list[ClaimCheck] = Field(default_factory=list)
