"""Hierarchical knowledge trees, long-form chapter planning, and long-to-short repurposing."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

AudienceLevel = Literal["school", "beginner", "undergraduate", "advanced", "general"]
ClaimKind = Literal["fact", "identity", "approximation", "analogy", "hypothesis"]


class KnowledgeClaim(BaseModel):
    statement: str
    kind: ClaimKind
    latex: Optional[str] = None
    verified: bool = False


class KnowledgeNode(BaseModel):
    node_id: str
    title: str
    prerequisites: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    claims: list[KnowledgeClaim] = Field(default_factory=list)
    short_hook_idea: Optional[str] = None


class KnowledgeTree(BaseModel):
    topic: str
    audience: AudienceLevel = "general"
    nodes: list[KnowledgeNode] = Field(default_factory=list)

    def derive_short_topics(self) -> list[dict[str, str]]:
        """Extract independent short-form video angles from validated knowledge nodes."""
        shorts: list[dict[str, str]] = []
        for node in self.nodes:
            if node.short_hook_idea:
                shorts.append({"node_id": node.node_id, "title": node.title, "hook": node.short_hook_idea})
        return shorts


class ContinuityState(BaseModel):
    defined_variables: dict[str, str] = Field(default_factory=dict)
    introduced_terms: list[str] = Field(default_factory=list)
    color_map: dict[str, str] = Field(default_factory=dict)


class ChapterSpec(BaseModel):
    chapter_id: str
    title: str
    node_ids: list[str]
    target_duration_s: float = 90.0
    summary: str = ""


class LongFormPlan(BaseModel):
    topic: str
    audience: AudienceLevel = "general"
    knowledge_tree: KnowledgeTree
    chapters: list[ChapterSpec] = Field(default_factory=list)
    continuity: ContinuityState = Field(default_factory=ContinuityState)
