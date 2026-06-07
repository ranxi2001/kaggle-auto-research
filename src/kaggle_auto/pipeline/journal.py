"""Solution journal: tree-structured experiment tracking (inspired by AIDE).

Each experiment is a Node in a tree. The search policy selects which node
to improve/debug next, guided by metric feedback.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Node:
    """A single experiment node in the solution tree."""

    plan: str
    code_path: str = ""
    step: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)

    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    metric_value: float | None = None
    metric_name: str = ""
    is_buggy: bool = False
    error_message: str = ""
    analysis: str = ""

    stage: Literal["draft", "debug", "improve"] = "draft"
    features_version: str = ""
    model_version: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plan": self.plan,
            "code_path": self.code_path,
            "step": self.step,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "is_buggy": self.is_buggy,
            "error_message": self.error_message,
            "analysis": self.analysis,
            "stage": self.stage,
            "features_version": self.features_version,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Journal:
    """Tree-structured experiment journal.

    Tracks all experiments as a tree where:
    - Root nodes are initial "draft" approaches
    - Children are "improve" or "debug" iterations
    - The search policy picks the most promising node to expand
    """

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.journal_file = workspace_path / ".state" / "journal.json"
        self.nodes: dict[str, Node] = {}
        self._load()

    def add_node(
        self,
        plan: str,
        parent_id: str | None = None,
        stage: Literal["draft", "debug", "improve"] = "draft",
    ) -> Node:
        """Add a new experiment node."""
        node = Node(
            plan=plan,
            parent_id=parent_id,
            stage=stage,
            step=len(self.nodes),
        )

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node.id)

        self.nodes[node.id] = node
        self._save()
        return node

    def update_node(
        self,
        node_id: str,
        metric_value: float | None = None,
        is_buggy: bool = False,
        error_message: str = "",
        analysis: str = "",
        code_path: str = "",
        features_version: str = "",
        model_version: str = "",
    ) -> None:
        """Update node with execution results."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if metric_value is not None:
            node.metric_value = metric_value
        node.is_buggy = is_buggy
        node.error_message = error_message
        node.analysis = analysis
        if code_path:
            node.code_path = code_path
        if features_version:
            node.features_version = features_version
        if model_version:
            node.model_version = model_version
        self._save()

    def get_best_node(self, minimize: bool = True) -> Node | None:
        """Get the node with the best metric value."""
        valid = [n for n in self.nodes.values()
                 if n.metric_value is not None and not n.is_buggy]
        if not valid:
            return None
        if minimize:
            return min(valid, key=lambda n: n.metric_value)
        return max(valid, key=lambda n: n.metric_value)

    def get_leaf_nodes(self) -> list[Node]:
        """Get all leaf nodes (no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_buggy_nodes(self) -> list[Node]:
        """Get nodes that errored out."""
        return [n for n in self.nodes.values() if n.is_buggy]

    def get_draft_nodes(self) -> list[Node]:
        """Get initial draft nodes."""
        return [n for n in self.nodes.values() if n.stage == "draft"]

    def search_policy(self, num_drafts: int = 3, debug_prob: float = 0.3) -> Node | None:
        """Select next node to work on (AIDE-inspired tree search).

        Returns None to signal "draft a new approach" or a Node to improve/debug.
        """
        import random

        drafts = self.get_draft_nodes()
        if len(drafts) < num_drafts:
            return None

        if random.random() < debug_prob:
            buggy_leaves = [n for n in self.get_buggy_nodes()
                           if not n.children_ids and len(n.children_ids) < 3]
            if buggy_leaves:
                return random.choice(buggy_leaves)

        valid = [n for n in self.nodes.values()
                 if n.metric_value is not None and not n.is_buggy]
        if not valid:
            return None

        return self.get_best_node()

    def get_tree_summary(self) -> str:
        """Get a text summary of the experiment tree."""
        lines = ["# Experiment Tree", ""]
        roots = [n for n in self.nodes.values() if n.parent_id is None]

        for root in roots:
            self._format_subtree(root, lines, indent=0)

        return "\n".join(lines)

    def _format_subtree(self, node: Node, lines: list[str], indent: int) -> None:
        prefix = "  " * indent
        status = "BUG" if node.is_buggy else (
            f"{node.metric_value:.6f}" if node.metric_value else "pending"
        )
        lines.append(f"{prefix}- [{node.stage}] {node.plan[:60]}... => {status}")

        for child_id in node.children_ids:
            if child_id in self.nodes:
                self._format_subtree(self.nodes[child_id], lines, indent + 1)

    def _load(self) -> None:
        if self.journal_file.exists():
            with open(self.journal_file) as f:
                data = json.load(f)
            self.nodes = {k: Node.from_dict(v) for k, v in data.items()}

    def _save(self) -> None:
        self.journal_file.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self.nodes.items()}
        with open(self.journal_file, "w") as f:
            json.dump(data, f, indent=2)
