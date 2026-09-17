"""
schematic_retrieval.py

Multi-hop retrieval over the component/net graph produced by
`efficient_schematic_graph.SchematicGraphBuilder`.

The builder already gives you 1-hop queries (net_of, components_on_net,
connected_components) and single shortest-path queries (path_between).
This module adds:

    1. Entry-point resolution from a free-text query (refdes match,
       net-name match, fuzzy fallback).
    2. Bounded multi-hop BFS expansion from one or more entry nodes,
       with per-hop confidence decay and a node budget.
    3. Multi-hop *path* retrieval between two or more components
       (not just the single shortest path).
    4. A formatter that turns the retrieved subgraph into compact,
       citation-friendly text you can drop straight into an LLM prompt
       (RAG-style retrieval over a schematic).

Usage
-----
    from efficient_schematic_graph import SchematicGraphBuilder
    from schematic_retrieval import MultiHopRetriever

    builder = SchematicGraphBuilder("board.pdf")
    builder.parse()
    builder.build_graph()

    retriever = MultiHopRetriever(builder)

    result = retriever.retrieve("what is connected to Q3?", max_hops=2)
    print(retriever.to_context_text(result))

    paths = retriever.retrieve_paths("Q3", "U2", max_hops=4, max_paths=5)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from efficient_schematic_graph import SchematicGraphBuilder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    # How many component<->net hops to expand from each entry point.
    # One "hop" here = component -> net -> component.
    max_hops: int = 2

    # Multiply cumulative confidence by this factor per hop, so distant
    # nodes are still returned but ranked lower.
    hop_decay: float = 0.85

    # Stop expanding once this many nodes have been collected, to keep
    # results bounded on densely connected boards.
    max_nodes: int = 200

    # Below this cumulative confidence a node is dropped even if within
    # max_hops.
    min_confidence: float = 0.05

    # Fuzzy-match cutoff (0-1) used when a query term doesn't exactly
    # match a refdes or net name.
    fuzzy_cutoff: float = 0.6


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievedNode:
    node_id: str
    kind: str                  # "component" | "net"
    hop: int                   # hops from nearest entry point
    confidence: float          # decayed confidence, 0-1
    data: dict = field(default_factory=dict)


@dataclass
class RetrievedEdge:
    source: str
    target: str
    distance: Optional[float]
    confidence: Optional[float]


@dataclass
class RetrievalResult:
    query: str
    entry_nodes: list[str]
    nodes: dict[str, RetrievedNode]
    edges: list[RetrievedEdge]

    def components(self) -> list[RetrievedNode]:
        return sorted(
            (n for n in self.nodes.values() if n.kind == "component"),
            key=lambda n: (n.hop, -n.confidence),
        )

    def nets(self) -> list[RetrievedNode]:
        return sorted(
            (n for n in self.nodes.values() if n.kind == "net"),
            key=lambda n: (n.hop, -n.confidence),
        )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MultiHopRetriever:
    def __init__(
        self,
        builder: SchematicGraphBuilder,
        config: RetrievalConfig | None = None,
    ):
        if builder.graph is None:
            raise RuntimeError(
                "Call builder.build_graph() before creating a retriever."
            )

        self.builder = builder
        self.graph: nx.Graph = builder.graph
        self.config = config or RetrievalConfig()

        # Precompute lookup indexes so entry-point resolution doesn't scan
        # the whole graph on every query.
        self._component_nodes: dict[str, list[str]] = {}
        self._net_nodes: dict[str, str] = {}

        for node_id, data in self.graph.nodes(data=True):
            if data.get("kind") == "component":
                self._component_nodes.setdefault(
                    data["refdes"], []
                ).append(node_id)
            elif data.get("kind") == "net" and data.get("label"):
                self._net_nodes[data["label"]] = node_id

    # ------------------------------------------------------------------
    # Entry-point resolution
    # ------------------------------------------------------------------

    def resolve_entry_nodes(self, query: str) -> list[str]:
        """
        Turn a free-text query into a list of starting node ids.

        Strategy, in order:
            1. Exact refdes tokens in the query (e.g. "Q3", "R101").
            2. Exact net-name tokens (e.g. "VCC_3V3").
            3. Fuzzy match against known refdes / net names for any
               remaining capitalized-looking tokens.
        """
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.upper())
        entries: list[str] = []
        matched_tokens: set[str] = set()

        for tok in tokens:
            if tok in self._component_nodes:
                entries.extend(self._component_nodes[tok])
                matched_tokens.add(tok)
            elif tok in self._net_nodes:
                entries.append(self._net_nodes[tok])
                matched_tokens.add(tok)

        if entries:
            return sorted(set(entries))

        # Fuzzy fallback: try to match unmatched tokens against known
        # refdes / net-name vocab.
        vocab = list(self._component_nodes.keys()) + list(self._net_nodes.keys())

        for tok in tokens:
            if tok in matched_tokens or len(tok) < 2:
                continue

            close = difflib.get_close_matches(
                tok, vocab, n=1, cutoff=self.config.fuzzy_cutoff
            )
            if not close:
                continue

            match = close[0]
            if match in self._component_nodes:
                entries.extend(self._component_nodes[match])
            elif match in self._net_nodes:
                entries.append(self._net_nodes[match])

        return sorted(set(entries))

    # ------------------------------------------------------------------
    # Multi-hop BFS expansion
    # ------------------------------------------------------------------

    def expand(
        self,
        entry_nodes: list[str],
        max_hops: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Bounded BFS from one or more entry nodes.

        Confidence is cumulative: an edge's own confidence (from the
        geometry-based distance score) times the decayed confidence of
        the node it's being reached from. This means a node reachable
        by a short, high-confidence chain of connections outranks one
        reachable only via long or uncertain hops, even at the same
        hop-distance.
        """
        cfg = self.config
        hops = max_hops if max_hops is not None else cfg.max_hops

        nodes: dict[str, RetrievedNode] = {}
        edges: list[RetrievedEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        frontier: list[tuple[str, int, float]] = []

        for node_id in entry_nodes:
            if node_id not in self.graph:
                continue

            data = dict(self.graph.nodes[node_id])
            nodes[node_id] = RetrievedNode(
                node_id=node_id,
                kind=data.get("kind", "unknown"),
                hop=0,
                confidence=1.0,
                data=data,
            )
            frontier.append((node_id, 0, 1.0))

        while frontier and len(nodes) < cfg.max_nodes:
            current_id, hop, confidence = frontier.pop(0)

            if hop >= hops:
                continue

            for neighbor_id in self.graph.neighbors(current_id):
                edge_data = self.graph.edges[current_id, neighbor_id]
                edge_conf = edge_data.get("confidence", 1.0) or 1.0

                next_confidence = confidence * edge_conf * cfg.hop_decay
                next_hop = hop + 1

                edge_key = tuple(sorted((current_id, neighbor_id)))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(
                        RetrievedEdge(
                            source=current_id,
                            target=neighbor_id,
                            distance=edge_data.get("distance"),
                            confidence=edge_data.get("confidence"),
                        )
                    )

                if next_confidence < cfg.min_confidence:
                    continue

                existing = nodes.get(neighbor_id)

                if existing is not None and (
                    existing.hop < next_hop
                    or existing.confidence >= next_confidence
                ):
                    # Already have an equal-or-better path to this node.
                    continue

                neighbor_data = dict(self.graph.nodes[neighbor_id])
                nodes[neighbor_id] = RetrievedNode(
                    node_id=neighbor_id,
                    kind=neighbor_data.get("kind", "unknown"),
                    hop=next_hop,
                    confidence=round(next_confidence, 4),
                    data=neighbor_data,
                )

                if len(nodes) >= cfg.max_nodes:
                    break

                frontier.append((neighbor_id, next_hop, next_confidence))

        return RetrievalResult(
            query="",
            entry_nodes=list(entry_nodes),
            nodes=nodes,
            edges=edges,
        )

    # ------------------------------------------------------------------
    # High-level query entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_hops: Optional[int] = None,
    ) -> RetrievalResult:
        entry_nodes = self.resolve_entry_nodes(query)

        if not entry_nodes:
            return RetrievalResult(
                query=query,
                entry_nodes=[],
                nodes={},
                edges=[],
            )

        result = self.expand(entry_nodes, max_hops=max_hops)
        result.query = query
        return result

    # ------------------------------------------------------------------
    # Multi-hop path retrieval between named components
    # ------------------------------------------------------------------

    def retrieve_paths(
        self,
        refdes_a: str,
        refdes_b: str,
        max_hops: int = 6,
        max_paths: int = 5,
    ) -> list[list[str]]:
        """
        Return up to `max_paths` simple paths between any instance of
        refdes_a and any instance of refdes_b, each no longer than
        `max_hops` component<->net hops (i.e. cutoff = 2 * max_hops
        graph edges, since paths alternate component/net nodes).
        """
        a_nodes = self.builder.component_nodes(refdes_a)
        b_nodes = self.builder.component_nodes(refdes_b)

        if not a_nodes or not b_nodes:
            return []

        cutoff = max_hops * 2
        found: list[list[str]] = []

        for a in a_nodes:
            for b in b_nodes:
                if a == b:
                    continue

                try:
                    gen = nx.shortest_simple_paths(self.graph, a, b)
                except nx.NetworkXNoPath:
                    continue

                for path in gen:
                    if len(path) - 1 > cutoff:
                        break

                    found.append(path)

                    if len(found) >= max_paths:
                        return found

        return found

    # ------------------------------------------------------------------
    # Formatting for LLM context
    # ------------------------------------------------------------------

    def to_context_text(self, result: RetrievalResult) -> str:
        if not result.nodes:
            return f'No matches found for query: "{result.query}"'

        lines = [f'Retrieved schematic context for: "{result.query}"', ""]

        lines.append("Components:")
        for n in result.components():
            refdes = n.data.get("refdes")
            ctype = n.data.get("component_type") or "unknown"
            page = n.data.get("page")
            lines.append(
                f"  - {refdes} (type={ctype}, page={page}, "
                f"hop={n.hop}, confidence={n.confidence:.2f})"
            )

        lines.append("")
        lines.append("Nets:")
        for n in result.nets():
            label = n.data.get("label") or "(unnamed)"
            pages = n.data.get("pages")
            lines.append(
                f"  - {label} (pages={pages}, hop={n.hop}, "
                f"confidence={n.confidence:.2f})"
            )

        lines.append("")
        lines.append("Connections:")
        for e in result.edges:
            src = result.nodes.get(e.source)
            dst = result.nodes.get(e.target)
            if src is None or dst is None:
                continue

            src_label = self._display_label(src)
            dst_label = self._display_label(dst)

            lines.append(
                f"  - {src_label} -- {dst_label} "
                f"(distance={e.distance}, confidence={e.confidence})"
            )

        return "\n".join(lines)

    @staticmethod
    def _display_label(node: RetrievedNode) -> str:
        if node.kind == "component":
            return node.data.get("refdes", node.node_id)
        return node.data.get("label") or node.node_id

    def path_to_context_text(self, path: list[str]) -> str:
        parts = []

        for node_id in path:
            data = self.graph.nodes[node_id]

            if data.get("kind") == "component":
                parts.append(data.get("refdes", node_id))
            else:
                parts.append(data.get("label") or "(unnamed net)")

        return " -> ".join(parts)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-hop retrieval demo over a schematic graph."
    )
    parser.add_argument("pdf", help="Path to schematic PDF")
    parser.add_argument("query", help='e.g. "what is connected to Q3?"')
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument(
        "--path-to",
        help="Optional second refdes; prints multi-hop paths to it instead "
        "of a neighborhood expansion.",
    )

    args = parser.parse_args()

    builder = SchematicGraphBuilder(args.pdf)
    builder.parse()
    builder.build_graph()

    retriever = MultiHopRetriever(builder)

    if args.path_to:
        entry_tokens = re.findall(r"[A-Za-z0-9_]+", args.query.upper())
        refdes_a = entry_tokens[0] if entry_tokens else args.query.upper()

        paths = retriever.retrieve_paths(
            refdes_a, args.path_to.upper(), max_hops=args.max_hops
        )

        if not paths:
            print(f"No path found between {refdes_a} and {args.path_to.upper()}.")
            return

        for i, path in enumerate(paths, start=1):
            print(f"Path {i}: {retriever.path_to_context_text(path)}")
        return

    result = retriever.retrieve(args.query, max_hops=args.max_hops)
    print(retriever.to_context_text(result))


if __name__ == "__main__":
    main()
