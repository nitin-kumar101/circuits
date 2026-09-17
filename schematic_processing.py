"""
efficient_schematic_graph.py

Geometry-driven electrical connectivity extraction from vector schematic PDFs.

Pipeline
--------
PDF -> vector text + line segments
    -> spatially-indexed wire clustering
    -> net-label attachment
    -> component detection
    -> component <-> net graph
    -> optional PostgreSQL-friendly connection export

Important:
This is still a geometry-driven reconstruction. Exact MOSFET/transistor pin
identity (G/D/S, B/C/E, etc.) cannot be inferred reliably from a flattened PDF
unless pin geometry/symbol information is available. The code therefore stores
component-level connectivity plus connection confidence/distance, and provides
hooks for pin-level extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import pdfplumber


# ---------------------------------------------------------------------------
# Tunable configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphConfig:
    wire_join_tolerance: float = 1.5
    label_to_wire_tolerance: float = 12.0
    component_to_wire_tolerance: float = 40.0

    # Spatial-grid size. A few multiples of the join tolerance keeps the
    # number of candidate comparisons small without making the grid huge.
    wire_grid_cell: float = 6.0
    object_grid_cell: float = 30.0

    border_line_fraction: float = 0.85

    # Avoid treating extremely short decorative marks as wires.
    min_wire_length: float = 0.25

    # Component designators.
    refdes_prefixes: tuple[str, ...] = (
        "R", "C", "L", "D", "Q", "U", "Y", "J", "SW", "TP", "TPV", "TS",
        "FB", "K", "X", "LED", "F", "B", "T", "W", "P", "S", "M",
    )


NET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")
PAGE_REF_RE = re.compile(r"^Page\[?([\d,\s]+)\]?$", re.IGNORECASE)

NET_NAME_STOPWORDS = {
    "REV", "DNP", "TITLE", "NOTES", "TOC", "BOM", "ASSY", "OPT", "REF",
    "DES", "PAGE", "SHEET", "SIZE", "DATE", "DRAWN", "APPROVED",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Segment:
    p1: Point
    p2: Point

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(self.p1.x, self.p2.x),
            min(self.p1.y, self.p2.y),
            max(self.p1.x, self.p2.x),
            max(self.p1.y, self.p2.y),
        )


@dataclass(slots=True)
class TextLabel:
    text: str
    page: int
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(slots=True)
class WireCluster:
    page: int
    cluster_id: int
    segments: list[Segment] = field(default_factory=list)
    net_names: set[str] = field(default_factory=set)
    bbox: tuple[float, float, float, float] | None = None

    def finalize(self) -> None:
        if not self.segments:
            self.bbox = None
            return

        xs = []
        ys = []
        for s in self.segments:
            xs.extend((s.p1.x, s.p2.x))
            ys.extend((s.p1.y, s.p2.y))

        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    def min_distance_to_point(self, x: float, y: float) -> float:
        """
        Exact minimum point-to-segment distance.

        This is more accurate than checking only segment endpoints and is
        important for long wires.
        """
        best = float("inf")

        for s in self.segments:
            ax, ay = s.p1.x, s.p1.y
            bx, by = s.p2.x, s.p2.y
            dx, dy = bx - ax, by - ay
            denom = dx * dx + dy * dy

            if denom == 0:
                d = math.hypot(x - ax, y - ay)
            else:
                t = ((x - ax) * dx + (y - ay) * dy) / denom
                t = max(0.0, min(1.0, t))
                px = ax + t * dx
                py = ay + t * dy
                d = math.hypot(x - px, y - py)

            if d < best:
                best = d

        return best


@dataclass(slots=True)
class Component:
    refdes: str
    page: int
    x: float
    y: float
    value_text: Optional[str] = None
    component_type: Optional[str] = None


@dataclass(slots=True)
class Connection:
    refdes: str
    page: int
    net_id: str
    net_label: Optional[str]
    distance: float
    confidence: float


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return True


# ---------------------------------------------------------------------------
# Spatial grid
# ---------------------------------------------------------------------------

class SpatialGrid:
    """
    Lightweight spatial index.

    This avoids repeatedly comparing every object against every other object.
    For schematic PDFs containing thousands of wire segments this is one of
    the largest performance improvements over a naive O(n^2) implementation.
    """

    def __init__(self, cell_size: float):
        self.cell_size = max(cell_size, 0.001)
        self.cells: dict[tuple[int, int], list[int]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(x / self.cell_size),
            math.floor(y / self.cell_size),
        )

    def add_point(self, x: float, y: float, index: int) -> None:
        self.cells.setdefault(self._cell(x, y), []).append(index)

    def add_bbox(self, bbox: tuple[float, float, float, float], index: int) -> None:
        x0, y0, x1, y1 = bbox
        cx0, cy0 = self._cell(x0, y0)
        cx1, cy1 = self._cell(x1, y1)

        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                self.cells.setdefault((cx, cy), []).append(index)

    def nearby_point(self, x: float, y: float, radius: float) -> Iterable[int]:
        c = self._cell(x, y)
        r = max(1, math.ceil(radius / self.cell_size))

        seen = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for idx in self.cells.get((c[0] + dx, c[1] + dy), ()):
                    if idx not in seen:
                        seen.add(idx)
                        yield idx


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class SchematicGraphBuilder:
    def __init__(
        self,
        pdf_path: str,
        config: GraphConfig | None = None,
    ):
        self.pdf_path = str(pdf_path)
        self.config = config or GraphConfig()

        self.page_sizes: dict[int, tuple[float, float]] = {}
        self.labels: list[TextLabel] = []
        self.wire_clusters: dict[int, list[WireCluster]] = {}
        self.components: list[Component] = []

        self.net_label_occurrences: dict[str, list[TextLabel]] = {}
        self.explicit_page_refs: dict[tuple[int, str], set[int]] = {}

        self.connections: list[Connection] = []
        self.graph: Optional[nx.Graph] = None

        prefixes = sorted(
            self.config.refdes_prefixes,
            key=len,
            reverse=True,
        )
        self.refdes_re = re.compile(
            r"^(?:" + "|".join(prefixes) + r")\d+[A-Z]?$"
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, verbose: bool = True) -> None:
        # Clear previous state so parse() can safely be called again.
        self.labels.clear()
        self.wire_clusters.clear()
        self.components.clear()
        self.net_label_occurrences.clear()
        self.explicit_page_refs.clear()
        self.connections.clear()
        self.graph = None

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                self.page_sizes[page_idx] = (page.width, page.height)

                self._parse_page_text(page, page_idx)
                self._parse_page_wires(page, page_idx)

        self._classify_labels()

        if verbose:
            print(
                f"Parsed {len(self.page_sizes)} page(s): "
                f"{len(self.labels)} text labels, "
                f"{sum(map(len, self.wire_clusters.values()))} wire clusters, "
                f"{len(self.components)} components."
            )

    def _parse_page_text(self, page, page_idx: int) -> None:
        for w in page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        ):
            text = w["text"].strip()
            if not text:
                continue

            self.labels.append(
                TextLabel(
                    text=text,
                    page=page_idx,
                    x0=float(w["x0"]),
                    x1=float(w["x1"]),
                    top=float(w["top"]),
                    bottom=float(w["bottom"]),
                )
            )

    def _parse_page_wires(self, page, page_idx: int) -> None:
        cfg = self.config
        pw, ph = page.width, page.height
        max_len = cfg.border_line_fraction * max(pw, ph)

        segments: list[Segment] = []

        # Vector lines.
        for ln in page.lines:
            p1 = Point(float(ln["x0"]), float(ln["top"]))
            p2 = Point(float(ln["x1"]), float(ln["bottom"]))

            length = math.hypot(p2.x - p1.x, p2.y - p1.y)

            if length < cfg.min_wire_length or length > max_len:
                continue

            segments.append(Segment(p1, p2))

        # Hairline rectangles.
        for r in page.rects:
            w = abs(float(r["x1"]) - float(r["x0"]))
            h = abs(float(r["bottom"]) - float(r["top"]))

            if min(w, h) > 1.0:
                continue
            if max(w, h) < cfg.min_wire_length:
                continue
            if max(w, h) > max_len:
                continue

            p1 = Point(float(r["x0"]), float(r["top"]))
            p2 = Point(float(r["x1"]), float(r["bottom"]))
            segments.append(Segment(p1, p2))

        if not segments:
            self.wire_clusters[page_idx] = []
            return

        # Build endpoint spatial index.
        grid = SpatialGrid(
            max(cfg.wire_grid_cell, cfg.wire_join_tolerance * 2.0)
        )

        for i, s in enumerate(segments):
            grid.add_point(s.p1.x, s.p1.y, i)
            grid.add_point(s.p2.x, s.p2.y, i)

        uf = UnionFind(len(segments))
        tol = cfg.wire_join_tolerance

        # Each endpoint only checks nearby endpoint candidates.
        for i, s in enumerate(segments):
            for pt in (s.p1, s.p2):
                for j in grid.nearby_point(pt.x, pt.y, tol):
                    if j <= i:
                        continue

                    other = segments[j]

                    if (
                        self._points_close(pt, other.p1, tol)
                        or self._points_close(pt, other.p2, tol)
                    ):
                        uf.union(i, j)

        clusters_by_root: dict[int, WireCluster] = {}

        for i, segment in enumerate(segments):
            root = uf.find(i)
            cluster = clusters_by_root.get(root)

            if cluster is None:
                cluster = WireCluster(
                    page=page_idx,
                    cluster_id=root,
                )
                clusters_by_root[root] = cluster

            cluster.segments.append(segment)

        clusters = list(clusters_by_root.values())

        for idx, cluster in enumerate(clusters):
            cluster.cluster_id = idx
            cluster.finalize()

        self.wire_clusters[page_idx] = clusters

    @staticmethod
    def _points_close(a: Point, b: Point, tolerance: float) -> bool:
        return (
            abs(a.x - b.x) <= tolerance
            and abs(a.y - b.y) <= tolerance
            and math.hypot(a.x - b.x, a.y - b.y) <= tolerance
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_labels(self) -> None:
        by_line: dict[tuple[int, int], list[TextLabel]] = {}

        for lbl in self.labels:
            key = (lbl.page, round(lbl.top / 2))
            by_line.setdefault(key, []).append(lbl)

        for lbl in self.labels:
            text = lbl.text
            upper = text.upper()

            if self.refdes_re.match(upper):
                self.components.append(
                    Component(
                        refdes=upper,
                        page=lbl.page,
                        x=lbl.cx,
                        y=lbl.cy,
                        component_type=self._infer_component_type(upper),
                    )
                )
                continue

            if upper in NET_NAME_STOPWORDS:
                continue

            if NET_NAME_RE.match(upper) and not upper.isdigit():
                self.net_label_occurrences.setdefault(upper, []).append(lbl)

        # Explicit Page[...]-style references.
        for key, line_labels in by_line.items():
            line_labels.sort(key=lambda x: x.x0)

            for a, b in zip(line_labels, line_labels[1:]):
                m = PAGE_REF_RE.match(b.text)

                if m and NET_NAME_RE.match(a.text.upper()):
                    pages = {
                        int(p)
                        for p in re.findall(r"\d+", m.group(1))
                    }

                    self.explicit_page_refs.setdefault(
                        (key[0], a.text.upper()),
                        set(),
                    ).update(pages)

        # Deduplicate accidental duplicate component text labels.
        unique = {}
        for comp in self.components:
            key = (
                comp.refdes,
                comp.page,
                round(comp.x, 2),
                round(comp.y, 2),
            )
            unique[key] = comp

        self.components = list(unique.values())

    @staticmethod
    def _infer_component_type(refdes: str) -> str:
        """
        Coarse type only. This is not symbol recognition.
        """
        upper = refdes.upper()

        if upper.startswith("Q"):
            return "transistor"
        if upper.startswith("R"):
            return "resistor"
        if upper.startswith("C"):
            return "capacitor"
        if upper.startswith("L"):
            return "inductor"
        if upper.startswith("D") or upper.startswith("LED"):
            return "diode"
        if upper.startswith("U"):
            return "integrated_circuit"
        if upper.startswith("J"):
            return "connector"
        if upper.startswith("SW"):
            return "switch"
        if upper.startswith("F"):
            return "fuse"

        return "unknown"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, verbose: bool = True) -> nx.Graph:
        g = nx.Graph()

        # --------------------------------------------------------------
        # 1. Attach net labels to nearest wire cluster.
        #
        # Build one spatial index per page so this is approximately
        # O(labels * nearby_clusters) instead of O(labels * clusters).
        # --------------------------------------------------------------
        cluster_indexes: dict[int, SpatialGrid] = {}

        for page_idx, clusters in self.wire_clusters.items():
            index = SpatialGrid(self.config.object_grid_cell)

            for i, cluster in enumerate(clusters):
                if cluster.bbox:
                    # Expand bbox slightly so nearby clusters are discoverable.
                    x0, y0, x1, y1 = cluster.bbox
                    pad = self.config.label_to_wire_tolerance
                    index.add_bbox(
                        (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                        i,
                    )

            cluster_indexes[page_idx] = index

        for net_name, occurrences in self.net_label_occurrences.items():
            for lbl in occurrences:
                clusters = self.wire_clusters.get(lbl.page, [])
                index = cluster_indexes.get(lbl.page)

                if not clusters or index is None:
                    continue

                best_idx = None
                best_distance = self.config.label_to_wire_tolerance

                for idx in index.nearby_point(
                    lbl.cx,
                    lbl.cy,
                    self.config.label_to_wire_tolerance,
                ):
                    distance = clusters[idx].min_distance_to_point(
                        lbl.cx,
                        lbl.cy,
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_idx = idx

                if best_idx is not None:
                    clusters[best_idx].net_names.add(net_name)

        # --------------------------------------------------------------
        # 2. Logical nets.
        # Same named net => same logical net across pages.
        # --------------------------------------------------------------
        for net_name, occurrences in self.net_label_occurrences.items():
            g.add_node(
                f"NET:{net_name}",
                kind="net",
                label=net_name,
                pages=sorted({o.page for o in occurrences}),
            )

        cluster_to_net: dict[tuple[int, int], str] = {}
        anonymous_id = 0

        for page_idx, clusters in self.wire_clusters.items():
            for cluster in clusters:
                if cluster.net_names:
                    # Prefer a deterministic name.
                    name = sorted(cluster.net_names)[0]
                    net_id = f"NET:{name}"
                else:
                    anonymous_id += 1
                    net_id = f"NET:_anon_{page_idx}_{anonymous_id}"

                    g.add_node(
                        net_id,
                        kind="net",
                        label=None,
                        pages=[page_idx],
                    )

                cluster_to_net[(page_idx, cluster.cluster_id)] = net_id

        # --------------------------------------------------------------
        # 3. Component nodes.
        # --------------------------------------------------------------
        for comp in self.components:
            node_id = self._component_node_id(comp)

            g.add_node(
                node_id,
                kind="component",
                refdes=comp.refdes,
                component_type=comp.component_type,
                page=comp.page,
                x=comp.x,
                y=comp.y,
            )

        # --------------------------------------------------------------
        # 4. Component -> nearest wire clusters.
        # --------------------------------------------------------------
        self.connections.clear()

        for page_idx, clusters in self.wire_clusters.items():
            # Build a page-local bbox index.
            index = SpatialGrid(self.config.object_grid_cell)

            for idx, cluster in enumerate(clusters):
                if cluster.bbox:
                    x0, y0, x1, y1 = cluster.bbox
                    pad = self.config.component_to_wire_tolerance
                    index.add_bbox(
                        (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                        idx,
                    )

            page_components = [
                c for c in self.components if c.page == page_idx
            ]

            for comp in page_components:
                component_id = self._component_node_id(comp)
                linked_nets: dict[str, float] = {}

                for idx in index.nearby_point(
                    comp.x,
                    comp.y,
                    self.config.component_to_wire_tolerance,
                ):
                    cluster = clusters[idx]

                    distance = cluster.min_distance_to_point(
                        comp.x,
                        comp.y,
                    )

                    if distance > self.config.component_to_wire_tolerance:
                        continue

                    net_id = cluster_to_net.get(
                        (page_idx, cluster.cluster_id)
                    )

                    if net_id is None:
                        continue

                    previous = linked_nets.get(net_id)

                    if previous is None or distance < previous:
                        linked_nets[net_id] = distance

                for net_id, distance in linked_nets.items():
                    confidence = self._distance_confidence(
                        distance,
                        self.config.component_to_wire_tolerance,
                    )

                    g.add_edge(
                        component_id,
                        net_id,
                        distance=round(distance, 3),
                        confidence=round(confidence, 4),
                    )

                    net_label = g.nodes[net_id].get("label")

                    self.connections.append(
                        Connection(
                            refdes=comp.refdes,
                            page=comp.page,
                            net_id=net_id,
                            net_label=net_label,
                            distance=distance,
                            confidence=confidence,
                        )
                    )

        self.graph = g

        if verbose:
            n_comp = sum(
                d.get("kind") == "component"
                for _, d in g.nodes(data=True)
            )
            n_net = sum(
                d.get("kind") == "net"
                for _, d in g.nodes(data=True)
            )

            print(
                f"Graph built: {n_comp} components, "
                f"{n_net} nets, "
                f"{g.number_of_edges()} connections."
            )

        return g

    @staticmethod
    def _distance_confidence(distance: float, max_distance: float) -> float:
        if max_distance <= 0:
            return 0.0

        confidence = 1.0 - distance / max_distance
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _component_node_id(comp: Component) -> str:
        return f"COMP:{comp.refdes}@p{comp.page}"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _require_graph(self) -> nx.Graph:
        if self.graph is None:
            raise RuntimeError("Call build_graph() first.")
        return self.graph

    def component_nodes(self, refdes: str) -> list[str]:
        g = self._require_graph()
        refdes = refdes.upper()

        return [
            node
            for node, data in g.nodes(data=True)
            if data.get("kind") == "component"
            and data.get("refdes") == refdes
        ]

    def net_of(self, refdes: str) -> list[dict]:
        """
        Return nets touched by a component.

        Example:
            builder.net_of("Q3")
        """
        g = self._require_graph()
        result = []

        for component_id in self.component_nodes(refdes):
            for net_id in g.neighbors(component_id):
                net_data = g.nodes[net_id]
                edge = g.edges[component_id, net_id]

                result.append({
                    "net_id": net_id,
                    "net_name": net_data.get("label"),
                    "page": net_data.get("pages"),
                    "distance": edge.get("distance"),
                    "confidence": edge.get("confidence"),
                })

        return result

    def components_on_net(self, net_name: str) -> list[dict]:
        g = self._require_graph()
        node_id = f"NET:{net_name.upper()}"

        if node_id not in g:
            return []

        result = []

        for node in g.neighbors(node_id):
            data = g.nodes[node]

            if data.get("kind") != "component":
                continue

            edge = g.edges[node, node_id]

            result.append({
                "refdes": data["refdes"],
                "component_type": data.get("component_type"),
                "page": data["page"],
                "distance": edge.get("distance"),
                "confidence": edge.get("confidence"),
            })

        return result

    def connected_components(
        self,
        refdes: str,
        include_same_net: bool = True,
    ) -> list[dict]:
        """
        Direct electrical neighbors.

        For Q3:
            1. find all nets Q3 touches
            2. find all other components on those nets

        This is the key query for a frontend "what is connected to Q3?"
        use case.
        """
        g = self._require_graph()
        refdes = refdes.upper()

        result = {}
        for component_id in self.component_nodes(refdes):
            for net_id in g.neighbors(component_id):
                for neighbor in g.neighbors(net_id):
                    if neighbor == component_id:
                        continue

                    data = g.nodes[neighbor]
                    if data.get("kind") != "component":
                        continue

                    key = neighbor
                    edge = g.edges[neighbor, net_id]

                    existing = result.get(key)
                    candidate = {
                        "refdes": data["refdes"],
                        "component_type": data.get("component_type"),
                        "page": data["page"],
                        "via_net": g.nodes[net_id].get("label") or net_id,
                        "confidence": edge.get("confidence"),
                    }

                    if existing is None or candidate["confidence"] > existing["confidence"]:
                        result[key] = candidate

        return sorted(
            result.values(),
            key=lambda x: (
                -(x["confidence"] or 0),
                x["refdes"],
            ),
        )

    def path_between(
        self,
        refdes_a: str,
        refdes_b: str,
    ) -> Optional[list[str]]:
        g = self._require_graph()

        best = None

        for a in self.component_nodes(refdes_a):
            for b in self.component_nodes(refdes_b):
                try:
                    path = nx.shortest_path(g, a, b)
                except nx.NetworkXNoPath:
                    continue

                if best is None or len(path) < len(best):
                    best = path

        return best

    # ------------------------------------------------------------------
    # PostgreSQL-friendly export
    # ------------------------------------------------------------------

    def export_connections_csv(self, path: str) -> None:
        """
        Flat table suitable for PostgreSQL COPY/import.

        Columns:
            refdes, page, net_id, net_label, distance, confidence
        """
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "refdes",
                "page",
                "net_id",
                "net_label",
                "distance",
                "confidence",
            ])

            for c in self.connections:
                writer.writerow([
                    c.refdes,
                    c.page,
                    c.net_id,
                    c.net_label or "",
                    round(c.distance, 3),
                    round(c.confidence, 4),
                ])

    def export_postgres_sql(self, path: str) -> None:
        """
        Generate schema + INSERT statements.

        For large schematics, prefer CSV + PostgreSQL COPY rather than INSERTs.
        """
        g = self._require_graph()

        lines = [
            "-- Generated by efficient_schematic_graph.py",
            "",
            "CREATE TABLE IF NOT EXISTS components (",
            "    id BIGSERIAL PRIMARY KEY,",
            "    refdes TEXT NOT NULL,",
            "    page INTEGER NOT NULL,",
            "    component_type TEXT,",
            "    x DOUBLE PRECISION,",
            "    y DOUBLE PRECISION",
            ");",
            "",
            "CREATE TABLE IF NOT EXISTS nets (",
            "    id TEXT PRIMARY KEY,",
            "    name TEXT,",
            "    pages INTEGER[]",
            ");",
            "",
            "CREATE TABLE IF NOT EXISTS connections (",
            "    component_id BIGINT NOT NULL,",
            "    net_id TEXT NOT NULL,",
            "    distance DOUBLE PRECISION,",
            "    confidence DOUBLE PRECISION",
            ");",
            "",
            "CREATE INDEX IF NOT EXISTS idx_components_refdes",
            "    ON components(refdes);",
            "",
            "CREATE INDEX IF NOT EXISTS idx_connections_component",
            "    ON connections(component_id);",
            "",
            "CREATE INDEX IF NOT EXISTS idx_connections_net",
            "    ON connections(net_id);",
            "",
        ]

        # The SQL exporter intentionally emits the graph structure rather than
        # attempting to create application-specific IDs.
        # For production ingestion, CSV + COPY is recommended.
        lines.append("-- Component rows")
        for _, data in g.nodes(data=True):
            if data.get("kind") != "component":
                continue

            refdes = data["refdes"].replace("'", "''")
            ctype = (data.get("component_type") or "").replace("'", "''")

            lines.append(
                "INSERT INTO components "
                "(refdes, page, component_type, x, y) VALUES "
                f"('{refdes}', {int(data['page'])}, '{ctype}', "
                f"{data['x']}, {data['y']});"
            )

        lines.append("")
        lines.append("-- Net rows")

        for node_id, data in g.nodes(data=True):
            if data.get("kind") != "net":
                continue

            name = data.get("label")
            name_sql = "NULL" if name is None else "'" + name.replace("'", "''") + "'"
            pages = "{" + ",".join(map(str, data.get("pages", []))) + "}"

            lines.append(
                "INSERT INTO nets (id, name, pages) VALUES "
                f"('{node_id}', {name_sql}, '{pages}');"
            )

        Path(path).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # JSON / GraphML
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        g = self._require_graph()

        data = nx.node_link_data(
            g,
            edges="edges",
        )

        Path(path).write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

    def export_graphml(self, path: str) -> None:
        g = self._require_graph()
        export_graph = g.copy()

        for _, data in export_graph.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, set, tuple)):
                    data[key] = ",".join(map(str, value))
                elif value is None:
                    data[key] = ""

        for _, _, data in export_graph.edges(data=True):
            for key, value in list(data.items()):
                if value is None:
                    data[key] = ""

        nx.write_graphml(export_graph, path)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        g = self._require_graph()

        component_count = sum(
            data.get("kind") == "component"
            for _, data in g.nodes(data=True)
        )

        named_nets = sum(
            data.get("kind") == "net" and bool(data.get("label"))
            for _, data in g.nodes(data=True)
        )

        anonymous_nets = sum(
            data.get("kind") == "net" and not data.get("label")
            for _, data in g.nodes(data=True)
        )

        print("=" * 70)
        print(f"Components:       {component_count}")
        print(f"Named nets:       {named_nets}")
        print(f"Anonymous nets:   {anonymous_nets}")
        print(f"Connections:      {g.number_of_edges()}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an electrical connectivity graph from a schematic PDF."
    )

    parser.add_argument("pdf", help="Path to schematic PDF")
    parser.add_argument(
        "--out-prefix",
        default="schematic",
        help="Output prefix",
    )
    parser.add_argument(
        "--no-graphml",
        action="store_true",
        help="Do not write GraphML",
    )

    args = parser.parse_args()

    builder = SchematicGraphBuilder(args.pdf)

    builder.parse()
    builder.build_graph()
    builder.summary()

    prefix = Path(args.out_prefix)

    builder.export_json(str(prefix.with_suffix(".json")))
    builder.export_connections_csv(
        str(prefix.parent / f"{prefix.name}_connections.csv")
    )

    if not args.no_graphml:
        builder.export_graphml(str(prefix.with_suffix(".graphml")))

    print("Done.")


if __name__ == "__main__":
    main()
