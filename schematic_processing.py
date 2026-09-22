#!/usr/bin/env python3
"""
schematic_graph_v4.py

Generic, fast schematic connectivity extractor for vector PDF schematics.

Design goals
------------
* Works without Altium/KiCad-specific hidden metadata.
* Automatically handles common wire colours (red/blue/magenta/black/etc.).
* Uses visible text for reference designators and net names.
* Reconstructs horizontal/vertical wire topology with a spatial index.
* Handles endpoint joins, T-junctions and explicit junction dots.
* Does not treat arbitrary proximity as a physical wire connection.
* Falls back gracefully on PDFs that contain little/no vector geometry.
* Produces an auditable component <-> net graph even when exact pin numbers
  are not recoverable from the PDF.

Outputs
-------
<prefix>.json
<prefix>.graphml
<prefix>_component_nets.csv
<prefix>_component_edges.csv
<prefix>_unresolved_components.csv
<prefix>_net_labels.csv
<prefix>_summary.json
optional: <prefix>_overview.png

Usage
-----
python schematic_graph_v4.py input.pdf --out-prefix result --viz

Dependencies: PyMuPDF, NetworkX
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable

import fitz
import networkx as nx


# ----------------------------- configuration -----------------------------
WIRE_JOIN_TOL = 1.8
T_JUNCTION_TOL = 1.8
JUNCTION_DOT_TOL = 2.5
LABEL_WIRE_TOL = 18.0
COMPONENT_WIRE_TOL = 18.0
BODY_WIRE_TOL = 7.0
MIN_SEGMENT = 1.5
MAX_WIRE_ANGLE_DEG = 4.0
MAX_BORDER_FRACTION = 0.86
GRID_CELL = 12.0

# Common reference designators. The parser is deliberately conservative:
# arbitrary capital words are not components.
REF_PREFIXES = (
    "LED", "BAT", "OSC", "CON", "CN", "FB", "TP", "SW", "IC",
    "R", "C", "L", "D", "Q", "U", "Y", "J", "K", "X", "F",
    "B", "T", "W", "P", "S", "M",
)
REF_RE = re.compile(r"^(?:" + "|".join(sorted(REF_PREFIXES, key=len, reverse=True)) + r")\d+[A-Z]?$", re.I)

# Net names are intentionally broader than the previous versions. These
# examples are all common in real schematics: +3V, VCORE, RESET#, USBP1-,
# H_A#35, PCIE_RXP0, /RESET, etc.
NET_RE = re.compile(r"^[+\-/]?[A-Za-z][A-Za-z0-9_#./+\-]*$|^[A-Z0-9_+#./\-]{2,}$")
PAGE_RE = re.compile(r"^(?:Page|Sheet)\s*\[?[^\]]+\]?$", re.I)

STOPWORDS = {
    "SIZE", "DOCUMENT", "NUMBER", "REV", "DATE", "SHEET", "OF", "PROJECT",
    "CUSTOM", "QUANTA", "COMPUTER", "INC", "TITLE", "NOTES", "DRAWN",
    "APPROVED", "CHECKED", "BLOCK", "DIAGRAM", "FOR", "SUPPORT", "ONLY",
    "PAGE", "GND",  # GND is handled as a power net, but still allowed later.
}
POWER_WORDS = {
    "GND", "VDD", "VSS", "VCC", "VDDA", "VSSA", "VBAT", "VBUS",
}


@dataclass(frozen=True)
class Segment:
    page: int
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[float, float, float] | None = None

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def horizontal(self) -> bool:
        return abs(self.y2 - self.y1) <= abs(self.x2 - self.x1) * 0.08 + 0.25

    @property
    def vertical(self) -> bool:
        return abs(self.x2 - self.x1) <= abs(self.y2 - self.y1) * 0.08 + 0.25


@dataclass
class TextItem:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2


@dataclass
class Component:
    ref: str
    page: int
    x: float
    y: float
    source: str = "visible-text"
    occurrences: int = 1


@dataclass
class NetLabel:
    name: str
    page: int
    x: float
    y: float
    source: str = "visible-text"


@dataclass
class WireCluster:
    page: int
    cluster_id: int
    segments: list[int] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)

    def distance_to_point(self, x: float, y: float, segs: list[Segment]) -> float:
        best = float("inf")
        for i in self.segments:
            s = segs[i]
            best = min(best, point_segment_distance(x, y, s))
        return best

    def nearest_segment_distance(self, x: float, y: float, segs: list[Segment]):
        best = (float("inf"), None)
        for i in self.segments:
            d = point_segment_distance(x, y, segs[i])
            if d < best[0]:
                best = (d, segs[i])
        return best


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


class SpatialGrid:
    """Uniform grid used to avoid O(N^2) segment comparisons."""
    def __init__(self, cell=GRID_CELL):
        self.cell = cell
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def key(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def insert_segment(self, idx, s: Segment):
        xa, xb = sorted((s.x1, s.x2)); ya, yb = sorted((s.y1, s.y2))
        cx0, cx1 = self.key(xa, 0)[0], self.key(xb, 0)[0]
        cy0, cy1 = self.key(0, ya)[1], self.key(0, yb)[1]
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                self.buckets[(cx, cy)].append(idx)

    def nearby(self, x, y, radius=1):
        cx, cy = self.key(x, y)
        out, seen = [], set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for idx in self.buckets.get((cx + dx, cy + dy), ()):
                    if idx not in seen:
                        seen.add(idx); out.append(idx)
        return out


def point_segment_distance(px, py, s: Segment):
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    den = dx * dx + dy * dy
    if den == 0:
        return math.hypot(px - s.x1, py - s.y1)
    t = ((px - s.x1) * dx + (py - s.y1) * dy) / den
    t = max(0.0, min(1.0, t))
    qx, qy = s.x1 + t * dx, s.y1 + t * dy
    return math.hypot(px - qx, py - qy)


def point_line_distance(px, py, s: Segment):
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    den = math.hypot(dx, dy)
    if den == 0:
        return math.hypot(px - s.x1, py - s.y1)
    return abs(dy * px - dx * py + s.x2 * s.y1 - s.y2 * s.x1) / den


def endpoint_near(x, y, s: Segment, tol=1.8):
    return math.hypot(x - s.x1, y - s.y1) <= tol or math.hypot(x - s.x2, y - s.y2) <= tol


def is_axis(s: Segment):
    if s.length == 0:
        return False
    angle = math.degrees(math.atan2(abs(s.y2 - s.y1), abs(s.x2 - s.x1)))
    angle = min(angle, 90 - angle)
    return angle <= MAX_WIRE_ANGLE_DEG


def color_key(c):
    if c is None:
        return None
    return tuple(round(float(x), 3) for x in c)


def parse_ref(text: str) -> bool:
    return bool(REF_RE.fullmatch(text.strip()))


def likely_net(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 80 or PAGE_RE.fullmatch(t):
        return False
    if t.upper() in STOPWORDS:
        return t.upper() == "GND"
    if parse_ref(t):
        return False
    if t.isdigit():
        return False
    # Values, dates, dimensions, and ordinary prose are not net labels.
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:K|M|U|N|P|V|A|OHM|F)?", t, re.I):
        return False
    if not NET_RE.fullmatch(t):
        return False
    if t.startswith(("+", "/")):
        return True
    # Strong value/part-number filters. Component values such as
    # 10K/F_4, .1U/10V_4 and HCB1608KF-181T15_6 are abundant in board
    # schematics and must not become electrical nets merely because they
    # happen to sit close to a wire.
    if re.fullmatch(r"[.+-]?\d+(?:\.\d+)?[A-Za-z]+(?:/[A-Za-z0-9.]+)?(?:_\d+)?", t):
        return False
    if "/" in t and re.match(r"^[.+-]?\d", t):
        return False
    if re.fullmatch(r"[A-Za-z]{2,8}\d{3,}[A-Za-z0-9_-]*", t) and "_" not in t:
        return False
    # Part numbers containing several digit runs and no obvious signal
    # separator are usually values rather than nets. Keep names such as
    # PCIE_RXP6, H_D#35 and CPU_VID0.
    digit_runs = len(re.findall(r"\d+", t))
    if digit_runs >= 3 and not any(ch in t for ch in "_#"):
        return False
    if t.upper() in POWER_WORDS or t.startswith(("+", "/")):
        return True
    return any(ch in t for ch in "_#-./") or any(ch.isdigit() for ch in t) or t.isupper()


def is_probable_wire_segment(s: Segment, page_w, page_h):
    if s.length < MIN_SEGMENT:
        return False
    if s.length > MAX_BORDER_FRACTION * max(page_w, page_h):
        return False
    if not is_axis(s):
        return False
    return True


def extract_segments(page, page_idx):
    """Extract all plausible horizontal/vertical vector lines, independent of colour."""
    pw, ph = page.rect.width, page.rect.height
    out = []
    for dr in page.get_drawings():
        stroke = color_key(dr.get("color"))
        for item in dr.get("items", []):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            s = Segment(page_idx, len(out), a.x, a.y, b.x, b.y, stroke)
            if is_probable_wire_segment(s, pw, ph):
                out.append(s)
    return out


def extract_text(page, page_idx):
    out = []
    for w in page.get_text("words"):
        if len(w) < 5:
            continue
        text = str(w[4]).strip()
        if not text:
            continue
        out.append(TextItem(page_idx, text, float(w[0]), float(w[1]), float(w[2]), float(w[3]), 0.0))
    return out


def extract_junction_dots(page):
    dots = []
    for dr in page.get_drawings():
        fill = dr.get("fill")
        r = dr.get("rect")
        if not fill or r is None:
            continue
        w, h = r.width, r.height
        if 0.8 <= w <= 8 and 0.8 <= h <= 8 and max(fill) - min(fill) > 0.05:
            dots.append((r.x0 + r.width / 2, r.y0 + r.height / 2))
        elif 0.8 <= w <= 8 and 0.8 <= h <= 8:
            dots.append((r.x0 + r.width / 2, r.y0 + r.height / 2))
    return dots


def build_clusters(segments: list[Segment], dots):
    """Topology reconstruction using a spatial grid; never performs all-pairs testing."""
    n = len(segments)
    if n == 0:
        return [], {}, UnionFind(0)
    uf = UnionFind(n)
    grid = SpatialGrid()
    for i, s in enumerate(segments):
        grid.insert_segment(i, s)

    # Endpoint -> endpoint or endpoint -> interior.
    for i, s in enumerate(segments):
        for x, y in ((s.x1, s.y1), (s.x2, s.y2)):
            for j in grid.nearby(x, y, radius=1):
                if j == i:
                    continue
                q = segments[j]
                if endpoint_near(x, y, q, WIRE_JOIN_TOL):
                    uf.union(i, j)
                elif point_line_distance(x, y, q) <= T_JUNCTION_TOL:
                    # Only accept a T if the projected point is actually inside q.
                    dx, dy = q.x2 - q.x1, q.y2 - q.y1
                    den = dx * dx + dy * dy
                    if den:
                        t = ((x - q.x1) * dx + (y - q.y1) * dy) / den
                        if 0.02 < t < 0.98:
                            uf.union(i, j)

    # Explicit junction dots can join a true crossing.
    for x, y in dots:
        touching = []
        for j in grid.nearby(x, y, radius=1):
            if point_segment_distance(x, y, segments[j]) <= JUNCTION_DOT_TOL:
                touching.append(j)
        if len(touching) >= 2:
            for j in touching[1:]:
                uf.union(touching[0], j)

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    clusters = []
    root_to_cluster = {}
    for cid, (root, ids) in enumerate(groups.items()):
        root_to_cluster[root] = cid
        clusters.append(WireCluster(page=segments[0].page, cluster_id=cid, segments=ids))
    return clusters, root_to_cluster, uf


def nearest_cluster(x, y, clusters, segments, max_dist):
    best = None
    best_d = max_dist
    for wc in clusters:
        d = wc.distance_to_point(x, y, segments)
        if d < best_d:
            best_d, best = d, wc
    return best, best_d


def body_candidates(page):
    """Find plausible IC/component body rectangles without assuming a specific EDA format."""
    rects = []
    for dr in page.get_drawings():
        r = dr.get("rect")
        if r is None:
            continue
        w, h = r.width, r.height
        if 8 <= w <= 500 and 8 <= h <= 500 and min(w, h) >= 10:
            # Very large page frames/title boxes are ignored.
            if w < page.rect.width * 0.75 and h < page.rect.height * 0.75:
                rects.append(r)
    return rects


def rect_boundary_distance(x, y, r):
    dx = max(r.x0 - x, 0, x - r.x1)
    dy = max(r.y0 - y, 0, y - r.y1)
    return math.hypot(dx, dy)


class Extractor:
    def __init__(self, pdf_path):
        self.pdf_path = str(pdf_path)
        self.doc = fitz.open(self.pdf_path)
        self.pages = {}
        self.text = {}
        self.segments = {}
        self.clusters = {}
        self.uf = {}
        self.root_to_cluster = {}
        self.seg_grid = {}
        self.body_rects = {}
        self.components: list[Component] = []
        self.net_labels: list[NetLabel] = []
        self.cluster_labels: dict[tuple[int, int], set[str]] = defaultdict(set)
        self.component_nets: list[dict] = []
        self.unresolved: list[dict] = []
        self.graph = nx.Graph()
        self.stats = {}

    def parse(self, verbose=True):
        # Pass 1: page extraction. No expensive nested geometry loops here.
        for pi, page in enumerate(self.doc, start=1):
            self.pages[pi] = page
            words = extract_text(page, pi)
            self.text[pi] = words
            self.segments[pi] = extract_segments(page, pi)
            self.body_rects[pi] = body_candidates(page)
            dots = extract_junction_dots(page)
            clusters, roots, uf = build_clusters(self.segments[pi], dots)
            self.clusters[pi] = clusters
            self.uf[pi] = uf
            self.root_to_cluster[pi] = roots
            grid = SpatialGrid()
            for si, seg in enumerate(self.segments[pi]):
                grid.insert_segment(si, seg)
            self.seg_grid[pi] = grid

            if verbose:
                print(f"page {pi:>2}/{len(self.doc)}: text={len(words):>4}, vector-lines={len(self.segments[pi]):>5}, wire-clusters={len(clusters):>4}")

        self._extract_components_and_labels()
        self._attach_labels()
        self._attach_components()
        self._build_graph()
        self.stats = self.summary()
        return self

    def _nearest_cluster_fast(self, page_idx, x, y, max_dist):
        segs = self.segments[page_idx]
        grid = self.seg_grid[page_idx]
        candidates = grid.nearby(x, y, radius=max(1, int(math.ceil(max_dist / grid.cell))))
        best_d = max_dist
        best_cid = None
        roots = self.root_to_cluster[page_idx]
        uf = self.uf[page_idx]
        for si in candidates:
            d = point_segment_distance(x, y, segs[si])
            if d < best_d:
                best_d = d
                best_cid = roots.get(uf.find(si))
        if best_cid is None:
            return None, best_d
        return self.clusters[page_idx][best_cid], best_d

    def _extract_components_and_labels(self):
        # Deduplicate identical reference text occurrences on the same page.
        by_ref_page = defaultdict(list)
        for pi, words in self.text.items():
            for w in words:
                if parse_ref(w.text):
                    by_ref_page[(pi, w.text.upper())].append(w)

        for (pi, ref), occs in by_ref_page.items():
            # A real symbol reference is often printed twice in exported PDFs.
            # Use the median occurrence, which avoids relying on draw order.
            x = sum(o.cx for o in occs) / len(occs)
            y = sum(o.cy for o in occs) / len(occs)
            self.components.append(Component(ref, pi, x, y, occurrences=len(occs)))

        for pi, words in self.text.items():
            for w in words:
                if likely_net(w.text):
                    self.net_labels.append(NetLabel(w.text, pi, w.cx, w.cy))

        # Deduplicate exact label positions/names.
        seen = set(); unique = []
        for n in self.net_labels:
            key = (n.page, n.name, round(n.x, 1), round(n.y, 1))
            if key not in seen:
                seen.add(key); unique.append(n)
        self.net_labels = unique

    def _attach_labels(self):
        for label in self.net_labels:
            # Avoid title-block labels: only accept labels with nearby vector geometry.
            wc, d = self._nearest_cluster_fast(label.page, label.x, label.y, LABEL_WIRE_TOL)
            if wc is None:
                continue
            self.cluster_labels[(label.page, wc.cluster_id)].add(label.name)

    def _component_candidate_clusters(self, comp: Component):
        # Generic fallback: connect a component only to the closest traced
        # physical wire evidence. We deliberately do NOT attach every wire
        # inside a radius; that creates a combinatorial explosion and many
        # false positives on dense BGA/CPU sheets.
        wc0, d0 = self._nearest_cluster_fast(comp.page, comp.x, comp.y, COMPONENT_WIRE_TOL)
        if wc0 is None:
            return []
        return [(d0, wc0, "refdes-near-wire")]

    def _attach_components(self):
        for comp in self.components:
            candidates = self._component_candidate_clusters(comp)
            # Do not attach to every nearby line. Keep only the closest evidence
            # and ties that are genuinely very close. This prevents a dense bus
            # from creating dozens of false component connections.
            chosen = []
            if candidates:
                best_d = candidates[0][0]
                for d, wc, method in candidates:
                    if d <= best_d + 2.0 and d <= COMPONENT_WIRE_TOL:
                        chosen.append((d, wc, method))
            if not chosen:
                self.unresolved.append({
                    "page": comp.page, "component": comp.ref,
                    "x": comp.x, "y": comp.y,
                    "reason": "no nearby topologically traced wire"
                })
                continue

            for d, wc, method in chosen:
                names = sorted(self.cluster_labels.get((comp.page, wc.cluster_id), set()))
                if names:
                    # A physical wire normally has one logical net name.
                    # PDFs often repeat the same name many times and may also
                    # place cross-sheet annotations beside it. Do not explode
                    # one component into every nearby text token. Choose one
                    # deterministic canonical name, while the raw labels remain
                    # available in net_labels.csv/JSON for audit.
                    non_power = [n for n in names if n.upper() not in POWER_WORDS]
                    pool = non_power or names
                    name = max(pool, key=lambda n: (len(n), n))
                    self.component_nets.append({
                        "component": comp.ref, "page": comp.page,
                        "net": name, "cluster": wc.cluster_id,
                        "distance": round(d, 3), "evidence": method,
                    })
                else:
                    self.component_nets.append({
                        "component": comp.ref, "page": comp.page,
                        "net": f"_ANON_P{comp.page}_{wc.cluster_id}",
                        "cluster": wc.cluster_id,
                        "distance": round(d, 3), "evidence": method,
                    })

    def _build_graph(self):
        g = nx.Graph()
        for comp in self.components:
            g.add_node(f"C:{comp.ref}@p{comp.page}", kind="component", ref=comp.ref, page=comp.page)

        # Map cluster aliases to stable logical net IDs.
        for rec in self.component_nets:
            net = rec["net"]
            node = f"N:{net}"
            g.add_node(node, kind="net", name=net)
            comp_node = f"C:{rec['component']}@p{rec['page']}"
            g.add_edge(comp_node, node,
                       page=rec["page"], cluster=rec["cluster"],
                       distance=rec["distance"], evidence=rec["evidence"])
        self.graph = g

    def component_edges(self):
        by_net = defaultdict(list)
        for rec in self.component_nets:
            by_net[rec["net"]].append(rec)
        rows = []
        seen = set()
        for net, rows0 in by_net.items():
            comps = defaultdict(list)
            for r in rows0:
                comps[r["component"]].append(r["cluster"])
            refs = sorted(comps)
            for i, a in enumerate(refs):
                for b in refs[i+1:]:
                    key = (a, b, net)
                    if key in seen: continue
                    seen.add(key)
                    rows.append({"source": a, "target": b, "net": net,
                                 "source_clusters": ",".join(map(str, sorted(set(comps[a])))),
                                 "target_clusters": ",".join(map(str, sorted(set(comps[b]))))})
        return rows

    def summary(self):
        return {
            "pages": len(self.doc),
            "text_items": sum(len(v) for v in self.text.values()),
            "vector_line_candidates": sum(len(v) for v in self.segments.values()),
            "wire_clusters": sum(len(v) for v in self.clusters.values()),
            "components": len(self.components),
            "unique_net_labels": len(set(n.name for n in self.net_labels)),
            "net_label_occurrences": len(self.net_labels),
            "component_net_records": len(self.component_nets),
            "components_with_connections": len(set(r["component"] for r in self.component_nets)),
            "unresolved_components": len(self.unresolved),
            "component_edges": len(self.component_edges()),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
        }

    # ------------------------------ exports ------------------------------
    def export_json(self, path):
        payload = {
            "source": self.pdf_path,
            "summary": self.summary(),
            "components": [c.__dict__ for c in self.components],
            "net_labels": [n.__dict__ for n in self.net_labels],
            "component_nets": self.component_nets,
            "component_edges": self.component_edges(),
            "unresolved_components": self.unresolved,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_graphml(self, path):
        # GraphML requires scalar attributes; remove/convert any non-scalars.
        g = nx.Graph()
        for n, attrs in self.graph.nodes(data=True):
            g.add_node(n, **{k: str(v) for k, v in attrs.items()})
        for a, b, attrs in self.graph.edges(data=True):
            g.add_edge(a, b, **{k: str(v) for k, v in attrs.items()})
        nx.write_graphml(g, path)

    def export_csvs(self, prefix):
        prefix = Path(prefix)
        with open(str(prefix) + "_component_nets.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["component", "page", "net", "cluster", "distance", "evidence"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.component_nets)

        with open(str(prefix) + "_component_edges.csv", "w", newline="", encoding="utf-8") as f:
            rows = self.component_edges(); fields = ["source", "target", "net", "source_clusters", "target_clusters"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

        with open(str(prefix) + "_unresolved_components.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "component", "x", "y", "reason"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.unresolved)

        with open(str(prefix) + "_net_labels.csv", "w", newline="", encoding="utf-8") as f:
            rows = []
            for n in self.net_labels:
                clusters = sorted(self.cluster_labels.get((n.page, self._label_cluster(n)), set()))
                rows.append({"page": n.page, "name": n.name, "x": n.x, "y": n.y,
                             "clusters": ",".join(clusters)})
            fields = ["page", "name", "x", "y", "clusters"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    def _label_cluster(self, label):
        wc, _ = self._nearest_cluster_fast(label.page, label.x, label.y, LABEL_WIRE_TOL)
        return wc.cluster_id if wc else -1

    def export_summary(self, path):
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def export_overview(self, path):
        # Simple logical graph overview; deliberately not a schematic-layout drawing.
        import matplotlib.pyplot as plt
        g = self.graph
        if g.number_of_nodes() == 0:
            return
        pos = nx.spring_layout(g, seed=42, k=0.7 / math.sqrt(max(1, g.number_of_nodes())))
        plt.figure(figsize=(16, 12))
        comp_nodes = [n for n, a in g.nodes(data=True) if a.get("kind") == "component"]
        net_nodes = [n for n, a in g.nodes(data=True) if a.get("kind") == "net"]
        nx.draw_networkx_nodes(g, pos, nodelist=comp_nodes, node_size=60)
        nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, node_size=25)
        nx.draw_networkx_edges(g, pos, width=0.4, alpha=0.35)
        labels = {n: g.nodes[n].get("ref", g.nodes[n].get("name", n).replace("N:", "")) for n in comp_nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=5)
        plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="input schematic PDF")
    ap.add_argument("--out-prefix", default="schematic_v4")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    ex = Extractor(args.pdf).parse(verbose=True)
    ex.export_json(str(out) + ".json")
    ex.export_graphml(str(out) + ".graphml")
    ex.export_csvs(out)
    ex.export_summary(str(out) + "_summary.json")
    if args.viz:
        ex.export_overview(str(out) + "_overview.png")
    print("\nExtraction complete")
    print(json.dumps(ex.summary(), indent=2))


if __name__ == "__main__":
    main()
