"""
schematic_graph.py
===================

Reconstruct a full connectivity netlist (components <-> pins <-> nets)
from a *vector-drawn* schematic PDF, using the same visual conventions
an engineer reading the schematic would use. Generalized to work on any
schematic export (Altium, KiCad, OrCAD, Eagle, ...) that draws wires as
real vector line segments - no tool-specific metadata required.

THE ELECTRICAL RULES THIS ENCODES
----------------------------------
A schematic is a *drawing*; a netlist is the *truth* underneath it. The
rules a human uses to go from one to the other are:

1. Two wire segments that touch **end-to-end** (a corner / L-bend) are
   the same wire.
2. A wire segment whose end lands on the **interior** of another segment
   (a "T") is electrically joined to it - this is how most tools draw a
   branch/tap without a junction dot.
3. Two wires that visually **cross but neither ends there** are *NOT*
   connected, UNLESS a filled junction dot is drawn at the crossing -
   the dot is what tells you "these four arms are one node", otherwise
   it's just one wire jumping over another on paper.
4. A **net name label** placed on a wire stub names that wire. Two wire
   runs anywhere in the document (even on different sheets) that carry
   the same net name are the same electrical net - multi-sheet
   schematics rely on this instead of redrawing the wire across pages.
5. A component is connected to whatever net its pins/symbol body touch.

This module implements exactly these five rules geometrically:
  - endpoint-to-endpoint joining (rule 1)
  - endpoint-to-segment-interior joining (rule 2)
  - junction-dot-triggered joining at true crossings (rule 3)
  - net-name-label -> wire-cluster attachment, merged by name across the
    whole document (rule 4)
  - component-body / refdes-label -> wire-cluster proximity linking,
    annotated with the nearest pin-number label when one is found
    (rule 5)

WHAT IT DOES NOT DO
--------------------
It cannot recover a manufacturer's internal pin-to-symbol-pad mapping
beyond what's printed on the sheet - if the sheet prints it (most do),
this module captures it as a pin label on the connecting edge. It also
can't perfectly separate two different components whose symbols happen
to overlap in the drawing; this is a best-effort geometric
reconstruction, not a verified netlist.

Some EDA tools (e.g. Altium) additionally embed an invisible,
click-to-highlight netlist as overlapping text runs stacked at the same
coordinates. That data is real but not reliably recoverable through
ordinary text extraction (overlapping runs interleave character-by-
character) and isn't present in exports from other tools, so this
module doesn't depend on it - everything here is derived from the
geometry any vector schematic PDF contains.

USAGE
-----
    from schematic_graph import SchematicGraphBuilder

    builder = SchematicGraphBuilder("schematic.pdf")
    builder.parse()
    graph = builder.build_graph()

    builder.export_graphml("out.graphml")
    builder.export_json("out.json")
    builder.export_component_net_table("out_connections.csv")
    builder.summary()

    builder.net_of("U36")                       # nets a component touches
    builder.components_on_net("RESET_S32K")      # who's on a given net
    builder.path_between("R1", "C4")             # connective path via nets
"""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
import networkx as nx


# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------

# Endpoint-to-endpoint / endpoint-to-interior "touching" tolerance (pt).
WIRE_JOIN_TOLERANCE = 1.2

# How far off a segment's *interior* an endpoint may land and still count
# as a T-junction tap (pt). Kept tight - this is not a crossing test.
T_JUNCTION_TOLERANCE = 1.0

# A junction dot's centre must be within this distance of a segment's
# line (its interior, not just near an endpoint) to force-join it at a
# "+" crossing (pt).
DOT_JOIN_TOLERANCE = 2.0

# Filled curves this small (both dimensions) are treated as junction
# dots rather than large filled shapes/graphics (pt).
JUNCTION_DOT_MAX_DIAMETER = 8.0

# A text label attaches to the nearest wire cluster if within this
# distance of the cluster's nearest point (pt).
LABEL_TO_WIRE_TOLERANCE = 12.0

# Component-body-rect -> wire-cluster attach distance (pt). Larger than
# the label tolerance since pins stick out from all sides of a chip.
BODY_TO_WIRE_TOLERANCE = 8.0

# Fallback: refdes-label-point -> wire-cluster attach distance (pt),
# used only when no component body rectangle could be associated (this
# is the common case for 2/3-terminal parts like R/C/D/Q, which are
# drawn as bare glyph strokes rather than a filled body rect). Kept
# tight: refdes labels sit right next to their symbol, and a loose
# radius on a densely packed sheet will "see" several unrelated
# neighbouring nets that just happen to pass nearby on paper without
# actually touching this component's leads.
LABEL_TO_COMPONENT_TOLERANCE = 16.0

# A candidate component-body rectangle: non-hairline (both sides at
# least this large) real geometry, e.g. an IC outline.
MIN_BODY_RECT_SIDE = 4.0

# A refdes label is assigned to a body rect if within this distance of
# the rect's boundary (pt) - refdes labels usually sit just above/beside
# the symbol body. Kept tight and paired with a one-to-one nearest-match
# assignment (see _assign_refdes_to_bodies): a small component must not
# be able to "steal" a large neighbouring IC's body just because it's
# the closest *rectangle* within a loose radius - each rect goes to its
# single nearest refdes label, and vice versa.
REFDES_TO_BODY_TOLERANCE = 22.0

# Ignore vector lines / rect "hairlines" longer than this fraction of
# the page's larger dimension - almost certainly sheet border, title
# block rules, or table gridlines rather than schematic wires.
BORDER_LINE_FRACTION = 0.85

# Many PDF renderers draw small round pin/pad markers as *zero-length*
# lines with a round cap (a cheap way to paint a dot). These have no
# real extent, carry no wire direction, and - critically - if fed into
# the tracer as ordinary segments they can chain unrelated wires
# together through pure geometric coincidence (a stray dot landing
# within tolerance of some other net's wire). Segments shorter than
# this are treated as drawing artifacts, not wires, and dropped before
# tracing. Genuine wire stubs in schematics are essentially never this
# short (pin spacing is normally >= a couple of points at minimum).
MIN_WIRE_SEGMENT_LENGTH = 1.5

# Schematic *wires* are drawn strictly horizontal or vertical in nearly
# every tool's default style. Component *symbol glyphs* - a resistor's
# zigzag/hatching, a diode's arrowhead, a transistor's angled leads -
# are drawn with diagonal strokes. If diagonal glyph strokes are fed
# into the wire tracer, a component's own decorative body art can touch
# both of its lead stubs and short its two pins together in the
# extracted graph, which then chain-fuses unrelated neighbouring
# subcircuits into one false "net". Restricting wire-tracing to
# near-axis-aligned segments (within this angle of horizontal or
# vertical, in degrees) excludes glyph art while keeping real wires.
# Raise this if a schematic style routes wires diagonally.
MAX_WIRE_ANGLE_DEVIATION_DEG = 3.0

# Reference-designator prefixes recognised out of the box. Extend this
# set for schematics that use other conventions.
REFDES_PREFIXES = (
    "IC", "R", "C", "L", "D", "Q", "U", "Y", "J", "SW", "TP", "TPV", "TS",
    "FB", "K", "X", "LED", "F", "B", "T", "W", "P", "S", "M", "BAT", "OSC",
)

REFDES_RE = re.compile(
    r"^(?:" + "|".join(sorted(REFDES_PREFIXES, key=len, reverse=True)) + r")\d+[A-Z]?$"
)

# Net / signal name heuristic: mostly-uppercase token with letters,
# digits, underscores or a leading slash (active-low), length >= 2.
NET_NAME_RE = re.compile(r"^/?[A-Z][A-Z0-9_]{1,40}$")

# A short, purely numeric (or numeric+1 letter, e.g. BGA "N3") token in
# small font near a wire end is very likely a *pin number*, not a net
# name or refdes.
PIN_NUMBER_RE = re.compile(r"^[A-Z]?\d{1,3}[A-Z]?$")

PAGE_REF_RE = re.compile(r"^Page\[?([\d,\s]+)\]?$", re.IGNORECASE)

# Some EDA tools (notably Altium) additionally stamp an invisible,
# click-to-highlight netlist as overlapping text runs at identical
# coordinates: "PI<refdes><pin>" (pin identifier), "CO<refdes>"
# (component locator), "NL<net>" (net locator, with "_" encoded as a
# literal "0"). This data is real but not reliably separable through
# ordinary text extraction (the overlapping runs interleave character-
# by-character), so rather than half-trust it, we detect and exclude it
# from the ordinary label classifier entirely - otherwise thousands of
# these tokens masquerade as plausible net names / pin numbers and
# pollute the geometrically-derived graph. The pattern: prefix followed
# immediately by a letter block then a digit block (refdes+pin shape),
# no spaces - ordinary schematic text essentially never looks like this.
EMBEDDED_METADATA_RE = re.compile(r"^(?:PI|CO|NL)[A-Z]{1,4}\d[A-Z0-9]*$")

NET_NAME_STOPWORDS = {
    "REV", "DNP", "TITLE", "NOTES", "TOC", "BOM", "ASSY", "OPT", "REF",
    "DES", "PAGE", "SHEET", "SIZE", "DATE", "DRAWN", "APPROVED", "GND",
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class TextLabel:
    text: str
    page: int
    x0: float
    x1: float
    top: float
    bottom: float
    font_size: float = 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class WireCluster:
    """A set of touching/joined line segments = one electrical wire."""
    page: int
    cluster_id: int
    points: list = field(default_factory=list)   # all segment endpoints
    net_names: set = field(default_factory=set)   # net-name labels attached
    pin_labels: list = field(default_factory=list)  # (x, y, text) small labels attached

    def min_dist_to_point(self, x: float, y: float) -> float:
        return min(math.hypot(px - x, py - y) for px, py in self.points)

    def nearest_pin_label(self, x: float, y: float, max_dist: float):
        best, best_d = None, max_dist
        for px, py, text in self.pin_labels:
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best, best_d = text, d
        return best


@dataclass
class BodyRect:
    page: int
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2

    def dist_to_point(self, x: float, y: float) -> float:
        dx = max(self.x0 - x, 0, x - self.x1)
        dy = max(self.top - y, 0, y - self.bottom)
        return math.hypot(dx, dy)


@dataclass
class Component:
    refdes: str
    page: int
    x: float
    y: float
    body: Optional[BodyRect] = None


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _is_axis_aligned(p1, p2, max_deg: float) -> bool:
    """True if the segment is within max_deg of horizontal or vertical."""
    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
    if dx == 0 or dy == 0:
        return True
    angle = math.degrees(math.atan2(min(dx, dy), max(dx, dy)))
    return angle <= max_deg


def _point_segment_distance(px, py, x1, y1, x2, y2) -> float:
    """Perpendicular distance from (px,py) to the segment (x1,y1)-(x2,y2),
    clamped to the segment (not the infinite line)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


class SpatialGrid:
    """Simple uniform grid for fast 'what's near this point' queries."""

    def __init__(self, cell_size: float):
        self.cell = cell_size
        self.buckets: dict[tuple[int, int], list[int]] = {}

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell), int(y // self.cell))

    def insert_point(self, idx: int, x: float, y: float) -> None:
        self.buckets.setdefault(self._cell_of(x, y), []).append(idx)

    def insert_segment(self, idx: int, p1, p2) -> None:
        # Register the segment's index in every grid cell its bounding
        # box overlaps, so point/segment queries near any part of it
        # find it. Segments are pre-filtered to be reasonably short
        # (border/long bus lines are dropped upstream).
        x0, x1 = sorted((p1[0], p2[0]))
        y0, y1 = sorted((p1[1], p2[1]))
        cx0, cx1 = int(x0 // self.cell), int(x1 // self.cell)
        cy0, cy1 = int(y0 // self.cell), int(y1 // self.cell)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                self.buckets.setdefault((cx, cy), []).append(idx)

    def nearby(self, x: float, y: float, radius_cells: int = 1):
        cx, cy = self._cell_of(x, y)
        seen = set()
        out = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for i in self.buckets.get((cx + dx, cy + dy), []):
                    if i not in seen:
                        seen.add(i)
                        out.append(i)
        return out


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------

class SchematicGraphBuilder:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.page_sizes: dict[int, tuple[float, float]] = {}
        self.labels: list[TextLabel] = []
        self.wire_clusters: dict[int, list[WireCluster]] = {}
        self.components: list[Component] = []
        self.net_label_occurrences: dict[str, list[TextLabel]] = {}
        self.pin_label_pool: dict[int, list[TextLabel]] = {}   # page -> small labels
        self.body_rects: dict[int, list[BodyRect]] = {}
        self.graph: Optional[nx.Graph] = None

    # ---------------------------------------------------------- parsing --

    def parse(self, verbose: bool = True) -> None:
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                self.page_sizes[page_idx] = (page.width, page.height)
                self._parse_page_text(page, page_idx)
                self._parse_page_geometry(page, page_idx)
        self._classify_labels()
        self._assign_refdes_to_bodies()
        if verbose:
            n_wc = sum(len(v) for v in self.wire_clusters.values())
            print(
                f"Parsed {len(self.page_sizes)} page(s): "
                f"{len(self.labels)} text labels, {n_wc} wire clusters, "
                f"{len(self.components)} components "
                f"({sum(1 for c in self.components if c.body)} with a detected body)."
            )

    def _parse_page_text(self, page, page_idx: int) -> None:
        words = page.extract_words(
            use_text_flow=False, keep_blank_chars=False, extra_attrs=["size"]
        )
        for w in words:
            text = w["text"].strip()
            if not text:
                continue
            self.labels.append(
                TextLabel(
                    text=text, page=page_idx,
                    x0=w["x0"], x1=w["x1"], top=w["top"], bottom=w["bottom"],
                    font_size=w.get("size", 0.0),
                )
            )

    def _parse_page_geometry(self, page, page_idx: int) -> None:
        pw, ph = page.width, page.height
        max_len = BORDER_LINE_FRACTION * max(pw, ph)

        segments = []  # (p1, p2)
        for ln in page.lines:
            p1 = (ln["x0"], ln["top"])
            p2 = (ln["x1"], ln["bottom"])
            length = _dist(p1, p2)
            if length < MIN_WIRE_SEGMENT_LENGTH or length > max_len:
                continue
            if not _is_axis_aligned(p1, p2, MAX_WIRE_ANGLE_DEVIATION_DEG):
                continue  # component-symbol glyph art (hatching, arrows, etc.)
            segments.append((p1, p2))

        body_rects = []
        for r in page.rects:
            w_, h_ = r["x1"] - r["x0"], r["bottom"] - r["top"]
            short, long_ = min(w_, h_), max(w_, h_)
            if short <= 1.0:
                # Hairline rect: used as a wire/pin stub by many tools.
                # Require real extent too, for the same reason zero-length
                # lines are dropped above.
                if long_ < MIN_WIRE_SEGMENT_LENGTH or long_ > max_len:
                    continue
                segments.append(((r["x0"], r["top"]), (r["x1"], r["bottom"])))
            elif short >= MIN_BODY_RECT_SIDE and long_ <= max_len:
                # A "real" box: likely a component symbol body / IC outline.
                body_rects.append(BodyRect(page_idx, r["x0"], r["x1"], r["top"], r["bottom"]))
        self.body_rects[page_idx] = body_rects

        junction_dots = [
            ((c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2)
            for c in page.curves
            if c.get("fill")
            and (c["x1"] - c["x0"]) <= JUNCTION_DOT_MAX_DIAMETER
            and (c["bottom"] - c["top"]) <= JUNCTION_DOT_MAX_DIAMETER
        ]

        self.wire_clusters[page_idx] = self._trace_wires(segments, junction_dots)

    def _trace_wires(self, segments, junction_dots) -> list[WireCluster]:
        n = len(segments)
        uf = UnionFind(n)
        if n == 0:
            return []

        cell = max(WIRE_JOIN_TOLERANCE * 6, 6.0)
        grid = SpatialGrid(cell)
        for i, (p1, p2) in enumerate(segments):
            grid.insert_segment(i, p1, p2)

        # --- Rule 1 + Rule 2: endpoint-endpoint and endpoint-interior joins.
        for i, (p1, p2) in enumerate(segments):
            for pt in (p1, p2):
                for j in grid.nearby(pt[0], pt[1]):
                    if j == i:
                        continue
                    q1, q2 = segments[j]
                    # Endpoint-to-endpoint (corner).
                    if _dist(pt, q1) <= WIRE_JOIN_TOLERANCE or _dist(pt, q2) <= WIRE_JOIN_TOLERANCE:
                        uf.union(i, j)
                        continue
                    # Endpoint-to-interior (T-junction tap).
                    d = _point_segment_distance(pt[0], pt[1], q1[0], q1[1], q2[0], q2[1])
                    if d <= T_JUNCTION_TOLERANCE:
                        uf.union(i, j)

        # --- Rule 3: junction-dot-forced joins at true "+" crossings.
        for dx, dy in junction_dots:
            touching = []
            for j in grid.nearby(dx, dy):
                q1, q2 = segments[j]
                d = _point_segment_distance(dx, dy, q1[0], q1[1], q2[0], q2[1])
                if d <= DOT_JOIN_TOLERANCE:
                    touching.append(j)
            for b in touching[1:]:
                uf.union(touching[0], b)

        clusters_by_root: dict[int, WireCluster] = {}
        for i, (p1, p2) in enumerate(segments):
            root = uf.find(i)
            wc = clusters_by_root.get(root)
            if wc is None:
                wc = WireCluster(page=0, cluster_id=root)
                clusters_by_root[root] = wc
            wc.points.append(p1)
            wc.points.append(p2)

        clusters = list(clusters_by_root.values())
        for idx, wc in enumerate(clusters):
            wc.cluster_id = idx
        return clusters

    # ------------------------------------------------------ classification --

    def _classify_labels(self) -> None:
        font_sizes = [l.font_size for l in self.labels if l.font_size > 0]
        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 8.0

        for lbl in self.labels:
            text = lbl.text
            if EMBEDDED_METADATA_RE.match(text):
                continue  # tool-internal hidden annotation, not schematic text
            if REFDES_RE.match(text):
                self.components.append(Component(refdes=text, page=lbl.page, x=lbl.cx, y=lbl.cy))
                continue
            if text.upper() in NET_NAME_STOPWORDS:
                continue
            if PIN_NUMBER_RE.match(text) and lbl.font_size and lbl.font_size <= median_size * 0.85:
                # Small numeric label -> very likely a pin number sitting
                # right where a wire meets a symbol; keep it separately
                # so we can annotate component-net edges with it.
                self.pin_label_pool.setdefault(lbl.page, []).append(lbl)
                continue
            if NET_NAME_RE.match(text) and not text.isdigit():
                self.net_label_occurrences.setdefault(text, []).append(lbl)

    def _assign_refdes_to_bodies(self) -> None:
        """One-to-one nearest matching between refdes labels and body
        rects, per page. A loose "any rect within N points" match would
        let a small component (e.g. a crystal) latch onto a large
        neighbouring IC's body just because that IC's huge perimeter
        happens to pass within range from some direction - which then
        silently inherits every one of that IC's pin connections. Greedy
        nearest-pair assignment prevents that: each rect can be claimed
        by at most one refdes, and only if it's mutually the closest
        candidate on both sides.
        """
        by_page: dict[int, list[Component]] = {}
        for comp in self.components:
            by_page.setdefault(comp.page, []).append(comp)

        for page_idx, comps in by_page.items():
            rects = self.body_rects.get(page_idx, [])
            if not rects:
                continue
            pairs = []
            for ci, comp in enumerate(comps):
                for ri, rect in enumerate(rects):
                    d = rect.dist_to_point(comp.x, comp.y)
                    if d <= REFDES_TO_BODY_TOLERANCE:
                        pairs.append((d, ci, ri))
            pairs.sort(key=lambda t: t[0])
            used_comp, used_rect = set(), set()
            for d, ci, ri in pairs:
                if ci in used_comp or ri in used_rect:
                    continue
                comps[ci].body = rects[ri]
                used_comp.add(ci)
                used_rect.add(ri)

    # -------------------------------------------------------- graph build --

    def build_graph(self, verbose: bool = True) -> nx.Graph:
        g = nx.Graph()
        net_node = lambda name: f"NET:{name}"

        # 1) Snap net-name labels onto their nearest wire cluster.
        for net_name, occurrences in self.net_label_occurrences.items():
            for lbl in occurrences:
                clusters = self.wire_clusters.get(lbl.page, [])
                best, best_d = None, LABEL_TO_WIRE_TOLERANCE
                for wc in clusters:
                    d = wc.min_dist_to_point(lbl.cx, lbl.cy)
                    if d < best_d:
                        best, best_d = wc, d
                if best is not None:
                    best.net_names.add(net_name)

        # 2) Register pin-number labels onto their nearest wire cluster
        #    too, so we can report "which pin" a component-net edge uses.
        for page_idx, pins in self.pin_label_pool.items():
            clusters = self.wire_clusters.get(page_idx, [])
            for lbl in pins:
                best, best_d = None, LABEL_TO_WIRE_TOLERANCE
                for wc in clusters:
                    d = wc.min_dist_to_point(lbl.cx, lbl.cy)
                    if d < best_d:
                        best, best_d = wc, d
                if best is not None:
                    best.pin_labels.append((lbl.cx, lbl.cy, lbl.text))

        # 3) Create net nodes: named nets are merged by name across the
        #    whole document (this is what stitches multi-sheet nets
        #    together); unnamed wire clusters get their own anonymous node
        #    so direct, unlabeled component-to-component wiring isn't lost.
        for net_name in self.net_label_occurrences:
            pages = sorted({o.page for o in self.net_label_occurrences[net_name]})
            g.add_node(net_node(net_name), kind="net", label=net_name, pages=pages)

        cluster_to_netnode: dict[tuple[int, int], list[str]] = {}
        anon_counter = 0
        for page_idx, clusters in self.wire_clusters.items():
            for wc in clusters:
                if wc.net_names:
                    cluster_to_netnode[(page_idx, wc.cluster_id)] = [
                        net_node(n) for n in wc.net_names
                    ]
                else:
                    anon_counter += 1
                    node_id = f"NET:_anon_p{page_idx}_{anon_counter}"
                    g.add_node(node_id, kind="net", label=None, pages=[page_idx])
                    cluster_to_netnode[(page_idx, wc.cluster_id)] = [node_id]

        # 4) Component nodes.
        for comp in self.components:
            node_id = f"U:{comp.refdes}@p{comp.page}"
            if node_id not in g:
                g.add_node(node_id, kind="component", refdes=comp.refdes,
                           page=comp.page, x=comp.x, y=comp.y,
                           has_body=comp.body is not None)

        # 5) Link components to nets: prefer distance to the detected
        #    component BODY (correct for multi-pin ICs whose pins are
        #    far from the refdes text label); fall back to the refdes
        #    label point for symbols with no detected body (R/C/D/... are
        #    usually drawn as bare lines, not a filled rect).
        for comp in self.components:
            node_id = f"U:{comp.refdes}@p{comp.page}"
            clusters = self.wire_clusters.get(comp.page, [])
            tolerance = BODY_TO_WIRE_TOLERANCE if comp.body else LABEL_TO_COMPONENT_TOLERANCE
            for wc in clusters:
                if not wc.points:
                    continue
                if comp.body:
                    d = min(comp.body.dist_to_point(px, py) for px, py in wc.points)
                    d = min(d, wc.min_dist_to_point(comp.body.cx, comp.body.cy))
                else:
                    d = wc.min_dist_to_point(comp.x, comp.y)
                if d <= tolerance:
                    anchor_x = comp.body.cx if comp.body else comp.x
                    anchor_y = comp.body.cy if comp.body else comp.y
                    pin = wc.nearest_pin_label(anchor_x, anchor_y, max_dist=25.0)
                    for net_id in cluster_to_netnode.get((comp.page, wc.cluster_id), []):
                        if g.has_edge(node_id, net_id):
                            continue
                        g.add_edge(node_id, net_id, pin=pin)

        self.graph = g
        if verbose:
            n_comp = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "component")
            n_net = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "net")
            print(f"Graph built: {n_comp} component nodes, {n_net} net nodes, "
                  f"{g.number_of_edges()} edges.")
        return g

    # ---------------------------------------------------------- exports --

    def export_graphml(self, path: str) -> None:
        assert self.graph is not None, "call build_graph() first"
        g = self.graph.copy()
        for _, data in g.nodes(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (list, set)):
                    data[k] = ",".join(str(x) for x in v)
                elif v is None:
                    data[k] = ""
        for _, _, data in g.edges(data=True):
            if data.get("pin") is None:
                data["pin"] = ""
        nx.write_graphml(g, path)

    def export_json(self, path: str) -> None:
        assert self.graph is not None, "call build_graph() first"
        data = nx.node_link_data(self.graph, edges="edges")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_component_net_table(self, path: str) -> None:
        """Flat CSV: refdes, page, pin, net_id, net_label - one row per pin."""
        assert self.graph is not None, "call build_graph() first"
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["refdes", "page", "pin", "net_id", "net_label"])
            for node_id, data in self.graph.nodes(data=True):
                if data.get("kind") != "component":
                    continue
                for neighbor in self.graph.neighbors(node_id):
                    ndata = self.graph.nodes[neighbor]
                    edata = self.graph.edges[node_id, neighbor]
                    writer.writerow([
                        data["refdes"], data["page"], edata.get("pin") or "",
                        neighbor, ndata.get("label") or "(unnamed wire)",
                    ])

    def export_visualization(self, path: str, max_nodes: int = 400) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assert self.graph is not None, "call build_graph() first"
        g = self.graph
        if g.number_of_nodes() > max_nodes:
            top_nodes = sorted(g.nodes, key=lambda n: -g.degree(n))[:max_nodes]
            g = g.subgraph(top_nodes)

        pos = nx.spring_layout(g, seed=42, k=0.6)
        comp_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "component"]
        net_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "net"]

        plt.figure(figsize=(16, 16))
        nx.draw_networkx_edges(g, pos, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(g, pos, nodelist=comp_nodes, node_color="#3b82f6",
                                node_size=90, label="Component")
        nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, node_color="#f59e0b",
                                node_size=35, label="Net")
        nx.draw_networkx_labels(
            g, pos, labels={n: g.nodes[n]["refdes"] for n in comp_nodes}, font_size=6
        )
        plt.legend(scatterpoints=1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    # ---------------------------------------------------------- queries --

    def net_of(self, refdes: str):
        assert self.graph is not None
        out = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("kind") == "component" and data.get("refdes") == refdes:
                out.extend(self.graph.neighbors(node_id))
        return out

    def components_on_net(self, net_name: str):
        assert self.graph is not None
        node_id = f"NET:{net_name}"
        if node_id not in self.graph:
            return []
        return [n for n in self.graph.neighbors(node_id)
                if self.graph.nodes[n].get("kind") == "component"]

    def path_between(self, refdes_a: str, refdes_b: str):
        assert self.graph is not None
        a_nodes = [n for n, d in self.graph.nodes(data=True)
                   if d.get("kind") == "component" and d.get("refdes") == refdes_a]
        b_nodes = [n for n, d in self.graph.nodes(data=True)
                   if d.get("kind") == "component" and d.get("refdes") == refdes_b]
        best = None
        for a in a_nodes:
            for b in b_nodes:
                try:
                    p = nx.shortest_path(self.graph, a, b)
                except nx.NetworkXNoPath:
                    continue
                if best is None or len(p) < len(best):
                    best = p
        return best

    def summary(self) -> None:
        assert self.graph is not None
        g = self.graph
        n_comp = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "component")
        n_net_named = sum(1 for _, d in g.nodes(data=True)
                           if d.get("kind") == "net" and d.get("label"))
        n_net_anon = sum(1 for _, d in g.nodes(data=True)
                          if d.get("kind") == "net" and not d.get("label"))
        n_with_pin = sum(1 for _, _, d in g.edges(data=True) if d.get("pin"))
        comps_by_conn_count = sorted(
            ((d["refdes"], g.degree(n)) for n, d in g.nodes(data=True)
             if d.get("kind") == "component"),
            key=lambda t: -t[1],
        )
        print("=" * 60)
        print(f"Components:          {n_comp}")
        print(f"Named nets:          {n_net_named}")
        print(f"Unnamed wires:       {n_net_anon}")
        print(f"Edges (connections): {g.number_of_edges()}")
        print(f"  of which pin-labeled: {n_with_pin}")
        print(f"Connected components (isolated excluded): "
              f"{sum(1 for _, c in comps_by_conn_count if c > 0)}")
        print("Most-connected components:")
        for refdes, deg in comps_by_conn_count[:10]:
            print(f"  {refdes:12s} degree={deg}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="Path to schematic PDF")
    ap.add_argument("--out-prefix", default="schematic", help="Output file prefix")
    ap.add_argument("--viz", action="store_true", help="Also render a PNG overview")
    args = ap.parse_args()

    builder = SchematicGraphBuilder(args.pdf)
    builder.parse()
    builder.build_graph()
    builder.summary()
    builder.export_graphml(f"{args.out_prefix}.graphml")
    builder.export_json(f"{args.out_prefix}.json")
    builder.export_component_net_table(f"{args.out_prefix}_connections.csv")
    if args.viz:
        builder.export_visualization(f"{args.out_prefix}_overview.png")
    print(f"Wrote {args.out_prefix}.graphml, {args.out_prefix}.json, "
          f"{args.out_prefix}_connections.csv")
