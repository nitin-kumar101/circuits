#!/usr/bin/env python3
"""
schematic_graph_v8_generic.py

Generic, fast multi-layout schematic extractor for vector PDF schematics: dense SITE/BGA pin maps, conventional circuits, power rails and repeated passive banks.

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
<prefix>_pin_records.csv
<prefix>_pin_tables.csv
<prefix>_external_links.csv
<prefix>_page_text.csv
optional: <prefix>_overview.png

Usage
-----
python schematic_graph_v8_generic.py input.pdf --out-prefix result --viz

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

# Dense IC/BGA/pin-table extraction. These pages often contain rows such as:
#   113_15  2_CCM_CLK1_N  P13  CCM_CLK1_N
# or the mirrored order:
#   CCM_CLK1_N  P13  2_CCM_CLK1_N  113_15
# We treat these as structured pin records rather than ordinary text labels.
PIN_ROW_Y_TOL = 3.2
PIN_COLUMN_TOL = 5.0
MIN_PIN_TABLE_ROWS = 8
PIN_COORD_RE = re.compile(r"^[A-Z]{1,3}\d{1,4}$", re.I)
EXTERNAL_ID_RE = re.compile(r"^\d{1,5}[_-]\d{1,5}$")
PIN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_#./+\-]*$")

# Common reference designators. The parser is deliberately conservative:
# arbitrary capital words are not components.
REF_PREFIXES = (
    "LED", "BAT", "OSC", "CON", "CN", "FB", "TP", "SW", "IC",
    "R", "C", "L", "D", "Q", "U", "Y", "J", "K", "X", "F",
    "B", "T", "W", "P", "S", "M",
)
REF_RE = re.compile(
    r"^(?:(?:[A-Z][A-Z0-9]*_)+)?(?:"
    + "|".join(sorted(REF_PREFIXES, key=len, reverse=True))
    + r")\d+(?:[A-Z]|_[A-Z0-9]+)?$", re.I
)

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
class PinRecord:
    page: int
    block: str
    pin_name: str
    pin_number: str
    net: str = ""
    external_id: str = ""
    side: str = "unknown"
    row_y: float = 0.0
    x: float = 0.0
    evidence: str = "row-aligned"


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
        self.pin_records: list[PinRecord] = []
        self.pin_tables: list[dict] = []
        self.external_links: list[dict] = []
        self.signal_rows: list[dict] = []
        self.page_profiles: list[dict] = []
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
        self._extract_pin_tables()
        self._extract_signal_rows()
        self._profile_pages()
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

    # --------------------- structured pin-table extraction ---------------------
    def _row_groups(self, words):
        """Group PDF text items into horizontal rows using baseline proximity."""
        rows = []
        for w in sorted(words, key=lambda z: (z.cy, z.x0)):
            placed = False
            for row in rows[-3:]:
                if abs(w.cy - row[0].cy) <= PIN_ROW_Y_TOL:
                    row.append(w)
                    placed = True
                    break
            if not placed:
                rows.append([w])
        for row in rows:
            row.sort(key=lambda z: z.x0)
        return rows

    @staticmethod
    def _is_pin_coord(text):
        return bool(PIN_COORD_RE.fullmatch(text.strip()))

    @staticmethod
    def _is_external_id(text):
        return bool(EXTERNAL_ID_RE.fullmatch(text.strip()))

    @staticmethod
    def _pin_name_candidate(text):
        t = text.strip()
        if not t or len(t) > 80:
            return False
        if Extractor._is_pin_coord(t) or Extractor._is_external_id(t):
            return False
        if parse_ref(t):
            return False
        if t.upper() in {"SITE1", "SITE2", "SITE3", "SITE4", "GND", "POWER", "INPUT", "OUTPUT"}:
            return False
        if not PIN_NAME_RE.fullmatch(t):
            return False
        # Pin names in these blocks are normally signal-like, not prose.
        return ("_" in t or "#" in t or "-" in t or any(c.isdigit() for c in t)
                or t.upper() == t)

    @staticmethod
    def _looks_structured_net(text):
        t = text.strip()
        if not t or Extractor._is_pin_coord(t) or Extractor._is_external_id(t):
            return False
        return likely_net(t)

    def _family_body_rect(self, page, items):
        """Return a large body rectangle containing the pin-name/coord columns."""
        rects = self.body_rects.get(page, [])
        if not rects:
            return None
        pts = [(c["coord"].cx, c["coord"].cy) for c in items]
        # The screenshot-style blocks have a large rectangular body with the
        # pin names and pin coordinates inside it. Require strong containment.
        for r in rects:
            if r.width < 80 or r.height < 60:
                continue
            inside = sum(r.x0-3 <= x <= r.x1+3 and r.y0-3 <= y <= r.y1+3 for x,y in pts)
            if inside / max(1, len(pts)) >= 0.80:
                return r
        return None

    def _extract_pin_tables(self):
        """Extract dense IC/BGA/site pin rows from repeated aligned columns.

        Important distinction: a token such as ``R295`` or ``X1`` can look like
        a BGA coordinate syntactically. We therefore never accept one row in
        isolation. A pin table must form a repeated column pattern over at least
        MIN_PIN_TABLE_ROWS rows. This is what makes the detector conservative on
        ordinary schematic pages while still handling pages like SITE2.
        """
        raw = []
        for pi, words in self.text.items():
            rows = self._row_groups(words)
            for row in rows:
                if len(row) < 2:
                    continue
                coords = [w for w in row
                          if self._is_pin_coord(w.text) and not parse_ref(w.text)]
                if not coords:
                    continue
                for coord in coords:
                    pin_names = [w for w in row
                                 if w is not coord and self._pin_name_candidate(w.text)]
                    if not pin_names:
                        continue
                    pin = min(pin_names, key=lambda w: abs(w.cx-coord.cx))
                    nets = [w for w in row
                            if w is not coord and w is not pin and self._looks_structured_net(w.text)]
                    exts = [w for w in row if self._is_external_id(w.text)]
                    netw = None
                    if nets:
                        # Avoid selecting a component value / reference when a
                        # stronger signal-looking label is available.
                        nets.sort(key=lambda w: (
                            not ("_" in w.text or "#" in w.text or re.match(r"^\\d+_", w.text)),
                            abs(w.cx-coord.cx)))
                        netw = nets[0]
                    extw = min(exts, key=lambda w: abs(w.cx-coord.cx)) if exts else None
                    if netw is None and extw is None:
                        continue
                    raw.append({
                        "page": pi, "pin": pin, "coord": coord,
                        "netw": netw, "extw": extw,
                    })

        # Build repeated column families. A real pin-table has stable x-columns:
        # [external-id, net, pin-coordinate, pin-name] or the mirror image.
        families = defaultdict(list)
        for c in raw:
            pi = c["page"]
            q = lambda v: round(v / PIN_COLUMN_TOL)
            key = (
                pi,
                q(c["coord"].cx),
                q(c["pin"].cx),
            )
            families[key].append(c)

        accepted = []
        accepted_rects = {}
        for key, items in families.items():
            # Rows must be distinct vertically; duplicate PDF text objects do not
            # count toward the table size.
            unique_y = sorted({round(c["coord"].cy, 1) for c in items})
            if len(unique_y) < MIN_PIN_TABLE_ROWS:
                continue
            # Reject a family spanning a huge vertical area with only a handful
            # of rows; dense pin tables have regular row spacing.
            diffs = [b-a for a, b in zip(unique_y, unique_y[1:]) if b-a > 0]
            if diffs:
                med = sorted(diffs)[len(diffs)//2]
                if med > 18.0:
                    continue
            body = self._family_body_rect(key[0], items)
            if body is None:
                continue
            accepted.extend(items)
            accepted_rects[key] = body

        # Convert accepted candidates to records.
        for c in accepted:
            coord, pin, netw, extw = c["coord"], c["pin"], c["netw"], c["extw"]
            side = "left" if pin.cx > coord.cx else "right"
            if netw is not None:
                side = "left" if netw.cx < coord.cx else "right"
            self.pin_records.append(PinRecord(
                page=c["page"], block="", pin_name=pin.text.strip(),
                pin_number=coord.text.strip(),
                net=(netw.text.strip() if netw else ""),
                external_id=(extw.text.strip() if extw else ""),
                side=side, row_y=round(coord.cy, 3), x=round(coord.cx, 3),
                evidence="repeated-aligned-pin-table",
            ))

        # Deduplicate exact records.
        seen = set(); uniq = []
        for r in self.pin_records:
            key = (r.page, r.pin_name, r.pin_number, r.net, r.external_id,
                   round(r.row_y, 1), round(r.x, 1))
            if key not in seen:
                seen.add(key); uniq.append(r)
        self.pin_records = uniq

        # Partition records into physical table groups by page/side and nearby
        # rows. This does not require a detected rectangle; some PDFs draw only
        # the vertical block boundary.
        for pi in sorted({r.page for r in self.pin_records}):
            page_recs = [r for r in self.pin_records if r.page == pi]
            page_recs.sort(key=lambda r: (r.side, r.x, r.row_y))
            groups = []
            for r in page_recs:
                placed = False
                for g in reversed(groups[-12:]):
                    if g[0].side != r.side:
                        continue
                    if abs(g[0].x-r.x) > PIN_COLUMN_TOL:
                        continue
                    if abs(r.row_y-g[-1].row_y) <= 25:
                        g.append(r); placed = True; break
                if not placed:
                    groups.append([r])
            gi = 0
            for g in groups:
                if len(g) < MIN_PIN_TABLE_ROWS:
                    continue
                gi += 1
                # Find a nearby heading, preferring SITE names and short uppercase
                # labels. Otherwise use a stable generated table identifier.
                y0 = min(r.row_y for r in g); x0 = sum(r.x for r in g)/len(g)
                words = self.text[pi]
                headings = []
                for w in words:
                    t = w.text.strip()
                    if not t or len(t) > 32 or parse_ref(t):
                        continue
                    if abs(w.cx-x0) > 260 or w.cy > y0 + 25 or w.cy < y0 - 220:
                        continue
                    if t.upper().startswith("SITE") or (t.upper() == t and len(t) <= 20):
                        headings.append((abs(w.cy-y0) - min(40, w.size), w))
                block = min(headings, key=lambda z: z[0])[1].text.strip() if headings else f"PIN_TABLE_P{pi}_{gi}"
                # Ensure uniqueness if the same heading is used on two sides.
                if any(t["page"] == pi and t["block"] == block for t in self.pin_tables):
                    block = f"{block}_{gi}"
                for r in g:
                    r.block = block
                self.pin_tables.append({
                    "page": pi, "block": block, "rows": len(g),
                    "y_min": round(min(r.row_y for r in g), 3),
                    "y_max": round(max(r.row_y for r in g), 3),
                    "side_counts": dict(Counter(r.side for r in g)),
                })

        self.external_links = [{
            "page": r.page, "block": r.block, "pin_name": r.pin_name,
            "pin_number": r.pin_number, "net": r.net,
            "external_id": r.external_id, "side": r.side,
            "evidence": r.evidence,
        } for r in self.pin_records if r.external_id or r.net]


    def _extract_signal_rows(self):
        """Capture visible signal/net rows even when they are not a formal SITE table.

        This is intentionally text-first: many laptop schematics encode useful
        cross-sheet references as text next to a short wire/port glyph.  Requiring
        a component body would lose that information.  Every accepted row must
        still be close to traced vector geometry.
        """
        seen = set()
        for pi, words in self.text.items():
            rows = self._row_groups(words)
            for row in rows:
                nets = [w for w in row if self._looks_structured_net(w.text)]
                exts = [w for w in row if self._is_external_id(w.text)]
                coords = [w for w in row if self._is_pin_coord(w.text) and not parse_ref(w.text)]
                if not nets:
                    continue
                # Prefer the strongest signal-like token in the row.
                nets.sort(key=lambda w: (
                    not ("_" in w.text or "#" in w.text or "/" in w.text or re.match(r"^\d+_", w.text)),
                    -len(w.text)))
                nw = nets[0]
                wc, d = self._nearest_cluster_fast(pi, nw.cx, nw.cy, LABEL_WIRE_TOL + 4.0)
                if wc is None:
                    continue
                ext = min(exts, key=lambda w: abs(w.cx-nw.cx)).text if exts else ""
                coord = min(coords, key=lambda w: abs(w.cx-nw.cx)).text if coords else ""
                key = (pi, nw.text, ext, coord, round(nw.cy,1))
                if key in seen:
                    continue
                seen.add(key)
                self.signal_rows.append({
                    "page": pi, "net": nw.text.strip(), "external_id": ext.strip(),
                    "pin_number": coord.strip(), "row_y": round(nw.cy,3),
                    "cluster": wc.cluster_id, "distance": round(d,3),
                    "evidence": "visible-row+vector-wire"
                })

    def _profile_pages(self):
        """Classify each page cheaply so downstream code can choose suitable logic."""
        pins_by_page = Counter(r.page for r in self.pin_records)
        comps_by_page = Counter(c.page for c in self.components)
        labels_by_page = Counter(n.page for n in self.net_labels)
        for pi in sorted(self.pages):
            p = self.pages[pi]
            segs = self.segments[pi]
            horiz = sum(s.horizontal for s in segs)
            vert = sum(s.vertical for s in segs)
            pins = pins_by_page[pi]
            comps = comps_by_page[pi]
            if pins >= MIN_PIN_TABLE_ROWS:
                kind = "dense_pin_map"
            elif comps >= 20 and vert >= 20:
                kind = "component_dense_circuit"
            elif vert > horiz * 0.55 and comps >= 8:
                kind = "power_or_passive_bank"
            else:
                kind = "general_schematic"
            self.page_profiles.append({
                "page": pi, "type": kind, "width": round(p.rect.width,2),
                "height": round(p.rect.height,2), "text_items": len(self.text[pi]),
                "wire_segments": len(segs), "wire_clusters": len(self.clusters[pi]),
                "components": comps, "net_labels": labels_by_page[pi],
                "pin_records": pins, "horizontal_segments": horiz,
                "vertical_segments": vert
            })

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
            "structured_pin_records": len(self.pin_records),
            "signal_rows": len(self.signal_rows),
            "page_profiles": len(self.page_profiles),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "pin_records": len(self.pin_records),
            "pin_tables": len(self.pin_tables),
            "external_links": len(self.external_links),
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
            "pin_records": [r.__dict__ for r in self.pin_records],
            "pin_tables": self.pin_tables,
            "external_links": self.external_links,
            "signal_rows": self.signal_rows,
            "page_profiles": self.page_profiles,
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

    def export_pin_csvs(self, prefix):
        prefix = Path(prefix)
        with open(str(prefix) + "_pin_records.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "block", "pin_name", "pin_number", "net", "external_id", "side", "row_y", "x", "evidence"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in self.pin_records:
                w.writerow(r.__dict__)

        with open(str(prefix) + "_pin_tables.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "block", "rows", "y_min", "y_max", "side_counts"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in self.pin_tables:
                rr = dict(r); rr["side_counts"] = json.dumps(rr["side_counts"], sort_keys=True)
                w.writerow(rr)

        with open(str(prefix) + "_external_links.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "block", "pin_name", "pin_number", "net", "external_id", "side", "evidence"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.external_links)

        with open(str(prefix) + "_signal_rows.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "net", "external_id", "pin_number", "row_y", "cluster", "distance", "evidence"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.signal_rows)

        with open(str(prefix) + "_page_profiles.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "type", "width", "height", "text_items", "wire_segments", "wire_clusters", "components", "net_labels", "pin_records", "horizontal_segments", "vertical_segments"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.page_profiles)

        with open(str(prefix) + "_page_text.csv", "w", newline="", encoding="utf-8") as f:
            fields = ["page", "text", "x0", "y0", "x1", "y1", "size"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for pi, words in self.text.items():
                for item in words:
                    w.writerow({"page": pi, **item.__dict__})

    def netlist_rows(self):
        """Build the canonical Net Name -> Net Pins table.

        Priority is given to structured pin records because those contain an
        explicit block/site + pin coordinate. Generic component/net records
        are retained as component endpoints when an exact pin is unavailable.
        Anonymous nets are omitted from the canonical human-facing netlist.
        """
        by_net = defaultdict(set)

        # Exact/structured pins: SITE1 + A11 => SITE1.A11.
        for r in self.pin_records:
            net = (r.net or "").strip()
            pin = (r.pin_number or "").strip()
            block = (r.block or "").strip()
            if not net or net.startswith("_ANON_") or not pin:
                continue
            endpoint = f"{block}.{pin}" if block else pin
            by_net[net].add(endpoint)

        # Generic component associations. These are useful for passives and
        # pages where the PDF does not expose an unambiguous pin coordinate.
        for r in self.component_nets:
            net = str(r.get("net", "")).strip()
            comp = str(r.get("component", "")).strip()
            if not net or net.startswith("_ANON_") or not comp:
                continue
            # Avoid adding a bare block name if exact pins for that block/net
            # are already present.
            if not any(x.startswith(comp + ".") for x in by_net[net]):
                by_net[net].add(comp)

        rows = []
        for net in sorted(by_net, key=lambda x: (x.upper(), x)):
            pins = sorted(by_net[net], key=lambda x: (x.upper(), x))
            rows.append({"Net Name": net, "Net Pins": " ".join(pins),
                         "Pin Count": len(pins)})
        return rows

    def export_netlist_csv(self, path):
        rows = self.netlist_rows()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            fields = ["Net Name", "Net Pins", "Pin Count"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

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
    ap.add_argument("--out-prefix", default="schematic_v6")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    ex = Extractor(args.pdf).parse(verbose=True)
    ex.export_json(str(out) + ".json")
    ex.export_graphml(str(out) + ".graphml")
    ex.export_csvs(out)
    ex.export_pin_csvs(out)
    ex.export_netlist_csv(str(out) + "_netlist.csv")
    ex.export_summary(str(out) + "_summary.json")
    if args.viz:
        ex.export_overview(str(out) + "_overview.png")
    print("\nExtraction complete")
    print(json.dumps(ex.summary(), indent=2))


if __name__ == "__main__":
    main()
