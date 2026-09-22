#!/usr/bin/env python3
"""
Schematic graph extractor v3

Designed for vector PDFs exported from EDA tools (especially Altium-style PDFs).

Core principle:
  1. Prefer embedded schematic metadata (PI..., NL..., PO..., CO... tokens) when present.
  2. Use vector wire geometry to reconstruct physical nets.
  3. Use pin metadata locations rather than guessing pins from component proximity.
  4. Treat wire crossings conservatively: an intersection is a connection only when
     a wire endpoint/T-junction reaches it. A pure crossing is NOT merged.
  5. Use visible/embedded net labels and power labels to merge nets across pages.
  6. Never invent a connection when evidence is insufficient; unresolved pins are
     reported separately.

Outputs:
  <prefix>.graphml
  <prefix>.json
  <prefix>_component_nets.csv
  <prefix>_component_edges.csv
  <prefix>_unresolved_pins.csv
  <prefix>_net_aliases.csv
  optional: <prefix>_overview.png

Run:
  python schematic_graph_v3.py input.pdf --out-prefix schematic --viz

Dependencies:
  PyMuPDF (fitz), NetworkX
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import networkx as nx


# ----------------------------- configuration -----------------------------

BLUE_COLORS = {
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 0.502),
    (0.0, 0.0, 0.5019999742507935),
    (0.0, 0.0, 0.7059),
}
BLACK = (0.0, 0.0, 0.0)
YELLOW_MIN = (0.80, 0.80, 0.25)

WIRE_TOUCH_TOL = 1.20
PIN_LINE_TOL = 3.50
LABEL_LINE_TOL = 4.50
PIN_WIRE_MAX_DIST = 18.0
BLACK_BRIDGE_MAX_DIST = 20.0
BLACK_BRIDGE_TOUCH_TOL = 1.8
LABEL_WIRE_MAX_DIST = 18.0
POWER_LABEL_MAX_DIST = 20.0

# Common EDA power names. These are only used when a visible power label is
# actually close to a wire endpoint; they are never assigned globally by guess.
POWER_NAMES = {
    "GND", "VDD", "VSSA", "VDDA", "VREF+", "3V", "5V", "2V5",
    "U5V", "VBUS", "COM", "VBAT", "VCC", "VSS", "VDD1", "VDD2",
    "VDD3", "VDD4", "VDD5", "VDD12", "VSS1", "VSS2", "VSS3",
    "VSS4", "VSS5",
}

# Component references that can be split into units in CO tokens.
UNIT_SUFFIX_RE = re.compile(r"^([A-Za-z]+\d+)[A-Za-z]$")


@dataclass(frozen=True)
class Segment:
    page: int
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    color: Tuple[float, float, float]

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class Pin:
    ref: str
    pin: str
    page: int
    x: float
    y: float
    source_token: str
    wire_cluster: Optional[int] = None
    attachment_distance: Optional[float] = None
    attachment_method: str = "unresolved"


@dataclass
class NetAlias:
    page: int
    name: str
    x: float
    y: float
    source: str
    token: str = ""
    confidence: float = 1.0


@dataclass
class ClusterInfo:
    page: int
    cluster_id: int
    segments: List[int] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    alias_sources: List[str] = field(default_factory=list)


# ----------------------------- geometry helpers -----------------------------


def point_segment_distance(px: float, py: float, s: Segment) -> Tuple[float, float]:
    vx = s.x2 - s.x1
    vy = s.y2 - s.y1
    vv = vx * vx + vy * vy
    if vv == 0:
        return math.hypot(px - s.x1, py - s.y1), 0.0
    t = ((px - s.x1) * vx + (py - s.y1) * vy) / vv
    tc = max(0.0, min(1.0, t))
    qx = s.x1 + tc * vx
    qy = s.y1 + tc * vy
    return math.hypot(px - qx, py - qy), tc


def infinite_line_distance(px: float, py: float, s: Segment) -> float:
    vx = s.x2 - s.x1
    vy = s.y2 - s.y1
    ll = math.hypot(vx, vy)
    if ll == 0:
        return math.hypot(px - s.x1, py - s.y1)
    return abs((px - s.x1) * vy - (py - s.y1) * vx) / ll


def bbox_close(a: Segment, b: Segment, tol: float) -> bool:
    return not (
        max(min(a.x1, a.x2), min(b.x1, b.x2))
        > min(max(a.x1, a.x2), max(b.x1, b.x2)) + tol
        or max(min(a.y1, a.y2), min(b.y1, b.y2))
        > min(max(a.y1, a.y2), max(b.y1, b.y2)) + tol
    )


def endpoint_to_segment_distance(x: float, y: float, s: Segment) -> float:
    return point_segment_distance(x, y, s)[0]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


# ----------------------------- token helpers -----------------------------


def normalize_ref(ref: str) -> str:
    # U4A/U4B are two units of the same logical component U4.
    m = UNIT_SUFFIX_RE.match(ref)
    if m and ref.startswith("U4"):
        return m.group(1)
    return ref


def tokenize_metadata_word(word: str) -> List[Tuple[str, str]]:
    """Split PDF words containing concatenated hidden EDA tokens.

    Example: PIU2032NLUSB0DM -> PIU2032, NLUSB0DM
    and PILD102PIR101 -> PILD102, PIR101.
    """
    matches = list(re.finditer(r"(?:PI|NL|PO|CO)(?=[A-Za-z0-9_])", word))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(word)
        tok = word[start:end]
        if len(tok) >= 3:
            out.append((tok, tok[:2]))
    return out


def extract_co_refs(page: fitz.Page) -> List[str]:
    refs: List[str] = []
    for w in page.get_text("words"):
        text = w[4]
        for tok, kind in tokenize_metadata_word(text):
            if kind == "CO" and len(tok) > 2:
                refs.append(normalize_ref(tok[2:]))
    return refs


def build_ref_set(doc: fitz.Document) -> List[str]:
    refs = set()
    for page in doc:
        refs.update(extract_co_refs(page))
    # CO can occasionally miss a component; recover obvious visible designators.
    for page in doc:
        for w in page.get_text("words"):
            t = w[4]
            if re.fullmatch(r"(?:R|C|D|L|X|U|CN|JP|SB|LD|B|T|P)\d+[A-Za-z]?", t):
                refs.add(normalize_ref(t))
    return sorted(refs, key=lambda x: (-len(x), x))


def parse_pi_token(token: str, refs: Sequence[str]) -> Optional[Tuple[str, str]]:
    if not token.startswith("PI"):
        return None
    body = token[2:]
    for ref in refs:
        if body.startswith(ref):
            pin = body[len(ref):]
            # Some EDA exports use 0 as a separator before alphabetic pin IDs.
            if pin.startswith("0") and len(pin) > 1:
                pin = pin[1:]
            if pin:
                return normalize_ref(ref), pin
    return None


def visible_words(page: fitz.Page) -> List[Tuple]:
    out = []
    for w in page.get_text("words"):
        t = w[4]
        if any(t.startswith(prefix) for prefix in ("PI", "CO", "NL", "PO")):
            continue
        out.append(w)
    return out


def nearest_visible_text(page: fitz.Page, x: float, y: float, max_dist: float = 8.0) -> Optional[Tuple[str, float]]:
    best = None
    for w in visible_words(page):
        cx = (w[0] + w[2]) / 2
        cy = (w[1] + w[3]) / 2
        d = math.hypot(x - cx, y - cy)
        if d <= max_dist and (best is None or d < best[1]):
            best = (w[4], d)
    return best


def extract_pins(doc: fitz.Document, refs: Sequence[str]) -> List[Pin]:
    pins: Dict[Tuple[int, str, str, int, int], Pin] = {}
    for page_idx, page in enumerate(doc):
        for w in page.get_text("words"):
            text = w[4]
            tokens = tokenize_metadata_word(text)
            # When a PI token is concatenated with another hidden token, the
            # PDF gives one bounding box for the whole string. Clean PI tokens
            # are emitted separately in this export; skipping the concatenated
            # PI avoids duplicate pins with incorrect coordinates.
            if len(tokens) != 1:
                continue
            for tok, kind in tokens:
                if kind != "PI":
                    continue
                parsed = parse_pi_token(tok, refs)
                if not parsed:
                    continue
                ref, pin_no = parsed
                x = (w[0] + w[2]) / 2
                y = (w[1] + w[3]) / 2
                key = (page_idx, ref, pin_no, round(x, 3), round(y, 3))
                pins[key] = Pin(ref, pin_no, page_idx, x, y, tok)
    return list(pins.values())


def extract_hidden_aliases(doc: fitz.Document) -> List[NetAlias]:
    aliases: List[NetAlias] = []
    for page_idx, page in enumerate(doc):
        for w in page.get_text("words"):
            text = w[4]
            for tok, kind in tokenize_metadata_word(text):
                if kind not in {"NL", "PO"}:
                    continue
                x = (w[0] + w[2]) / 2
                y = (w[1] + w[3]) / 2
                nearby = nearest_visible_text(page, x, y, 8.0)
                if nearby:
                    name = nearby[0]
                    conf = max(0.65, 1.0 - nearby[1] / 12.0)
                else:
                    # Conservative fallback decoder for common Altium hidden names.
                    raw = tok[2:]
                    name = decode_hidden_net_name(raw)
                    conf = 0.55
                aliases.append(NetAlias(page_idx, name, x, y, kind, tok, conf))
    return aliases


def decode_hidden_net_name(raw: str) -> str:
    """Best-effort fallback only; visible text wins whenever available."""
    # Altium's hidden text often encodes '_' as 0 when followed by a letter.
    s = raw.replace("0IN", "_IN").replace("0OUT", "_OUT")
    s = s.replace("0JTCK", "_JTCK").replace("0JTMS", "_JTMS")
    s = s.replace("0RST", "_RST").replace("0SWO", "_SWO")
    s = s.replace("0SWDIO", "_SWDIO")
    s = s.replace("0STLINK", "_STLINK")
    s = s.replace("0DM", "_DM").replace("0DP", "_DP")
    s = s.replace("0MCO", "_MCO")
    # Preserve ordinary port names such as PA10.
    return s


# ----------------------------- vector extraction -----------------------------


def is_blue(color: Optional[Tuple[float, float, float]]) -> bool:
    if color is None:
        return False
    return any(sum((color[i] - c[i]) ** 2 for i in range(3)) < 2e-5 for c in BLUE_COLORS)


def extract_blue_segments(page: fitz.Page, page_idx: int) -> List[Segment]:
    segs: List[Segment] = []
    idx = 0
    for drawing in page.get_drawings():
        color = drawing.get("color")
        if not is_blue(color):
            continue
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            s = Segment(page_idx, idx, a.x, a.y, b.x, b.y, tuple(color))
            if s.length > 0.5:
                segs.append(s)
                idx += 1
    return segs


def extract_black_segments(page: fitz.Page, page_idx: int) -> List[Segment]:
    segs: List[Segment] = []
    idx = 0
    for drawing in page.get_drawings():
        color = drawing.get("color")
        if color != BLACK:
            continue
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            s = Segment(page_idx, idx, a.x, a.y, b.x, b.y, BLACK)
            if s.length > 0.5:
                segs.append(s)
                idx += 1
    return segs


def _direction(s: Segment) -> Tuple[float, float]:
    dx = s.x2 - s.x1
    dy = s.y2 - s.y1
    ll = math.hypot(dx, dy)
    if ll == 0:
        return 0.0, 0.0
    return dx / ll, dy / ll


def _collinear_opposite(a: Segment, b: Segment, px: float, py: float, tol: float = 0.995) -> bool:
    """True when a and b form a straight continuation through (px, py)."""
    da = _direction(a)
    db = _direction(b)
    parallel = abs(da[0] * db[1] - da[1] * db[0]) < 0.08
    if not parallel:
        return False
    # Compare vectors from the junction toward the far endpoints.
    far_a = (a.x2, a.y2) if math.hypot(a.x2 - px, a.y2 - py) > math.hypot(a.x1 - px, a.y1 - py) else (a.x1, a.y1)
    far_b = (b.x2, b.y2) if math.hypot(b.x2 - px, b.y2 - py) > math.hypot(b.x1 - px, b.y1 - py) else (b.x1, b.y1)
    va = (far_a[0] - px, far_a[1] - py)
    vb = (far_b[0] - px, far_b[1] - py)
    la = math.hypot(*va)
    lb = math.hypot(*vb)
    if la == 0 or lb == 0:
        return False
    dot = (va[0] * vb[0] + va[1] * vb[1]) / (la * lb)
    return dot < -tol


def build_wire_clusters(segments: Sequence[Segment], tol: float = WIRE_TOUCH_TOL) -> Tuple[UnionFind, Dict[int, List[int]]]:
    """Build conservative wire topology.

    Endpoint-to-interior joins are treated as T-junctions only when the
    through-segment does not have a collinear continuation. This prevents a
    PDF that split a crossing line into two pieces from turning a plain
    crossing into a false electrical junction.
    """
    n = len(segments)
    uf = UnionFind(n)

    # First: exact/near endpoint-to-endpoint joins.
    for i in range(n):
        a = segments[i]
        for j in range(i + 1, n):
            b = segments[j]
            if not bbox_close(a, b, tol):
                continue
            endpoint_pairs = [
                (a.x1, a.y1, b.x1, b.y1), (a.x1, a.y1, b.x2, b.y2),
                (a.x2, a.y2, b.x1, b.y1), (a.x2, a.y2, b.x2, b.y2),
            ]
            if min(math.hypot(x1-x2, y1-y2) for x1,y1,x2,y2 in endpoint_pairs) <= tol:
                uf.union(i, j)

    # Build a spatial endpoint index for continuation checks.
    cell = max(4.0, tol * 4.0)
    endpoint_buckets: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = defaultdict(list)
    for i, s in enumerate(segments):
        for x, y in ((s.x1, s.y1), (s.x2, s.y2)):
            endpoint_buckets[(round(x/cell), round(y/cell))].append((i, x, y))

    def nearby_endpoints(px: float, py: float) -> Iterable[Tuple[int, float, float]]:
        gx, gy = round(px/cell), round(py/cell)
        for ix in range(gx-1, gx+2):
            for iy in range(gy-1, gy+2):
                for item in endpoint_buckets.get((ix, iy), []):
                    if math.hypot(item[1]-px, item[2]-py) <= tol:
                        yield item

    # Then: endpoint-to-interior T-junctions.
    for i, a in enumerate(segments):
        for px, py in ((a.x1, a.y1), (a.x2, a.y2)):
            for j, b in enumerate(segments):
                if i == j or not bbox_close(a, b, tol):
                    continue
                pd, t = point_segment_distance(px, py, b)
                if pd > tol or t <= tol / max(b.length, 1.0) or t >= 1.0 - tol / max(b.length, 1.0):
                    continue

                # If another segment continues b straight through this point,
                # this is a crossing unless the PDF contains an explicit dot.
                has_continuation = False
                for k, ex, ey in nearby_endpoints(px, py):
                    if k in (i, j):
                        continue
                    c = segments[k]
                    if _collinear_opposite(b, c, px, py):
                        has_continuation = True
                        break
                if not has_continuation:
                    uf.union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return uf, groups


def nearest_blue_cluster(
    x: float,
    y: float,
    segments: Sequence[Segment],
    uf: UnionFind,
    max_perp: float = PIN_LINE_TOL,
    max_dist: float = PIN_WIRE_MAX_DIST,
) -> Optional[Tuple[int, float, float]]:
    candidates = []
    for i, s in enumerate(segments):
        ld = infinite_line_distance(x, y, s)
        pd, t = point_segment_distance(x, y, s)
        if ld <= max_perp and pd <= max_dist:
            candidates.append((pd, i, t))
    if not candidates:
        return None
    pd, i, t = min(candidates, key=lambda z: z[0])
    return uf.find(i), pd, t


def attach_via_black_pin_stub(
    x: float,
    y: float,
    blue: Sequence[Segment],
    blue_uf: UnionFind,
    black: Sequence[Segment],
) -> Optional[Tuple[int, float]]:
    candidates = []
    for s in black:
        ld = infinite_line_distance(x, y, s)
        pd, _ = point_segment_distance(x, y, s)
        if ld <= PIN_LINE_TOL and pd <= BLACK_BRIDGE_MAX_DIST:
            candidates.append((pd, s))
    if not candidates:
        return None
    pd, black_seg = min(candidates, key=lambda z: z[0])
    best = None
    for ex, ey in ((black_seg.x1, black_seg.y1), (black_seg.x2, black_seg.y2)):
        for i, bs in enumerate(blue):
            d, _ = point_segment_distance(ex, ey, bs)
            if d <= BLACK_BRIDGE_TOUCH_TOL:
                if best is None or d < best[0]:
                    best = (d, i)
    if best is None:
        return None
    return blue_uf.find(best[1]), pd


def yellow_shape_anchor(page: fitz.Page, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Return the likely wire-tip of the nearest yellow EDA label shape."""
    candidates = []
    for d in page.get_drawings():
        fill = d.get("fill")
        r = d.get("rect")
        if not fill or r is None:
            continue
        if not (fill[0] >= YELLOW_MIN[0] and fill[1] >= YELLOW_MIN[1] and fill[2] <= YELLOW_MIN[2]):
            continue
        dx = 0 if r.x0 <= x <= r.x1 else min(abs(x-r.x0), abs(x-r.x1))
        dy = 0 if r.y0 <= y <= r.y1 else min(abs(y-r.y0), abs(y-r.y1))
        d2 = math.hypot(dx, dy)
        if d2 <= 8.0:
            candidates.append((d2, r))
    if not candidates:
        return None
    _, r = min(candidates, key=lambda z: z[0])
    # The text is normally just outside the arrow body. Choose the nearest
    # vertical edge; this is the connector/wire tip for left/right labels.
    if x >= r.x1:
        ax = r.x1
    elif x <= r.x0:
        ax = r.x0
    else:
        ax = r.x1 if abs(x-r.x1) < abs(x-r.x0) else r.x0
    ay = min(max(y, r.y0), r.y1)
    return ax, ay


def label_to_blue_cluster(
    alias: NetAlias,
    page: fitz.Page,
    segments: Sequence[Segment],
    uf: UnionFind,
) -> Optional[Tuple[int, float, float]]:
    # Hidden NL/PO tokens are usually positioned over the visible net label.
    # Use the nearest matching visible word's CENTER, which is more stable
    # than a text baseline when several parallel nets are spaced closely.
    best_word = None
    best_dist = None
    for w in visible_words(page):
        cx0 = (w[0] + w[2]) / 2
        cy0 = (w[1] + w[3]) / 2
        d0 = math.hypot(alias.x - cx0, alias.y - cy0)
        if d0 <= 8.0 and (best_dist is None or d0 < best_dist):
            best_word, best_dist = w, d0
    if best_word is not None:
        cx = (best_word[0] + best_word[2]) / 2
        cy = (best_word[1] + best_word[3]) / 2
        c = nearest_blue_cluster(
            cx, cy, segments, uf, max_perp=4.0, max_dist=LABEL_WIRE_MAX_DIST
        )
        if c:
            return c

    anchor = yellow_shape_anchor(page, alias.x, alias.y)
    x, y = anchor if anchor else (alias.x, alias.y)
    return nearest_blue_cluster(
        x, y, segments, uf, max_perp=LABEL_LINE_TOL, max_dist=LABEL_WIRE_MAX_DIST
    )


# ----------------------------- visible power labels -----------------------------


def attach_power_labels(
    page: fitz.Page,
    page_idx: int,
    blue: Sequence[Segment],
    uf: UnionFind,
) -> List[NetAlias]:
    out: List[NetAlias] = []
    for w in visible_words(page):
        name = w[4]
        if name not in POWER_NAMES:
            continue
        x = (w[0] + w[2]) / 2
        y = (w[1] + w[3]) / 2
        best = None
        for s in blue:
            pd, t = point_segment_distance(x, y, s)
            if pd <= POWER_LABEL_MAX_DIST:
                endpoint_distance = min(t, 1.0 - t)
                score = pd + endpoint_distance * 8.0
                if best is None or score < best[0]:
                    best = (score, pd, t, s)
        if best is None:
            continue
        _, pd, t, s = best
        # Power text is usually offset from the symbol/wire endpoint. Require
        # either a near endpoint or a very close text-to-wire distance.
        if min(t, 1 - t) <= 0.20 or pd <= 7.0:
            out.append(NetAlias(page_idx, name, x, y, "power-label", name, 0.90 if pd <= 7 else 0.75))
    return out


# ----------------------------- yellow net-label detection -----------------------------


def extract_yellow_label_aliases(page: fitz.Page, page_idx: int) -> List[NetAlias]:
    """Read labels inside EDA-style yellow net/port shapes.

    This complements hidden NL/PO metadata and helps with local signal labels
    such as Audio_SDA, PDM_OUT, I2S3_SCK, etc.
    """
    yellow_rects = []
    for d in page.get_drawings():
        fill = d.get("fill")
        if not fill:
            continue
        if fill[0] >= YELLOW_MIN[0] and fill[1] >= YELLOW_MIN[1] and fill[2] <= YELLOW_MIN[2]:
            yellow_rects.append(d.get("rect"))

    if not yellow_rects:
        return []

    out = []
    for w in visible_words(page):
        x = (w[0] + w[2]) / 2
        y = (w[1] + w[3]) / 2
        inside = any(r.x0 - 1 <= x <= r.x1 + 1 and r.y0 - 1 <= y <= r.y1 + 1 for r in yellow_rects)
        if not inside:
            continue
        name = w[4].strip()
        if not name or name in POWER_NAMES:
            continue
        # Avoid component designators and obvious values.
        if re.fullmatch(r"(?:R|C|D|L|X|U|CN|JP|SB|LD|B|T|P)\d+", name):
            continue
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:K|M|nF|pF|uF|V|MHz|kHz|mA|A)?", name, re.I):
            continue
        out.append(NetAlias(page_idx, name, x, y, "yellow-label", name, 0.92))
    return out


# ----------------------------- graph builder -----------------------------


class SchematicGraphV3:
    def __init__(self, pdf_path: str):
        self.pdf_path = str(pdf_path)
        self.doc = fitz.open(self.pdf_path)
        self.refs = build_ref_set(self.doc)
        self.pins = extract_pins(self.doc, self.refs)
        self.hidden_aliases = extract_hidden_aliases(self.doc)
        self.page_blue: Dict[int, List[Segment]] = {}
        self.page_black: Dict[int, List[Segment]] = {}
        self.page_uf: Dict[int, UnionFind] = {}
        self.page_groups: Dict[int, Dict[int, List[int]]] = {}
        self.cluster_aliases: Dict[Tuple[int, int], List[NetAlias]] = defaultdict(list)
        self.pin_by_net: Dict[str, List[Pin]] = defaultdict(list)
        self.unresolved: List[Pin] = []
        self.net_aliases: Dict[str, List[str]] = defaultdict(list)
        self.graph = nx.Graph()

    def prepare_geometry(self) -> None:
        for page_idx, page in enumerate(self.doc):
            blue = extract_blue_segments(page, page_idx)
            black = extract_black_segments(page, page_idx)
            uf, groups = build_wire_clusters(blue)
            self.page_blue[page_idx] = blue
            self.page_black[page_idx] = black
            self.page_uf[page_idx] = uf
            self.page_groups[page_idx] = groups

    def attach_aliases(self) -> None:
        # Hidden NL/PO aliases.
        for alias in self.hidden_aliases:
            blue = self.page_blue[alias.page]
            uf = self.page_uf[alias.page]
            c = label_to_blue_cluster(alias, self.doc[alias.page], blue, uf)
            if c:
                self.cluster_aliases[(alias.page, c[0])].append(alias)

        # Visible yellow labels.
        for page_idx, page in enumerate(self.doc):
            for alias in extract_yellow_label_aliases(page, page_idx):
                c = label_to_blue_cluster(alias, page, self.page_blue[page_idx], self.page_uf[page_idx])
                if c:
                    self.cluster_aliases[(page_idx, c[0])].append(alias)

        # Power labels.
        for page_idx, page in enumerate(self.doc):
            for alias in attach_power_labels(page, page_idx, self.page_blue[page_idx], self.page_uf[page_idx]):
                c = label_to_blue_cluster(alias, page, self.page_blue[page_idx], self.page_uf[page_idx])
                if c:
                    self.cluster_aliases[(page_idx, c[0])].append(alias)

    def attach_pins(self) -> None:
        for pin in self.pins:
            blue = self.page_blue[pin.page]
            uf = self.page_uf[pin.page]
            direct = nearest_blue_cluster(pin.x, pin.y, blue, uf)
            if direct:
                pin.wire_cluster = direct[0]
                pin.attachment_distance = direct[1]
                pin.attachment_method = "pin-metadata-to-vector-wire"
                continue

            bridge = attach_via_black_pin_stub(
                pin.x, pin.y, blue, uf, self.page_black[pin.page]
            )
            if bridge:
                pin.wire_cluster = bridge[0]
                pin.attachment_distance = bridge[1]
                pin.attachment_method = "pin-metadata-to-black-stub-to-wire"
                continue

            pin.attachment_method = "unresolved"
            self.unresolved.append(pin)

    @staticmethod
    def canonical_net_name(names: Sequence[str]) -> Optional[str]:
        cleaned = []
        for n in names:
            n = n.strip()
            if n and n not in cleaned:
                cleaned.append(n)
        if not cleaned:
            return None
        # Prefer the shortest non-generic alias when several textual variants
        # describe the same physical net; preserve all aliases separately.
        generic = {"GND", "VDD", "3V", "5V"}
        non_generic = [n for n in cleaned if n not in generic]
        return sorted(non_generic or cleaned, key=lambda s: (len(s), s))[0]

    def build_nets(self) -> Dict[Tuple[int, int], str]:
        cluster_to_net: Dict[Tuple[int, int], str] = {}
        named_groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # First assign names to physical clusters.
        for key, aliases in self.cluster_aliases.items():
            names = []
            for a in aliases:
                if a.name not in names:
                    names.append(a.name)
            canonical = self.canonical_net_name(names)
            if canonical:
                cluster_to_net[key] = f"NET::{canonical}"
                for name in names:
                    named_groups[name].append(key)

        # Unnamed clusters remain page-local and cannot be globally merged.
        for page_idx, groups in self.page_groups.items():
            for root in groups:
                key = (page_idx, root)
                if key not in cluster_to_net:
                    cluster_to_net[key] = f"WIRE::P{page_idx+1}::{root}"

        # Merge all same-name aliases across pages by using the canonical name.
        # Alias variants are recorded for auditability.
        for name, keys in named_groups.items():
            canonical = f"NET::{name}"
            for key in keys:
                cluster_to_net[key] = canonical
                if name not in self.net_aliases[canonical]:
                    self.net_aliases[canonical].append(name)

        # Attach pins to their net nodes.
        for pin in self.pins:
            if pin.wire_cluster is None:
                continue
            key = (pin.page, pin.wire_cluster)
            net = cluster_to_net.get(key)
            if net:
                self.pin_by_net[net].append(pin)

        return cluster_to_net

    def build_graph(self) -> nx.Graph:
        self.prepare_geometry()
        self.attach_aliases()
        self.attach_pins()
        cluster_to_net = self.build_nets()

        g = nx.Graph()
        # Component nodes.
        component_refs = sorted({p.ref for p in self.pins})
        for ref in component_refs:
            g.add_node(ref, node_type="component")

        # Net nodes and component-net edges.
        for net, pins in self.pin_by_net.items():
            if not pins:
                continue
            g.add_node(net, node_type="net", name=net.replace("NET::", "").replace("WIRE::", ""))
            for pin in pins:
                g.add_edge(
                    pin.ref,
                    net,
                    pin=str(pin.pin),
                    page=pin.page + 1,
                    attachment_distance=round(pin.attachment_distance or 0.0, 3),
                    attachment_method=pin.attachment_method,
                )

        # Direct component graph as a separate edge set in attributes. We do
        # not replace the bipartite graph because the net node is the auditable
        # evidence for each connection.
        for net, pins in self.pin_by_net.items():
            by_ref = defaultdict(list)
            for p in pins:
                by_ref[p.ref].append(p)
            refs = sorted(by_ref)
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    a, b = refs[i], refs[j]
                    pin_a = ",".join(sorted(str(p.pin) for p in by_ref[a]))
                    pin_b = ",".join(sorted(str(p.pin) for p in by_ref[b]))
                    if g.has_edge(a, b) and g.edges[a, b].get("edge_type") == "component_connection":
                        old = g.edges[a, b]
                        old["nets"] = old.get("nets", "") + ";" + net
                    else:
                        # Do not add a second edge to the same Graph edge if the
                        # bipartite net node already occupies it; use a separate
                        # GraphML attribute later via component_edges.csv.
                        pass

        self.graph = g
        self.cluster_to_net = cluster_to_net
        return g

    def component_edges(self) -> List[Dict[str, str]]:
        rows = []
        for net, pins in sorted(self.pin_by_net.items()):
            by_ref = defaultdict(list)
            for p in pins:
                by_ref[p.ref].append(p)
            refs = sorted(by_ref)
            for i, a in enumerate(refs):
                for b in refs[i + 1:]:
                    rows.append({
                        "source": a,
                        "target": b,
                        "net": net.replace("NET::", "").replace("WIRE::", ""),
                        "source_pins": ",".join(sorted(str(p.pin) for p in by_ref[a])),
                        "target_pins": ",".join(sorted(str(p.pin) for p in by_ref[b])),
                    })
        return rows

    def summary(self) -> Dict[str, int]:
        connected_pins = sum(len(v) for v in self.pin_by_net.values())
        named_nets = sum(1 for n in self.pin_by_net if n.startswith("NET::"))
        return {
            "pages": len(self.doc),
            "components": len({p.ref for p in self.pins}),
            "pin_records": len(self.pins),
            "connected_pins": connected_pins,
            "unresolved_pins": len(self.unresolved),
            "nets_with_pins": len(self.pin_by_net),
            "named_nets_with_pins": named_nets,
            "component_edges": len(self.component_edges()),
        }

    # ------------------------- exports -------------------------

    def export_graphml(self, path: str) -> None:
        nx.write_graphml(self.graph, path)

    def export_json(self, path: str) -> None:
        data = {
            "source_pdf": self.pdf_path,
            "summary": self.summary(),
            "nodes": [dict(id=n, **d) for n, d in self.graph.nodes(data=True)],
            "edges": [dict(source=u, target=v, **d) for u, v, d in self.graph.edges(data=True)],
            "component_edges": self.component_edges(),
            "unresolved_pins": [asdict(p) for p in self.unresolved],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_component_nets(self, path: str) -> None:
        rows = []
        for net, pins in sorted(self.pin_by_net.items()):
            name = net.replace("NET::", "").replace("WIRE::", "")
            for p in sorted(pins, key=lambda x: (x.ref, str(x.pin))):
                rows.append({
                    "component": p.ref,
                    "pin": p.pin,
                    "net": name,
                    "page": p.page + 1,
                    "attachment_distance": round(p.attachment_distance or 0.0, 3),
                    "attachment_method": p.attachment_method,
                })
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "component", "pin", "net", "page", "attachment_distance", "attachment_method"
            ])
            writer.writeheader()
            writer.writerows(rows)

    def export_component_edges(self, path: str) -> None:
        rows = self.component_edges()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "target", "net", "source_pins", "target_pins"
            ])
            writer.writeheader()
            writer.writerows(rows)

    def export_unresolved(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "component", "pin", "page", "x", "y", "source_token", "reason"
            ])
            writer.writeheader()
            for p in self.unresolved:
                writer.writerow({
                    "component": p.ref,
                    "pin": p.pin,
                    "page": p.page + 1,
                    "x": round(p.x, 3),
                    "y": round(p.y, 3),
                    "source_token": p.source_token,
                    "reason": "No sufficiently supported vector-wire attachment found",
                })

    def export_aliases(self, path: str) -> None:
        rows = []
        for key, aliases in sorted(self.cluster_aliases.items()):
            for a in aliases:
                rows.append({
                    "page": a.page + 1,
                    "cluster": key[1],
                    "name": a.name,
                    "source": a.source,
                    "token": a.token,
                    "confidence": round(a.confidence, 3),
                })
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "page", "cluster", "name", "source", "token", "confidence"
            ])
            writer.writeheader()
            writer.writerows(rows)

    def export_visualization(self, path: str) -> None:
        import matplotlib.pyplot as plt

        comp_edges = self.component_edges()
        if not comp_edges:
            # Still create a useful component-only image.
            g = nx.Graph()
            for ref in sorted({p.ref for p in self.pins}):
                g.add_node(ref)
        else:
            g = nx.Graph()
            for row in comp_edges:
                g.add_edge(row["source"], row["target"], net=row["net"])

        plt.figure(figsize=(18, 12))
        if len(g):
            pos = nx.spring_layout(g, seed=42, k=None)
            nx.draw_networkx_nodes(g, pos, node_size=350)
            nx.draw_networkx_edges(g, pos, width=0.7, alpha=0.55)
            nx.draw_networkx_labels(g, pos, font_size=6)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()


# ----------------------------- CLI -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="Path to schematic PDF")
    ap.add_argument("--out-prefix", default="schematic_v3", help="Output file prefix")
    ap.add_argument("--viz", action="store_true", help="Also render component graph overview PNG")
    args = ap.parse_args()

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    extractor = SchematicGraphV3(args.pdf)
    extractor.build_graph()

    extractor.export_graphml(str(prefix.with_suffix(".graphml")))
    extractor.export_json(str(prefix.with_suffix(".json")))
    extractor.export_component_nets(str(prefix.parent / f"{prefix.name}_component_nets.csv"))
    extractor.export_component_edges(str(prefix.parent / f"{prefix.name}_component_edges.csv"))
    extractor.export_unresolved(str(prefix.parent / f"{prefix.name}_unresolved_pins.csv"))
    extractor.export_aliases(str(prefix.parent / f"{prefix.name}_net_aliases.csv"))
    if args.viz:
        extractor.export_visualization(str(prefix.parent / f"{prefix.name}_overview.png"))

    print("=" * 72)
    print("Schematic graph extraction complete")
    print(json.dumps(extractor.summary(), indent=2))
    print("=" * 72)
    print(f"GraphML:              {prefix.with_suffix('.graphml')}")
    print(f"JSON:                 {prefix.with_suffix('.json')}")
    print(f"Component/net CSV:    {prefix.parent / (prefix.name + '_component_nets.csv')}")
    print(f"Component edges CSV:  {prefix.parent / (prefix.name + '_component_edges.csv')}")
    print(f"Unresolved pins CSV:   {prefix.parent / (prefix.name + '_unresolved_pins.csv')}")
    print(f"Net aliases CSV:       {prefix.parent / (prefix.name + '_net_aliases.csv')}")


if __name__ == "__main__":
    main()
