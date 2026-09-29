#!/usr/bin/env python3
"""Floorplan + greedy placement for the PCBGolf board.

Run with KiCad's bundled Python -- it needs the pcbnew module:

    KPY=/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3
    $KPY tools/place.py 2>/dev/null               # smallest square board that fits, dry run
    $KPY tools/place.py --write 2>/dev/null       # ...and save it to pcbgolf.kicad_pcb
    $KPY tools/place.py --size 56 --write 2>/dev/null

(2>/dev/null hides wxWidgets debug chatter; the script's own output is stdout.)

Strategy (see the design log for the reasoning):

  * Single-sided. A via costs 50 points ~= 4 mm2 of board at Z ~12.6 mm; a
    0402 moved to the bottom saves ~2.5 mm2 but needs >=1 extra via, and
    opening the bottom side adds its tallest part to Z across the whole board.
  * Connectors are pinned to board edges, mating face outward. Big ICs are
    pinned to regions of the floorplan. Everything else is placed greedily at
    the nearest free spot to the pins it connects to.
  * Decoupling caps (parts touching only plane nets) are placed as soon as
    their IC is down, round-robin over that IC's power pins.
  * Courtyards never overlap; ordinary parts keep GAP between courtyards for
    routing, and fine-pitch ICs keep a HALO for fan-out.

Re-runnable: after any part swap, run it again. It rewrites positions and the
Edge.Cuts outline only -- tracks, vias and zones are not touched (there are none
yet; do not re-run after routing).
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import time

import pcbnew

MM = 1_000_000
ORIGIN = (50.0, 50.0)      # board top-left on the KiCad sheet, mm
PLANES = {"GND", "+12V", "+5V", "+3V3"}
STEP = 0.25                # search grid, mm
GAP = 0.20                 # courtyard-to-courtyard for ordinary parts
EDGE_KEEP = 0.20           # courtyard-to-outline for interior parts
HALO = {"U3": 1.5, "U4": 1.0}   # fan-out ring around fine-pitch ICs
PLUG_GAP = 1.5             # between the vertical OBD-C sockets: plug overmold room
PIN_PREFIX = ("U", "J", "Q", "Y")   # parts whose power pins attract decoupling caps

# Mating face at rotation 0, measured from courtyard overhang + fab outline.
OPENING = {"J1": "left", "J2": "top", "J3": "bottom"}

# Placed in this order. "at" = centre as a fraction of the board; "edge" =
# flush to that edge at a fraction along it, rotated so the mating face points out.
FLOORPLAN = [
    ("U3",  "at",   (0.60, 0.52)),
    ("J1",  "edge", ("left",   0.20)),
    ("J2",  "edge", ("left",   0.80)),
    ("J3",  "edge", ("bottom", 0.30)),
    ("J5 J6 J7 J8", "row", ("top", 0.72)),
    ("U4",  "at",   (0.30, 0.55)),
    ("U1",  "at",   (0.32, 0.14)),
    ("U2",  "at",   (0.46, 0.14)),
    ("SW1", "at",   (0.88, 0.90)),
    ("J4",  "at",   (0.90, 0.52)),
]


# --- geometry -------------------------------------------------------------------

def rel_rect(fp, rot):
    """Courtyard bbox relative to the footprint position at this rotation, mm."""
    fp.SetOrientationDegrees(rot)
    fp.SetPosition(pcbnew.VECTOR2I(0, 0))
    if hasattr(fp, "BuildCourtyardCaches"):
        fp.BuildCourtyardCaches()
    bb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
    return (bb.GetLeft() / MM, bb.GetTop() / MM, bb.GetRight() / MM, bb.GetBottom() / MM)


def opening_side(fp, rot):
    """Side where the courtyard overhangs the pad field most = the mating face."""
    c = rel_rect(fp, rot)
    xs, ys = [], []
    for p in fp.Pads():
        bb = p.GetBoundingBox()
        xs += [bb.GetLeft() / MM, bb.GetRight() / MM]
        ys += [bb.GetTop() / MM, bb.GetBottom() / MM]
    over = {"left": min(xs) - c[0], "right": c[2] - max(xs),
            "top": min(ys) - c[1], "bottom": c[3] - max(ys)}
    return max(over, key=over.get)


def grow(r, d):
    return (r[0] - d, r[1] - d, r[2] + d, r[3] + d)


class Occupancy:
    """Placed rectangles with a uniform grid index for fast overlap tests."""

    def __init__(self, cell=3.0):
        self.cell, self.rects, self.grid = cell, [], collections.defaultdict(list)

    def _cells(self, r):
        c = self.cell
        for gx in range(math.floor(r[0] / c), math.floor(r[2] / c) + 1):
            for gy in range(math.floor(r[1] / c), math.floor(r[3] / c) + 1):
                yield gx, gy

    def free(self, r):
        for k in self._cells(r):
            for i in self.grid.get(k, ()):
                q = self.rects[i]
                if r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]:
                    return False
        return True

    def add(self, r):
        i = len(self.rects)
        self.rects.append(r)
        for k in self._cells(r):
            self.grid[k].append(i)


def spiral(radius, step):
    n = int(radius / step)
    offs = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
            if i * i + j * j <= n * n]
    offs.sort(key=lambda t: (t[0] * t[0] + t[1] * t[1], t[1], t[0]))
    return offs


# --- the placer -----------------------------------------------------------------

_RECT_CACHE = {}


class Placer:
    def __init__(self, board, W, H, offsets):
        self.b, self.W, self.H, self.offs = board, W, H, offsets
        self.fps = {f.GetReference(): f for f in board.GetFootprints()}
        self.occ = Occupancy()
        self.placed = {}                      # ref -> (x, y, rot) in board-local mm
        self.rr = _RECT_CACHE                 # (ref, rot) -> rel courtyard rect
        self.sig = {r: set() for r in self.fps}
        self.planes = {r: set() for r in self.fps}
        self.members = collections.defaultdict(set)
        for r, f in self.fps.items():
            if f.IsFlipped():                 # single-sided: everything on top
                f.Flip(f.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
            for p in f.Pads():
                n = p.GetNetname()
                if not n or n.startswith("unconnected-"):
                    continue
                (self.planes if n in PLANES else self.sig)[r].add(n)
                self.members[n].add(r)
        self.net_cnt = collections.Counter()  # placed pads per signal net
        self.net_sum = collections.defaultdict(lambda: [0.0, 0.0])
        self.rr_ptr = collections.Counter()   # round-robin over IC power pins

    def rect(self, ref, rot):
        k = (ref, rot)
        if k not in self.rr:
            self.rr[k] = rel_rect(self.fps[ref], rot)
        return self.rr[k]

    def pad(self, ref):
        return HALO.get(ref, GAP / 2)

    def inside(self, a, keep):
        return a[0] >= keep and a[1] >= keep and a[2] <= self.W - keep and a[3] <= self.H - keep

    def commit(self, ref, x, y, rot):
        r = self.rect(ref, rot)
        self.occ.add(grow((x + r[0], y + r[1], x + r[2], y + r[3]), self.pad(ref)))
        f = self.fps[ref]
        f.SetOrientationDegrees(rot)
        f.SetPosition(pcbnew.VECTOR2I(int((ORIGIN[0] + x) * MM), int((ORIGIN[1] + y) * MM)))
        self.placed[ref] = (x, y, rot)
        for p in f.Pads():
            n = p.GetNetname()
            if n in self.sig[ref]:
                pos = p.GetPosition()
                self.net_cnt[n] += 1
                self.net_sum[n][0] += pos.x / MM - ORIGIN[0]
                self.net_sum[n][1] += pos.y / MM - ORIGIN[1]

    def fits(self, ref, x, y, rot, keep):
        r = self.rect(ref, rot)
        a = (x + r[0], y + r[1], x + r[2], y + r[3])
        return self.inside(a, keep) and self.occ.free(grow(a, GAP / 2))

    def nearest(self, ref, tx, ty, rots=(0, 90)):
        tx, ty = round(tx / STEP) * STEP, round(ty / STEP) * STEP
        for dx, dy in self.offs:
            for rot in rots:
                if self.fits(ref, tx + dx, ty + dy, rot, EDGE_KEEP):
                    return tx + dx, ty + dy, rot
        return None

    # -- pinned parts --

    def place_at(self, ref, fx, fy):
        spot = self.nearest(ref, fx * self.W, fy * self.H)
        if spot is None:
            raise RuntimeError(f"{ref}: no room near floorplan target")
        self.commit(ref, *spot)

    def facing(self, ref, side):
        if opening_side(self.fps[ref], 0) != OPENING[ref]:
            raise RuntimeError(f"{ref}: rot-0 mating face is not {OPENING[ref]}; recheck OPENING")
        for rot in (0, 90, 180, 270):
            if opening_side(self.fps[ref], rot) == side:
                return rot
        raise RuntimeError(f"{ref}: no rotation faces {side}")

    def place_edge(self, ref, side, frac):
        rot = self.facing(ref, side)
        r = self.rect(ref, rot)
        along = self.H if side in ("left", "right") else self.W
        target = frac * along
        for k in range(0, int(along / STEP) * 2 + 1):
            d = ((k + 1) // 2) * STEP * (1 if k % 2 else -1)
            t = target + d
            if side == "left":
                x, y = -r[0], t
            elif side == "right":
                x, y = self.W - r[2], t
            elif side == "top":
                x, y = t, -r[1]
            else:
                x, y = t, self.H - r[3]
            if self.fits(ref, x, y, rot, 0.0):
                self.commit(ref, x, y, rot)
                return
        raise RuntimeError(f"{ref}: no room on {side} edge")

    def place_row(self, refs, side, frac):
        # narrow dimension along the edge
        rot = min((0, 90), key=lambda q: (self.rect(refs[0], q)[2] - self.rect(refs[0], q)[0]))
        r = self.rect(refs[0], rot)
        w = r[2] - r[0]
        total = len(refs) * w + (len(refs) - 1) * PLUG_GAP
        x0 = frac * self.W - total / 2
        for i, ref in enumerate(refs):
            left = x0 + i * (w + PLUG_GAP)
            x = left - r[0]
            y = -r[1] if side == "top" else self.H - r[3]
            if not self.fits(ref, x, y, rot, 0.0):
                raise RuntimeError(f"{ref}: row position blocked")
            self.commit(ref, x, y, rot)

    # -- everything else --

    def score(self, ref):
        """How strongly this part is pulled toward already-placed parts."""
        if self.sig[ref]:
            return sum(self.net_cnt[n] for n in self.sig[ref])
        return 1000 if self.plane_target(ref, peek=True) else 0

    def plane_target(self, ref, peek=False):
        """Next power pin (round-robin) on the same sheet for a decoupling cap."""
        net = next((n for n in sorted(self.planes[ref]) if n != "GND"), None)
        if net is None:
            return None
        sheet = self.fps[ref].GetSheetname()
        pins = []
        for other in self.placed:
            f = self.fps[other]
            if not other.startswith(PIN_PREFIX) or f.GetSheetname() != sheet:
                continue
            for p in f.Pads():
                if p.GetNetname() == net:
                    pos = p.GetPosition()
                    pins.append((pos.x / MM - ORIGIN[0], pos.y / MM - ORIGIN[1]))
        if not pins:
            return None
        if peek:
            return True
        cx = sum(p[0] for p in pins) / len(pins)
        cy = sum(p[1] for p in pins) / len(pins)
        pins.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        k = (sheet, net)
        pt = pins[self.rr_ptr[k] % len(pins)]
        self.rr_ptr[k] += 1
        return pt

    def target(self, ref):
        if self.sig[ref] and any(self.net_cnt[n] for n in self.sig[ref]):
            sx = sum(self.net_sum[n][0] for n in self.sig[ref])
            sy = sum(self.net_sum[n][1] for n in self.sig[ref])
            c = sum(self.net_cnt[n] for n in self.sig[ref])
            return sx / c, sy / c
        pt = self.plane_target(ref)
        if pt:
            return pt
        same = [self.placed[o] for o in self.placed
                if self.fps[o].GetSheetname() == self.fps[ref].GetSheetname()]
        if same:
            return sum(p[0] for p in same) / len(same), sum(p[1] for p in same) / len(same)
        return self.W / 2, self.H / 2

    def run(self):
        for refs, kind, arg in FLOORPLAN:
            if kind == "at":
                self.place_at(refs, *arg)
            elif kind == "edge":
                self.place_edge(refs, *arg)
            else:
                self.place_row(refs.split(), *arg)
        todo = set(self.fps) - set(self.placed)
        area = {r: (lambda q: (q[2] - q[0]) * (q[3] - q[1]))(self.rect(r, 0)) for r in todo}
        while todo:
            ref = max(todo, key=lambda r: (self.score(r), area[r], r))
            spot = self.nearest(ref, *self.target(ref))
            if spot is None:
                raise RuntimeError(f"{ref}: no free spot on a {self.W}x{self.H} board")
            self.commit(ref, *spot)
            todo.discard(ref)


def attempt(board, W, H, offsets):
    p = Placer(board, W, H, offsets)
    try:
        p.run()
    except RuntimeError as e:
        return None, str(e)
    return (board, p), None


def set_outline(board, W, H):
    for d in list(board.GetDrawings()):
        if d.GetLayer() == pcbnew.Edge_Cuts:
            board.Remove(d)
    r = pcbnew.PCB_SHAPE(board)
    r.SetShape(pcbnew.SHAPE_T_RECT)
    r.SetLayer(pcbnew.Edge_Cuts)
    r.SetStart(pcbnew.VECTOR2I(int(ORIGIN[0] * MM), int(ORIGIN[1] * MM)))
    r.SetEnd(pcbnew.VECTOR2I(int((ORIGIN[0] + W) * MM), int((ORIGIN[1] + H) * MM)))
    r.SetWidth(int(0.05 * MM))
    board.Add(r)


def self_check(p):
    """Independent pairwise courtyard overlap test on the final positions."""
    rects = []
    for ref, (x, y, rot) in p.placed.items():
        r = p.rect(ref, rot)
        rects.append((ref, (x + r[0], y + r[1], x + r[2], y + r[3])))
    bad = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i][1], rects[j][1]
            if a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6:
                bad.append((rects[i][0], rects[j][0]))
    outside = [ref for ref, a in rects if a[0] < -1e-6 or a[1] < -1e-6 or a[2] > p.W + 1e-6 or a[3] > p.H + 1e-6]
    return bad, outside


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcb", default="pcbgolf.kicad_pcb")
    ap.add_argument("--size", type=float, help="force a square board of this side, mm")
    ap.add_argument("--aspect", type=float, default=1.0, help="H / W when searching")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    board = pcbnew.LoadBoard(args.pcb)       # loaded once; every attempt re-places all parts
    area = 0.0
    for f in board.GetFootprints():
        bb = f.GetCourtyard(pcbnew.F_CrtYd).BBox()
        area += (bb.GetWidth() / MM) * (bb.GetHeight() / MM)
    lo = math.ceil(math.sqrt(area / args.aspect))
    hi = lo * 2
    offsets = spiral(math.hypot(hi, hi * args.aspect), STEP)
    print(f"{len(board.GetFootprints())} footprints, {area:.0f} mm2 of courtyard; "
          f"single-sided lower bound {lo} mm square")

    if args.size:
        side = args.size
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            res, err = attempt(board, mid, round(mid * args.aspect, 2), offsets)
            print(f"  {mid} x {round(mid * args.aspect, 2)}: {'fits' if res else 'fails -- ' + err}")
            if res:
                hi = mid
            else:
                lo = mid + 1
        side = lo
    # the board object holds whichever attempt ran last; re-run the winner
    res, err = attempt(board, side, round(side * args.aspect, 2), offsets)
    if not res:
        print(f"no fit at {side}: {err}")
        sys.exit(1)
    board, p = res
    W, H = side, round(side * args.aspect, 2)
    set_outline(board, W, H)
    bad, outside = self_check(p)
    print(f"\nboard {W} x {H} mm; placed {len(p.placed)}/{len(p.fps)}; "
          f"courtyard overlaps {len(bad)}; outside outline {len(outside)}; {time.time() - t0:.0f}s")
    if bad or outside:
        print("  overlaps:", bad[:10], " outside:", outside[:10])
        sys.exit(1)
    if args.write:
        pcbnew.SaveBoard(args.pcb, board)
        print(f"written {args.pcb}. Reopen KiCad to see it; then run DRC.")
    else:
        print("dry run -- re-run with --write to save")


if __name__ == "__main__":
    main()
