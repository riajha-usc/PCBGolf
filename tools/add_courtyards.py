from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sexpr as sx
from score import BBox, local_layer_bbox, pad_points

TEMPLATE = """\t(fp_rect
\t\t(start {x1:.3f} {y1:.3f})
\t\t(end {x2:.3f} {y2:.3f})
\t\t(stroke
\t\t\t(width 0.05)
\t\t\t(type solid)
\t\t)
\t\t(fill no)
\t\t(layer "F.CrtYd")
\t\t(uuid "{uid}")
\t)
"""


def part_extent(fp: list) -> BBox:
    """The space the part actually occupies: its pads plus its fab outline."""
    box = BBox()
    for pad in sx.find(fp, "pad"):
        box.add_all(pad_points(pad))
    return box.merge(local_layer_bbox(fp, ".Fab"))


def encloses(crt: BBox, part: BBox, eps: float = 1e-6) -> bool:
    return (crt.xmin <= part.xmin + eps and crt.ymin <= part.ymin + eps
            and crt.xmax >= part.xmax - eps and crt.ymax >= part.ymax - eps)


def courtyard_rect(fp: list, margin: float) -> tuple[float, float, float, float] | None:
    """Rectangle to add: the part, grown by margin, unioned with any existing
    courtyard so an undersized one is enlarged rather than contradicted."""
    part = part_extent(fp)
    if part.empty:
        return None
    box = BBox().merge(part)
    box.add(part.xmin - margin, part.ymin - margin)
    box.add(part.xmax + margin, part.ymax + margin)
    box.merge(local_layer_bbox(fp, ".CrtYd"))
    return (box.xmin, box.ymin, box.xmax, box.ymax)


def insert_before_final_paren(text: str, block: str) -> str:
    """Splice a new top-level child in just before the footprint's closing ')'.

    Inserts ahead of that paren's own indentation, so the new block lands at
    one tab and the closing paren keeps its original position.
    """
    idx = text.rstrip().rfind(")")
    if idx < 0:
        raise ValueError("no closing paren")
    while idx > 0 and text[idx - 1] in "\t ":
        idx -= 1
    return text[:idx] + block + text[idx:]


_TOK = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+|\s+')


def node_spans(text: str) -> list[tuple[int, int, int]]:
    """(depth, start, end) for every node; the root expression is depth 1."""
    out, stack = [], []
    for m in _TOK.finditer(text):
        t = m.group()
        if t == "(":
            stack.append(m.start())
        elif t == ")":
            start = stack.pop()
            out.append((len(stack) + 1, start, m.end()))
    return out


def courtyard_drops(text: str, nodes, container) -> list[tuple[int, int]]:
    """Spans of F.CrtYd shapes inside `container` that an fp_rect supersedes.

    --fix-undersized enlarges a courtyard by adding a rectangle around the old
    outline. KiCad then sees overlapping outlines on F.CrtYd and reports the
    courtyard as malformed (self-intersecting). The rectangle alone is the
    correct courtyard, so the older shapes are dropped.
    """
    depth, c0, c1 = container
    crt = []
    for d, s, e in nodes:
        if d != depth + 1 or s < c0 or e > c1:
            continue
        head = text[s:s + 12]
        if not head.startswith("(fp_"):
            continue
        node = sx.loads(text[s:e])
        layer = sx.kid(node, "layer")
        if layer and len(layer) > 1 and str(layer[1]) == "F.CrtYd":
            crt.append(((s, e), sx.name(node)))
    if not any(k == "fp_rect" for _, k in crt) or len(crt) < 2:
        return []
    return [span for span, k in crt if k != "fp_rect"]


def cut(text: str, drops) -> str:
    """Remove spans together with the indentation and newline before each."""
    out, prev = [], 0
    for s, e in sorted(drops):
        j = s
        while j > prev and text[j - 1] in " \t":
            j -= 1
        if j > prev and text[j - 1] == "\n":
            j -= 1
        out.append(text[prev:j])
        prev = e
    out.append(text[prev:])
    return "".join(out)


def dedupe_library_file(text: str):
    nodes = node_spans(text)
    root = next(n for n in nodes if n[0] == 1)
    drops = courtyard_drops(text, nodes, root)
    return cut(text, drops), len(drops), []


def dedupe_board(text: str):
    """Same fix for the footprint copies embedded in the .kicad_pcb."""
    nodes = node_spans(text)
    drops, refs = [], []
    for d, s, e in nodes:
        if d == 2 and text.startswith("(footprint", s):
            got = courtyard_drops(text, nodes, (d, s, e))
            if got:
                drops += got
                m = re.search(r'\(property "Reference" "([^"]+)"', text[s:e])
                refs.append(m.group(1) if m else "?")
    return cut(text, drops), len(drops), refs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pretty", type=Path, nargs="?", default=Path("pcbgolf.pretty"))
    ap.add_argument("--margin", type=float, default=0.25,
                    help="clearance added on each side, mm (default 0.25)")
    ap.add_argument("--write", action="store_true", help="edit files; otherwise dry run")
    ap.add_argument("--fix-undersized", action="store_true",
                    help="also enlarge existing courtyards that do not enclose the part")
    ap.add_argument("--pcb", type=Path, default=Path("pcbgolf.kicad_pcb"),
                    help="board whose embedded footprint copies --dedupe also fixes")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop courtyard shapes superseded by an enclosing fp_rect")
    args = ap.parse_args(argv)

    if args.dedupe:
        total = 0
        for path in sorted(args.pretty.glob("*.kicad_mod")):
            new, n, _ = dedupe_library_file(path.read_text())
            if n:
                sx.loads(new)                       # must still parse
                print(f"  {'WRITE' if args.write else 'would'}  {path.stem:<30} drop {n} old courtyard shape(s)")
                if args.write:
                    path.write_text(new)
                total += n
        if args.pcb.exists():
            new, n, refs = dedupe_board(args.pcb.read_text())
            if n:
                sx.loads(new)
                print(f"  {'WRITE' if args.write else 'would'}  {args.pcb.name:<30} drop {n} shape(s) on {', '.join(refs)}")
                if args.write:
                    args.pcb.write_text(new)
                total += n
        print(f"\n{'dropped' if args.write else 'would drop'} {total} shape(s)")
        return 0

    changed = skipped = 0
    for path in sorted(args.pretty.glob("*.kicad_mod")):
        fp = sx.load(path)
        existing = local_layer_bbox(fp, ".CrtYd")
        part = part_extent(fp)
        if not existing.empty:
            if encloses(existing, part):
                skipped += 1
                continue
            if not args.fix_undersized:
                print(f"  UNDERSIZED  {path.stem:<26} courtyard "
                      f"{existing.w:.2f}x{existing.h:.2f} does not enclose "
                      f"{part.w:.2f}x{part.h:.2f}  (use --fix-undersized)")
                skipped += 1
                continue
        rect = courtyard_rect(fp, args.margin)
        if rect is None:
            print(f"  SKIP  {path.name}: no pads or fab outline to measure")
            continue
        x1, y1, x2, y2 = rect
        print(f"  {'WRITE' if args.write else 'would'}  {path.stem:<30} "
              f"{x1:>8.3f},{y1:>8.3f} .. {x2:>8.3f},{y2:>8.3f}   "
              f"({x2 - x1:.2f} x {y2 - y1:.2f} mm)")
        if args.write:
            block = TEMPLATE.format(x1=x1, y1=y1, x2=x2, y2=y2, uid=uuid.uuid4())
            path.write_text(insert_before_final_paren(path.read_text(), block))
        changed += 1

    verb = "edited" if args.write else "would edit"
    print(f"\n{verb} {changed} footprint(s), {skipped} already had a courtyard")
    if changed and not args.write:
        print("re-run with --write to apply")
    if changed and args.write:
        print("\nNow pull these into the board:")
        print("  PCB editor -> Tools -> Update Footprints from Library... -> Update all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())