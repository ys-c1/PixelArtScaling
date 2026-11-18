#!/usr/bin/env python3

"""
Acceptance benchmark for scalenx.edgescale (edge-directed 2x upscaling).

Generates four test materials (circle, diagonal line, text, checkerboard),
then for each of them reports:

- number of pixels differing from nearest-neighbour 2x output;
- edge continuity metric (fraction of edge pixels belonging to a
  continuous edge chain, i.e. having >= 2 edge pixels in 8-neighbourhood);
- stair-step ("jaggie") count before/after;
- scaling time.

Additionally verifies:

- disabled upscaler (enabled=False) is pixel-identical to nearest neighbour;
- all-zero weights are pixel-identical to nearest neighbour;
- 512x512 -> 1024x1024 scaling takes less than 150 ms for every material.

Before/after PNG pairs for circle and diagonal line are written to
``edgescale_out/`` next to this script.

Run:  python3 test_edgescale.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pypng import list2png
from scalenx.edgescale import edgescale2x

OUT_DIR = Path(__file__).resolve().parent / 'edgescale_out'

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
BLUE = [32, 64, 200]
ORANGE = [240, 180, 40]

PERF_LIMIT_MS = 150.0
QUALITY_SIZE = 128
PERF_SIZE = 512

# ---------------------------------------------------------------
# Test material generators (self-made, small)
# ---------------------------------------------------------------

FONT5x7 = {
    'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'),
    'C': ('01110', '10001', '10000', '10000', '10000', '10001', '01110'),
    'E': ('11111', '10000', '10000', '11110', '10000', '10000', '11111'),
    'L': ('10000', '10000', '10000', '10000', '10000', '10000', '11111'),
    'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'),
}


def make_canvas(size, bg):
    return [[list(bg) for _ in range(size)] for _ in range(size)]


def make_circle(size):
    img = make_canvas(size, WHITE)
    cx = cy = (size - 1) / 2.0
    r = size * 3.0 / 8.0
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                img[y][x] = list(BLACK)
    return img


def make_diagline(size):
    img = make_canvas(size, WHITE)
    for i in range(size):
        img[i][i] = list(BLACK)
    return img


def make_text(size, word='SCALE'):
    img = make_canvas(size, WHITE)
    glyph_w, glyph_h, spacing = 5, 7, 1
    word_w = len(word) * (glyph_w + spacing) - spacing
    zoom = max(1, min((size - 8) // word_w, (size - 8) // glyph_h))
    ox = (size - word_w * zoom) // 2
    oy = (size - glyph_h * zoom) // 2
    for k, ch in enumerate(word):
        glyph = FONT5x7[ch]
        gx = ox + k * (glyph_w + spacing) * zoom
        for gy, bits in enumerate(glyph):
            for gz, bit in enumerate(bits):
                if bit == '1':
                    for dy in range(zoom):
                        for dx in range(zoom):
                            img[oy + gy * zoom + dy][gx + gz * zoom + dx] = list(BLACK)
    return img


def make_checker(size, cell=8):
    img = make_canvas(size, BLUE)
    for y in range(size):
        for x in range(size):
            if (x // cell + y // cell) % 2:
                img[y][x] = list(ORANGE)
    return img


MATERIALS = {
    'circle': make_circle,
    'diagline': make_diagline,
    'text': make_text,
    'checker': make_checker,
}

# ---------------------------------------------------------------
# Reference nearest neighbour and metrics
# ---------------------------------------------------------------


def nearest2x(image):
    """Independent nearest-neighbour 2x implementation."""
    out = []
    for row in image:
        doubled = []
        for px in row:
            doubled.append(list(px))
            doubled.append(list(px))
        out.append([list(px) for px in doubled])
        out.append([list(px) for px in doubled])
    return out


def diff_count(img_a, img_b):
    """Number of pixel positions with different colour."""
    return sum(1 for row_a, row_b in zip(img_a, img_b) for px_a, px_b in zip(row_a, row_b) if px_a != px_b)


def edge_continuity(image):
    """Fraction of edge pixels having >= 2 edge pixels in 8-neighbourhood.

    Edge pixel = pixel differing from its right or below neighbour.
    1.0 = every edge pixel belongs to a continuous chain.
    """
    Y, X = len(image), len(image[0])
    edge = set()
    for y in range(Y):
        row = image[y]
        row_below = image[y + 1] if y + 1 < Y else None
        for x in range(X):
            px = row[x]
            if (x + 1 < X and row[x + 1] != px) or (row_below is not None and row_below[x] != px):
                edge.add((x, y))
    if not edge:
        return 1.0
    good = 0
    for x, y in edge:
        neighbours = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx or dy) and (x + dx, y + dy) in edge:
                    neighbours += 1
        if neighbours >= 2:
            good += 1
    return good / len(edge)


def _dominant_pair(image):
    counter = Counter()
    for row in image:
        for px in row:
            counter[tuple(px)] += 1
    common = counter.most_common(2)
    if len(common) < 2:
        return None
    return {common[0][0], common[1][0]}


def stair_steps(image):
    """Count boundary rows/columns forming visible staircase treads.

    For every image row the leftmost/rightmost *hard* boundary position
    (both sides exactly equal to one of the two dominant region colours)
    is taken; same for topmost/bottommost boundary per column. Within
    each of the four boundary sequences, a stair tread is a maximal flat
    run of identical boundary positions of length >= 2 that is adjacent
    to a jump of >= 2 px - i.e. a full 2x2 output-pixel step, which is
    the minimal step nearest neighbour can produce and exactly what the
    eye reads as a jaggie. The result is the total number of boundary
    rows/columns covered by such treads (jaggie coverage). Edge
    reconstruction replaces 2x2 blocks with unit (1 px) steps or softens
    them with blends, neither of which is counted.
    """
    pair = _dominant_pair(image)
    if pair is None:
        return 0
    Y, X = len(image), len(image[0])

    def hard(px_a, px_b):
        return px_a != px_b and tuple(px_a) in pair and tuple(px_b) in pair

    seqs = []
    for horizontal in (True, False):
        for take_last in (False, True):
            seq = []
            for major in range(Y if horizontal else X):
                hits = []
                if horizontal:
                    row = image[major]
                    for x in range(X - 1):
                        if hard(row[x], row[x + 1]):
                            hits.append(x)
                else:
                    for y in range(Y - 1):
                        if hard(image[y][major], image[y + 1][major]):
                            hits.append(y)
                seq.append((hits[-1] if take_last else hits[0]) if hits else None)
            seqs.append(seq)

    def count_runs(seq):
        # ↓ Compress to (value, length) runs; None breaks a run.
        runs = []
        for value in seq:
            if value is not None and runs and value == runs[-1][0]:
                runs[-1][1] += 1
            elif value is not None:
                runs.append([value, 1])
        total = 0
        for k, (value, length) in enumerate(runs):
            if length < 2:
                continue
            jump_before = abs(value - runs[k - 1][0]) if k else 0
            jump_after = abs(value - runs[k + 1][0]) if k + 1 < len(runs) else 0
            if max(jump_before, jump_after) >= 2:
                total += length
        return total

    return sum(count_runs(seq) for seq in seqs)


# ---------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------


def time_call(func, *args, repeats=3, **kwargs):
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def main():
    failures = []

    print('=' * 96)
    print('EdgeScale acceptance benchmark')
    print('=' * 96)

    # ↓ 1. Quality metrics at QUALITY_SIZE
    print(f'\n[1] Quality metrics, source {QUALITY_SIZE}x{QUALITY_SIZE} -> {2 * QUALITY_SIZE}x{2 * QUALITY_SIZE}')
    header = f'{"material":<10} {"diff vs NN":>12} {"continuity NN":>14} {"continuity ES":>14} {"stairs NN":>10} {"stairs ES":>10} {"time, ms":>10}'
    print(header)
    print('-' * len(header))

    quality_results = {}
    for name, maker in MATERIALS.items():
        src = maker(QUALITY_SIZE)
        ref_nn = nearest2x(src)
        t_nn = time_call(nearest2x, src)
        t_es = time_call(edgescale2x, src)
        out_es = edgescale2x(src)

        assert len(out_es) == 2 * QUALITY_SIZE and len(out_es[0]) == 2 * QUALITY_SIZE, 'output size mismatch'

        dcnt = diff_count(out_es, ref_nn)
        cont_nn = edge_continuity(ref_nn)
        cont_es = edge_continuity(out_es)
        stairs_nn = stair_steps(ref_nn)
        stairs_es = stair_steps(out_es)
        quality_results[name] = (stairs_nn, stairs_es)

        print(f'{name:<10} {dcnt:>12} {cont_nn:>14.3f} {cont_es:>14.3f} {stairs_nn:>10} {stairs_es:>10} {t_es:>10.1f}')

    # ↓ 2. Jaggie reduction check on circle and diagonal line
    print('\n[2] Stair-step reduction (must drop clearly on circle and diagline)')
    for name in ('circle', 'diagline'):
        stairs_nn, stairs_es = quality_results[name]
        ratio = stairs_es / stairs_nn if stairs_nn else 0.0
        status = 'OK' if stairs_es <= 0.5 * stairs_nn else 'FAIL'
        print(f'    {name:<10} stairs {stairs_nn} -> {stairs_es}  ({ratio:.0%} of NN)  [{status}]')
        if stairs_es > 0.5 * stairs_nn:
            failures.append(f'stair steps not reduced enough on {name}: {stairs_nn} -> {stairs_es}')

    # ↓ 3. Disabled / zero-weight identity with nearest neighbour
    print('\n[3] Identity with nearest neighbour')
    for name, maker in MATERIALS.items():
        src = maker(QUALITY_SIZE)
        ref_nn = nearest2x(src)
        out_disabled = edgescale2x(src, enabled=False)
        out_zero = edgescale2x(src, weights={'edge': 0.0, 'diagonal': 0.0, 'thin': 0.0})
        ok_disabled = out_disabled == ref_nn
        ok_zero = out_zero == ref_nn
        print(f'    {name:<10} enabled=False == NN: {ok_disabled}   zero weights == NN: {ok_zero}')
        if not ok_disabled:
            failures.append(f'enabled=False differs from NN on {name}')
        if not ok_zero:
            failures.append(f'zero weights differ from NN on {name}')

    # ↓ 4. Configurability smoke check: custom weights, smoothing off
    print('\n[4] Configurability (custom weights, diagonal smoothing off)')
    src = make_circle(QUALITY_SIZE)
    out_custom = edgescale2x(src, weights={'edge': 0.5, 'diagonal': 0.25}, smooth_diagonals=False)
    dcnt = diff_count(out_custom, nearest2x(src))
    print(f'    circle, edge=0.5 diagonal=0.25 smooth_diagonals=False: diff vs NN = {dcnt} px')
    if dcnt == 0:
        failures.append('custom weights had no effect on circle')

    # ↓ 5. Performance at PERF_SIZE
    print(f'\n[5] Performance, source {PERF_SIZE}x{PERF_SIZE} -> {2 * PERF_SIZE}x{2 * PERF_SIZE} (limit {PERF_LIMIT_MS:.0f} ms)')
    worst = 0.0
    for name, maker in MATERIALS.items():
        src = maker(PERF_SIZE)
        t_es = time_call(edgescale2x, src)
        worst = max(worst, t_es)
        status = 'OK' if t_es <= PERF_LIMIT_MS else 'FAIL'
        print(f'    {name:<10} {t_es:>8.1f} ms  [{status}]')
        if t_es > PERF_LIMIT_MS:
            failures.append(f'too slow on {name}: {t_es:.1f} ms > {PERF_LIMIT_MS:.0f} ms')
    print(f'    worst case: {worst:.1f} ms')

    # ↓ 6. Before/after PNG pairs for circle and diagonal line
    OUT_DIR.mkdir(exist_ok=True)
    for name in ('circle', 'diagline'):
        src = MATERIALS[name](QUALITY_SIZE)
        list2png(str(OUT_DIR / f'{name}_nn.png'), nearest2x(src), {'bitdepth': 8})
        list2png(str(OUT_DIR / f'{name}_edgescale.png'), edgescale2x(src), {'bitdepth': 8})
    print(f'\n[6] Before/after PNG pairs written to {OUT_DIR}/ (circle, diagline; *_nn vs *_edgescale)')

    print('\n' + '=' * 96)
    if failures:
        print('RESULT: FAIL')
        for failure in failures:
            print('  -', failure)
        return 1
    print('RESULT: ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
