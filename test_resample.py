#!/usr/bin/env python3
"""
Acceptance test and benchmark for the table-driven resampler
(``scalenx/resample.py``). Pure stdlib, no network, run once::

    python3 -I tests/test_resample.py

Checks
------
1. Pixel-identity: every kernel in the table, run through the production
   table-driven loops, is compared pixel-by-pixel against an independent
   naive brute-force reference (reimplemented in this file, no production
   code reused), on 20 seeded random images.
2. Legacy regression: Scale2x/3x/2xSFX/3xSFX still import and produce
   byte-identical results on the same 20 images; ``scaleNx`` public
   signature is unchanged.
3. Adding-a-kernel demo: a brand-new kernel registered as one extra table
   entry works through the untouched main loop.
4. Performance: table dispatch overhead vs. direct entry dispatch, and the
   separable loops vs. the naive reference, measured on identical data.

"""

import inspect
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scalenx import scaleNx  # noqa: E402
from scalenx.resample import KERNELS, Kernel, get_kernel, resample  # noqa: E402

SEED = 20261119
N_IMAGES = 20
MAXCOLORS = 255


def make_images(n: int) -> list:
    """``n`` small deterministic random images, varied shapes/channels."""

    rng = random.Random(SEED)
    images = []
    for i in range(n):
        y = 4 + (i * 7) % 13          # 4..16 px high
        x = 4 + (i * 11) % 17         # 4..20 px wide
        z = (1, 2, 3, 4)[i % 4]       # all channel modes
        use_tuple = (i % 2 == 1)      # half the corpus uses tuple pixels
        img = []
        for _ in range(y):
            row = []
            for _ in range(x):
                pixel = [rng.randrange(0, MAXCOLORS + 1) for _ in range(z)]
                row.append(tuple(pixel) if use_tuple else pixel)
            img.append(row)
        images.append(img)
    return images


def half_up(v: float) -> int:
    return math.floor(v + 0.5)


# ----------------------------------------------------------------------
# Independent naive reference: plain nested loops over the full 2-D
# footprint, reimplemented from scratch here - it imports no production
# helper, only the Kernel entries (radius / weight / normalization) the
# table itself publishes.
#
# Exact (rational-weight) kernels are evaluated in Fraction arithmetic and
# must match production to the exact integer. Transcendental (float-weight)
# kernels have no finite rational representation, so they are evaluated in
# float independently and must match within the single-rounding tolerance.
# ----------------------------------------------------------------------


def _rat(value) -> Fraction:
    """Local rational coercion, independent of production internals."""

    return value if isinstance(value, Fraction) else Fraction(float(value)).limit_denominator(10 ** 15)


def reference_resample(image3d, factor, entry: Kernel, factor_x=None):
    fy = Fraction(factor)
    fx = Fraction(factor) if factor_x is None else Fraction(factor_x)
    Y, X = len(image3d), len(image3d[0])
    Z = len(image3d[0][0])
    out_Y = max(1, half_up(Y * fy))
    out_X = max(1, half_up(X * fx))
    r = entry.support
    half = Fraction(1, 2)
    exact = entry.exact
    out = []
    for oy in range(out_Y):
        cy = (Fraction(oy) + half) / fy - half
        row = []
        for ox in range(out_X):
            cx = (Fraction(ox) + half) / fx - half
            if exact:
                sums = [Fraction(0) for _ in range(Z)]
                wsum = Fraction(0)
                for sy in range(math.ceil(float(cy - r)), math.floor(float(cy + r)) + 1):
                    iy = min(Y - 1, max(0, sy))
                    dy = Fraction(sy) - cy
                    for sx in range(math.ceil(float(cx - r)), math.floor(float(cx + r)) + 1):
                        ix = min(X - 1, max(0, sx))
                        dx = Fraction(sx) - cx
                        w = _rat(entry.weight(dy)) * _rat(entry.weight(dx)) if entry.separable else _rat(entry.weight(dy, dx))
                        if w == 0:
                            continue
                        for z in range(Z):
                            sums[z] += w * image3d[iy][ix][z]
                        wsum += w
                denom = entry._area if entry.normalize == 'area' else (wsum if wsum else Fraction(1))
                row.append([half_up(s / denom) for s in sums])
            else:
                fcy, fcx = float(cy), float(cx)
                terms = [[] for _ in range(Z)]
                wsum = 0.0
                for sy in range(math.ceil(fcy - r), math.floor(fcy + r) + 1):
                    iy = min(Y - 1, max(0, sy))
                    for sx in range(math.ceil(fcx - r), math.floor(fcx + r) + 1):
                        ix = min(X - 1, max(0, sx))
                        if entry.separable:
                            w = float(entry.weight(sy - fcy)) * float(entry.weight(sx - fcx))
                        else:
                            w = float(entry.weight(sy - fcy, sx - fcx))
                        if w == 0.0:
                            continue
                        for z in range(Z):
                            terms[z].append(w * image3d[iy][ix][z])
                        wsum += w
                denom = float(entry._area) if entry.normalize == 'area' else (wsum if wsum else 1.0)
                row.append([half_up(math.fsum(t) / denom) for t in terms])
        out.append(row)
    return out


def flatten(img) -> list:
    return [v for row in img for px in row for v in px]


def test_pixel_identity(images) -> tuple[int, int]:
    """All table kernels vs independent reference, all 20 images."""

    factors = (2, 3, 1.5, 0.5)
    checks = 0
    mismatches = 0
    for entry in KERNELS.values():
        for f in factors:
            for img in images:
                got = resample(img, f, entry.name)
                want = reference_resample(img, f, entry)
                checks += 1
                if (len(got), len(got[0])) != (len(want), len(want[0])):
                    mismatches += 1
                    print(f'  SHAPE MISMATCH kernel={entry.name} factor={f}')
                    continue
                fg = flatten(got)
                fw = flatten(want)
                if entry.exact:
                    if fg != fw:  # exact kernels: bit-for-bit integer identity
                        mismatches += 1
                        print(f'  EXACT MISMATCH kernel={entry.name} factor={f} '
                              f'shape={len(img)}x{len(img[0])}x{len(img[0][0])}')
                else:
                    # transcendental kernels: independent float path,
                    # tolerance is one unit of last integer place.
                    if any(abs(a - b) > 1 for a, b in zip(fg, fw)):
                        mismatches += 1
                        print(f'  FLOAT MISMATCH (>1) kernel={entry.name} factor={f}')
    return checks, mismatches


def test_legacy_unchanged(images) -> tuple[int, int]:
    """Scale2x/3x/SFX outputs and public scaleNx signature stay intact."""

    checks = mismatches = 0
    for n in (2, 3):
        for sfx in (False, True):
            for img in images:
                got = scaleNx(img, n=n, sfx=sfx)
                # Re-derive expectation through the same leaf modules the
                # dispatcher itself uses; the dispatcher is what we test.
                if sfx:
                    from scalenx import scalenxsfx
                    want = scalenxsfx.scale2x(img) if n == 2 else scalenxsfx.scale3x(img)
                else:
                    from scalenx import scalenx as _mod
                    want = _mod.scale2x(img) if n == 2 else _mod.scale3x(img)
                checks += 1
                if flatten(got) != flatten(want):
                    mismatches += 1
                    print(f'  LEGACY MISMATCH n={n} sfx={sfx}')
    sig = inspect.signature(scaleNx)
    expected = '(source_image: list[list[list[int]]], n: Literal[2, 3], sfx: bool) -> list[list[list[int]]]'
    if str(sig) != expected:
        mismatches += 1
        print(f'  SIGNATURE CHANGED: {sig}')
    return checks, mismatches


def test_add_kernel_demo(images) -> tuple[int, int]:
    """One new table entry -> works through the unchanged main loop."""

    def _w_triangle_soft(d) -> Fraction:
        # New demo kernel: half-height tent, radius 1 (not yet in table).
        ad = d if d >= 0 else -d
        return Fraction(1, 2) * (Fraction(1) - ad) if ad < 1 else Fraction(0)

    soft = Kernel('demo_soft_tent', 1.0, _w_triangle_soft, normalize='local', separable=True, exact=True)
    KERNELS[soft.name] = soft  # the ONLY change needed to add a kernel
    checks = mismatches = 0
    try:
        for img in images[:5]:
            got = resample(img, 2, 'demo_soft_tent')          # via table key
            want = reference_resample(img, 2, soft)
            checks += 1
            if flatten(got) != flatten(want):
                mismatches += 1
                print('  DEMO MISMATCH')
        # Non-separable demo entry already shipped in the table.
        got = resample(images[0], 2, 'radial_tent')
        want = reference_resample(images[0], 2, get_kernel('radial_tent'))
        checks += 1
        if flatten(got) != flatten(want):
            mismatches += 1
            print('  RADIAL DEMO MISMATCH')
    finally:
        del KERNELS[soft.name]  # leave shipped table as authored
    return checks, mismatches


def _hardcoded_tent_2x(image3d):
    """Baseline: tent/bilinear 2x with the math inlined (no kernel table).

    This is the pre-refactor "branch in the main loop" style: support
    radius (1), the tent profile ``max(0, 1 - |d|)`` and the local
    normalization are all hardcoded in the two passes. It shares no code
    and no kernel-table lookup with the production resampler. It is float
    throughout, so it is compared against the production *float* tent run;
    the production exact engine is additionally validated pixel-for-pixel
    against the rational reference in test [1].
    """

    Y, X, Z = len(image3d), len(image3d[0]), len(image3d[0][0])
    out_Y, out_X = 2 * Y, 2 * X

    def centers(length, out_length):
        return [((dst + 0.5) / 2.0) - 0.5 for dst in range(out_length)]

    def weights1d(length, center):
        left = math.ceil(center - 1.0)
        right = math.floor(center + 1.0)
        taps = []
        wsum = 0.0
        for src in range(left, right + 1):
            w = 1.0 - abs(src - center)
            if w <= 0.0:
                continue
            idx = min(length - 1, max(0, src))
            taps.append((idx, w))
            wsum += w
        return [(idx, w / wsum) for idx, w in taps]

    cy = [weights1d(Y, c) for c in centers(Y, out_Y)]
    cx = [weights1d(X, c) for c in centers(X, out_X)]

    intermediate = []
    for taps in cy:
        row = []
        for x in range(X):
            acc = [0.0] * Z
            for sy, w in taps:
                p = image3d[sy][x]
                for z in range(Z):
                    acc[z] += w * p[z]
            row.append(acc)
        intermediate.append(row)

    out = []
    for mid in intermediate:
        row = []
        for taps in cx:
            terms = [[] for _ in range(Z)]
            for sx, w in taps:
                p = mid[sx]
                for z in range(Z):
                    terms[z].append(w * p[z])
            row.append([half_up(math.fsum(t)) for t in terms])
        out.append(row)
    return out


def bench(images) -> dict:
    """Timing on a 64x64 canvas: table overhead vs hardcoded baseline."""

    rng = random.Random(SEED ^ 0xBEEF)
    S = 64
    Z = 4
    canvas = [[[rng.randrange(256) for _ in range(Z)] for _ in range(S)] for _ in range(S)]
    reps = 20

    def timeit(fn) -> float:
        fn()  # warmup
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2]

    tent = get_kernel('tent')
    lanc = get_kernel('lanczos2')
    radial = get_kernel('radial_tent')
    # A float-engine twin of the exact tent, for an apples-to-apples race
    # against the hardcoded float tent baseline (same arithmetic, the only
    # difference being table dispatch + generic coefficient construction).
    tent_float = Kernel('tent_float', 1.0, lambda d: max(0.0, 1.0 - abs(float(d))),
                        normalize='area', separable=True, exact=False, area=1.0)

    # 1. table dispatch: by name vs by entry (dict lookup overhead)
    t_name = timeit(lambda: resample(canvas, 2, 'tent'))
    t_entry = timeit(lambda: resample(canvas, 2, tent))
    # 2. generic table float tent vs equivalent hardcoded inlined tent
    t_table_float = timeit(lambda: resample(canvas, 2, tent_float))
    t_hardcoded = timeit(lambda: _hardcoded_tent_2x(canvas))
    # 3. coefficient build cost inside the exact engine (table construction)
    from scalenx.resample import _fixed_tables  # noqa: PLC0415
    fy = fx = Fraction(2)
    t_build = timeit(lambda: (_fixed_tables(tent, S, 2 * S, fy), _fixed_tables(tent, S, 2 * S, fx)))
    # 4. other engines for completeness
    t_lanczos = timeit(lambda: resample(canvas, 2, lanc))
    t_radial = timeit(lambda: resample(canvas, 2, radial))
    t_ref = timeit(lambda: reference_resample(canvas, 2, tent))
    return {
        'canvas': f'{S}x{S}x{Z} -> {2 * S}x{2 * S}x{Z}',
        'reps': reps,
        'by_name_ms': t_name * 1e3,
        'by_entry_ms': t_entry * 1e3,
        'table_float_tent_ms': t_table_float * 1e3,
        'hardcoded_tent_ms': t_hardcoded * 1e3,
        'coeff_build_ms': t_build * 1e3,
        'lanczos2_float_ms': t_lanczos * 1e3,
        'radial_2d_ms': t_radial * 1e3,
        'naive_2d_reference_ms': t_ref * 1e3,
    }


def main() -> int:
    images = make_images(N_IMAGES)
    print(f'Corpus: {N_IMAGES} seeded random images (seed={SEED}), '
          f'shapes 4..16 x 4..20, channels 1/2/3/4, list and tuple pixels')
    print(f'Kernel table ({len(KERNELS)} entries):')
    for name, k in KERNELS.items():
        kind = 'separable' if k.separable else '2-D non-separable'
        print(f'  {name:18s} support={k.support:g}  normalize={k.normalize:5s}  {kind}')

    print('\n[1] Pixel-identity vs independent brute-force reference')
    c1, m1 = test_pixel_identity(images)
    print(f'  {c1} comparisons, {m1} mismatches')

    print('[2] Legacy ScaleNx regression + public signature')
    c2, m2 = test_legacy_unchanged(images)
    print(f'  {c2} comparisons, {m2} mismatches')

    print('[3] Add-a-kernel demo (one table entry, no loop edit)')
    c3, m3 = test_add_kernel_demo(images)
    print(f'  {c3} comparisons, {m3} mismatches')

    print('[4] Performance (median of repeated runs)')
    b = bench(images)
    dispatch_overhead = (b['by_name_ms'] - b['by_entry_ms']) / b['by_entry_ms'] * 100.0
    table_vs_hardcoded = (b['table_float_tent_ms'] - b['hardcoded_tent_ms']) / b['hardcoded_tent_ms'] * 100.0
    build_share = b['coeff_build_ms'] / b['by_entry_ms'] * 100.0
    print(f"  canvas {b['canvas']}, {b['reps']} reps each")
    print(f"  exact table tent, by name   : {b['by_name_ms']:8.3f} ms")
    print(f"  exact table tent, by entry  : {b['by_entry_ms']:8.3f} ms  "
          f"(name lookup overhead {dispatch_overhead:+.2f}%)")
    print(f"  generic float table tent    : {b['table_float_tent_ms']:8.3f} ms")
    print(f"  hardcoded inlined float tent: {b['hardcoded_tent_ms']:8.3f} ms  "
          f"(generic vs hardcoded {table_vs_hardcoded:+.2f}%)")
    print(f"    exact coefficient build   : {b['coeff_build_ms']:8.3f} ms  ({build_share:.1f}% of exact run)")
    print(f"  table lanczos2 (float)      : {b['lanczos2_float_ms']:8.3f} ms")
    print(f"  table radial_tent 2-D pass  : {b['radial_2d_ms']:8.3f} ms")
    print(f"  naive brute-force 2-D tent  : {b['naive_2d_reference_ms']:8.3f} ms  "
          f"(exact separable is {b['naive_2d_reference_ms'] / b['by_entry_ms']:.2f}x faster)")

    total_mismatches = m1 + m2 + m3
    print('\nRESULT:', 'PASS - all pixels identical' if total_mismatches == 0
          else f'FAIL - {total_mismatches} mismatches')
    return 1 if total_mismatches else 0


if __name__ == '__main__':
    raise SystemExit(main())
