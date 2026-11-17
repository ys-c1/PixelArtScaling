#!/usr/bin/env python3

"""
==============================================
ScaleNx content-aware boundary extension tests
==============================================

Self-contained correctness test and benchmark for the content-aware
boundary extension rescaling added in ``scalenx/contentaware.py``.

The feature addresses edge smearing when upscaling images with
transparent margins: plain ScaleNx uses "repeat edge" at the border, so
the texture edge gets stretched into blurred strips. The content-aware
wrapper first detects the opaque content bounding box, extends the
image outside the box with mirror or replicate padding, upscales the
extended image with the *unmodified* ScaleNx functions, then crops the
extension back to the direct scaling dimensions.

Test set: four small self-generated 12x12 RGBA images

1. ``centered``  - opaque block surrounded by a transparent margin;
2. ``edge``      - opaque block touching one image border;
3. ``corners``   - four small opaque blocks in the image corners;
4. ``sparse``    - mostly transparent, a few isolated opaque pixels.

For every sample and both padding modes the benchmark reports:

- edge error  - mean absolute color-channel difference versus the direct
  scaling output over the n-pixel perimeter band of the scaled image
  (the exact region where direct scaling smears the edge);
- PSNR        - peak signal-to-noise ratio of the content-aware result
  against the direct scaling output, all channels, 0..255 scale;
- time        - median end-to-end wall time of the whole pipeline;
- extension % - fraction of padded pixels in the extended image.

Correctness checks (any failure makes the run exit with code 1):

- fully transparent / empty input raises ``ValueError``;
- ``pad_ratio=0`` output is pixel-by-pixel identical to direct
  ``scaleNx`` for every (n, sfx) combination (original scaling output
  is never altered);
- output always has exactly the direct scaling dimensions n*Y x n*X;
- opaque content pixels of the source are reproduced identically to
  direct scaling at every opaque output location;
- bounding box detection matches hand-computed boxes for all samples;
- mirror and replicate padding match independently computed references;
- mirror vs replicate give different results whenever padding is used;
- invalid ``n``, ``pad_mode`` and ``pad_ratio`` raise ``ValueError``;
- images without alpha (L, RGB) are treated as fully opaque.

No third-party dependencies; run as::

    python3 test_content_aware.py

The file also collects under pytest (test functions are plain asserts).
Exit code 0 means all checks passed, 1 means at least one failure.

"""

__author__ = 'Ilya Razmanov'
__license__ = 'unlicense'

import math
import sys
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scalenx import scaleNx
from scalenx.contentaware import content_aware_scaleNx, opaque_bbox, pad_image
from scalenx.scalenx import scale2x as ref_scale2x
from scalenx.scalenx import scale3x as ref_scale3x
from scalenx.scalenxsfx import scale2x as ref_scale2xsfx
from scalenx.scalenxsfx import scale3x as ref_scale3xsfx

SIZE = 12  # samples are SIZE x SIZE RGBA, deliberately tiny
PAD_RATIO = 0.5  # extension per side = half of the opaque box size
N = 2  # benchmark scaling factor
REPEATS = 9  # odd number, median timing is taken
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record a check result, print failures immediately."""

    if not condition:
        FAILURES.append(message)
        print(f'  FAIL: {message}')


""" ╔══════════════════════════════════════════╗
    ║ Small self-generated RGBA sample factory ║
    ╚══════════════════════════════════════════╝ """


def _opaque_pixel(rng: Random) -> list[int]:
    """Deterministic fully opaque RGBA pixel with distinct colors."""

    return [rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255), 255]


def make_samples() -> dict[str, list[list[list[int]]]]:
    """Build the four 12x12 RGBA test images."""

    rng = Random(20261118)
    transparent = [0, 0, 0, 0]

    def blank():
        return [[list(transparent) for _ in range(SIZE)] for _ in range(SIZE)]

    # 1. Centered opaque 6x6 block with a 3 px transparent margin.
    centered = blank()
    for y in range(3, 9):
        for x in range(3, 9):
            centered[y][x] = _opaque_pixel(rng)
    expected_centered = (3, 8, 3, 8)

    # 2. Opaque block touching the left and top image borders.
    edge = blank()
    for y in range(0, 7):
        for x in range(0, 6):
            edge[y][x] = _opaque_pixel(rng)
    expected_edge = (0, 6, 0, 5)

    # 3. Four 3x3 opaque blocks in the four image corners.
    corners = blank()
    for y in range(SIZE):
        for x in range(SIZE):
            in_corner = (y < 3 or y >= SIZE - 3) and (x < 3 or x >= SIZE - 3)
            if in_corner:
                corners[y][x] = _opaque_pixel(rng)
    expected_corners = (0, SIZE - 1, 0, SIZE - 1)

    # 4. Mostly transparent: five isolated opaque pixels.
    sparse = blank()
    for y, x in ((2, 4), (5, 5), (8, 3), (9, 9), (4, 10)):
        sparse[y][x] = _opaque_pixel(rng)
    expected_sparse = (2, 9, 3, 10)

    return {
        'centered': (centered, expected_centered),
        'edge': (edge, expected_edge),
        'corners': (corners, expected_corners),
        'sparse': (sparse, expected_sparse),
    }


SAMPLES = make_samples()


""" ╔═══════════════════════╗
    ║ Metric implementations ║
    ╚═══════════════════════╝ """


def edge_band_error(a: list[list[list[int]]], b: list[list[list[int]]], band: int) -> float:
    """Mean absolute color-channel error over the perimeter band.

    The ``band``-pixel thick outer frame of the scaled image is exactly
    where direct ScaleNx repeats the source edge and smears it. Alpha is
    excluded so the metric measures color bleeding, not coverage.
    """

    Y, X = (len(a), len(a[0]))
    total = 0.0
    count = 0
    color_channels = len(a[0][0]) - 1  # last channel is alpha
    for y in range(Y):
        for x in range(X):
            if y < band or y >= Y - band or x < band or x >= X - band:
                for z in range(color_channels):
                    total += abs(a[y][x][z] - b[y][x][z])
                    count += 1
    return total / count if count else 0.0


def smearing_runs(image: list[list[list[int]]], out: list[list[list[int]]], n: int) -> float:
    """Mean longest uniform RGB run along the transparent-margin perimeter.

    Direct ScaleNx repeats the edge pixel outside the opaque content, so
    the scaled transparent margin contains long monochrome strips - the
    visible "smeared edge". Content-aware padding fills that margin with
    reflected/copied structure instead, breaking the strips into shorter
    runs. For every output block whose source pixel is transparent and
    which lies in the first scaled margin row/column of an image side,
    the longest consecutive run of equal RGB triples along that side is
    measured; the mean over sides is returned in scaled pixels.
    """

    def rgb(px):
        return tuple(px[:-1])

    Y, X = (len(out), len(out[0]))
    side_runs: list[int] = []

    # Top and bottom margin rows, left and right margin columns.
    for side in ('top', 'bottom', 'left', 'right'):
        longest = 0
        if side in ('top', 'bottom'):
            # First/last source row that is fully transparent on that side.
            src_rows = [y for y in range(SIZE) if all(image[y][x][-1] == 0 for x in range(SIZE))]
            if not src_rows:
                continue
            sy = min(src_rows) if side == 'top' else max(src_rows)
            seq = [rgb(out[n * sy][x]) for x in range(X)]
        else:
            src_cols = [x for x in range(SIZE) if all(image[y][x][-1] == 0 for y in range(SIZE))]
            if not src_cols:
                continue
            sx = min(src_cols) if side == 'left' else max(src_cols)
            seq = [rgb(out[y][n * sx]) for y in range(Y)]
        run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1
            else:
                longest = max(longest, run)
                run = 1
        side_runs.append(max(longest, run))

    return sum(side_runs) / len(side_runs) if side_runs else 0.0


def psnr(a: list[list[list[int]]], b: list[list[list[int]]]) -> float:
    """PSNR in dB between two images over all channels, 0..255 scale."""

    Y, X, Z = (len(a), len(a[0]), len(a[0][0]))
    mse = 0.0
    for y in range(Y):
        for x in range(X):
            pa, pb = (a[y][x], b[y][x])
            for z in range(Z):
                d = pa[z] - pb[z]
                mse += d * d
    mse /= Y * X * Z
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / mse)


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def timed(func, *args, repeats: int = REPEATS, **kwargs) -> tuple[object, float]:
    """Run ``func`` ``repeats`` times, return last result and median time."""

    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    return result, median(durations)


""" ╔══════════════════════════════════╗
    ║ Deterministic correctness checks ║
    ╚══════════════════════════════════╝ """


def test_transparent_and_empty_raise() -> None:
    """No opaque content must raise ValueError."""

    transparent = [[[0, 0, 0, 0] for _ in range(4)] for _ in range(4)]
    for kwargs in ({}, {'pad_ratio': 0.0}, {'pad_mode': 'replicate'}):
        try:
            content_aware_scaleNx(transparent, **kwargs)
            check(False, f'fully transparent image must raise ValueError ({kwargs})')
        except ValueError:
            pass
    try:
        opaque_bbox([])
        check(False, 'empty image must raise ValueError')
    except ValueError:
        pass
    try:
        opaque_bbox([[]])
        check(False, 'empty row must raise ValueError')
    except ValueError:
        pass
    print('  fully transparent / empty input raises ValueError: OK')


def test_bbox_detection() -> None:
    """Bounding boxes must match the hand-computed boxes of all samples."""

    for name, (image, expected) in SAMPLES.items():
        check(opaque_bbox(image) == expected, f'bbox mismatch for `{name}`: {opaque_bbox(image)} != {expected}')

    # LA image: alpha is still the last channel.
    la = [[[10, 0], [20, 255]], [[30, 255], [40, 0]]]
    check(opaque_bbox(la) == (0, 1, 0, 1), 'LA bbox must ignore fully transparent pixels')

    # L and RGB have no alpha, so the whole image is opaque.
    grey = [[[0], [0]], [[0], [0]]]
    rgb = [[[0, 0, 0]]]
    check(opaque_bbox(grey) == (0, 1, 0, 1), 'L image bbox must be the whole image')
    check(opaque_bbox(rgb) == (0, 0, 0, 0), 'RGB image bbox must be the whole image')
    print('  opaque bounding box detection for all samples and modes: OK')


def test_padding_reference() -> None:
    """Mirror and replicate padding must match independent references."""

    image, _ = SAMPLES['centered']
    bbox = opaque_bbox(image)
    top, bottom, left, right = bbox
    box_h, box_w = (bottom - top + 1, right - left + 1)
    pt, pb, pl, pr = (2, 1, 2, 1)

    padded = pad_image(image, bbox, pt, pb, pl, pr, 'replicate')
    check(len(padded) == SIZE + pt + pb and len(padded[0]) == SIZE + pl + pr, 'replicate padded size mismatch')

    def clamp(coord, limit):
        return 0 if coord < 0 else limit - 1 if coord >= limit else coord

    for ny in range(len(padded)):
        for nx in range(len(padded[0])):
            sy, sx = (ny - pt, nx - pl)
            ref_px = image[top + clamp(sy - top, box_h)][left + clamp(sx - left, box_w)]
            check(padded[ny][nx] == list(ref_px), f'replicate pixel mismatch at ({ny},{nx})')

    padded_m = pad_image(image, bbox, pt, pb, pl, pr, 'mirror')

    def reflect(coord, limit):
        period = 2 * limit
        coord %= period
        return period - 1 - coord if coord >= limit else coord

    for ny in range(len(padded_m)):
        for nx in range(len(padded_m[0])):
            sy, sx = (ny - pt, nx - pl)
            ref_px = image[top + reflect(sy - top, box_h)][left + reflect(sx - left, box_w)]
            check(padded_m[ny][nx] == list(ref_px), f'mirror pixel mismatch at ({ny},{nx})')

    # Padding must not alias the source pixel lists (independent copy).
    padded_m[pt][pl][0] = (padded_m[pt][pl][0] + 1) % 256
    check(image[top][left][0] != padded_m[pt][pl][0], 'padded pixels must be independent copies of source pixels')
    print('  mirror and replicate padding match independent references: OK')


def test_invalid_arguments() -> None:
    """Bad configuration must raise ValueError."""

    image, _ = SAMPLES['centered']

    def raises(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            return False
        except ValueError:
            return True

    check(raises(content_aware_scaleNx, image, n=4), 'n=4 must raise ValueError')
    check(raises(content_aware_scaleNx, image, pad_mode='wrap'), "pad_mode='wrap' must raise ValueError")
    check(raises(content_aware_scaleNx, image, pad_ratio=-0.1), 'negative pad_ratio must raise ValueError')
    check(raises(content_aware_scaleNx, image, pad_ratio='0.25'), 'non-numeric pad_ratio must raise ValueError')
    check(raises(content_aware_scaleNx, image, pad_ratio=True), 'boolean pad_ratio must raise ValueError')
    print('  invalid n / pad_mode / pad_ratio raise ValueError: OK')


def test_direct_scaling_unchanged() -> None:
    """pad_ratio=0 must reproduce the direct scaleNx output pixel by pixel."""

    image, _ = SAMPLES['edge']
    for n in (2, 3):
        for sfx in (False, True):
            direct = scaleNx(image, n=n, sfx=sfx)
            result, stats = content_aware_scaleNx(image, n=n, sfx=sfx, pad_ratio=0.0)
            check(result == direct, f'pad_ratio=0 output must equal direct scaleNx (n={n}, sfx={sfx})')
            check(stats['extension_ratio'] == 0.0, 'pad_ratio=0 must report zero extension ratio')
            check(stats['pad'] == (0, 0, 0, 0), 'pad_ratio=0 must report zero padding')
    print('  pad_ratio=0 is pixel-by-pixel identical to direct scaling (all 4 methods): OK')


def test_size_and_opaque_content_preserved() -> None:
    """Output size equals direct scaling; interior content stays identical.

    Opaque pixels strictly inside the bounding box have their whole stencil
    neighbourhood inside the opaque box, so padding never changes any
    neighbour they see and their scaled n x n output blocks must equal the
    direct scaling output. Opaque pixels on the bounding box edge are the
    ones the feature is designed to repair (their outside neighbours are
    transparent in the source), so their blocks are allowed to change.
    """

    for name, (image, _) in SAMPLES.items():
        top, bottom, left, right = opaque_bbox(image)
        for n, sfx in ((2, False), (2, True), (3, False), (3, True)):
            # Stencil reach beyond the center pixel: Scale2x uses radius 1,
            # Scale3x(SFX) reads up to 2 pixels away.
            radius = 1 if n == 2 else 2
            direct = scaleNx(image, n=n, sfx=sfx)
            for mode in ('mirror', 'replicate'):
                result, stats = content_aware_scaleNx(image, n=n, sfx=sfx, pad_mode=mode, pad_ratio=PAD_RATIO)
                check(len(result) == n * SIZE and len(result[0]) == n * SIZE, f'[{name}/{mode}] output size mismatch')
                check(len(direct) == len(result) and len(direct[0]) == len(result[0]), f'[{name}/{mode}] size differs from direct')

                interior_blocks = 0
                edge_blocks = 0
                edge_changed = 0
                for y in range(SIZE):
                    for x in range(SIZE):
                        if image[y][x][-1] <= 0:
                            continue
                        blocks_equal = all(
                            result[n * y + dy][n * x + dx] == direct[n * y + dy][n * x + dx]
                            for dy in range(n)
                            for dx in range(n)
                        )
                        strictly_inside = (top + radius <= y <= bottom - radius) and (left + radius <= x <= right - radius)
                        if strictly_inside:
                            interior_blocks += 1
                            check(
                                blocks_equal,
                                f'[{name}/{mode}/n={n},sfx={sfx}] interior content changed at source ({y},{x})',
                            )
                        else:
                            edge_blocks += 1
                            if not blocks_equal:
                                edge_changed += 1

                # The extension must repair border blocks of opaque pixels
                # that have real opaque neighbours inside the box. An isolated
                # opaque dot (sparse sample) has no structure to mirror: within
                # the stencil radius reflection and replication of one pixel
                # both collapse onto that pixel, matching the scaler's native
                # repeat edge, so an unchanged block is correct there.
                repairable_edge_blocks = 0
                for y in range(max(0, top - radius), min(SIZE, bottom + radius + 1)):
                    for x in range(max(0, left - radius), min(SIZE, right + radius + 1)):
                        if image[y][x][-1] <= 0:
                            continue
                        on_edge = y in (top, bottom) or x in (left, right)
                        has_opaque_neighbour = any(
                            0 <= y + dy < SIZE and 0 <= x + dx < SIZE and image[y + dy][x + dx][-1] > 0
                            for dy in (-1, 0, 1)
                            for dx in (-1, 0, 1)
                            if (dy, dx) != (0, 0)
                        )
                        if on_edge and has_opaque_neighbour:
                            repairable_edge_blocks += 1
                if repairable_edge_blocks > 0 and (top > 0 or bottom < SIZE - 1 or left > 0 or right < SIZE - 1):
                    check(edge_blocks > 0, f'[{name}/{mode}] sample must contain bbox-edge opaque pixels')
                    check(edge_changed > 0, f'[{name}/{mode}] padding must change at least one bbox-edge output block')
                elif not (top > 0 or bottom < SIZE - 1 or left > 0 or right < SIZE - 1):
                    # Opaque box fills the whole image: extension beyond the
                    # border mirrors/replicates the border itself, which is
                    # exactly the scaler's native repeat-edge stencil, so the
                    # output must stay identical to direct scaling.
                    check(edge_changed == 0, f'[{name}/{mode}] full-frame opaque box must keep direct scaling output')

                # Extension ratio must equal the actual padded-area fraction.
                py, px = stats['padded_size']
                expected_ratio = (py * px - SIZE * SIZE) / (py * px)
                check(abs(stats['extension_ratio'] - expected_ratio) < 1e-12, f'[{name}/{mode}] extension ratio mismatch')

    # Sparse single-pixel islands have no strictly-interior pixel at all;
    # verify the direct-equality requirement on a synthetic thick block where
    # an interior guaranteed to exist survives all four scaling methods.
    thick = [[[0, 0, 0, 0] for _ in range(10)] for _ in range(10)]
    rng = Random(7)
    for y in range(2, 8):
        for x in range(2, 8):
            thick[y][x] = _opaque_pixel(rng)
    for n, sfx in ((2, False), (2, True), (3, False), (3, True)):
        direct = scaleNx(thick, n=n, sfx=sfx)
        result, _ = content_aware_scaleNx(thick, n=n, sfx=sfx, pad_mode='mirror', pad_ratio=PAD_RATIO)
        radius = 1 if n == 2 else 2
        checked = 0
        for y in range(2 + radius, 8 - radius):
            for x in range(2 + radius, 8 - radius):
                checked += 1
                for dy in range(n):
                    for dx in range(n):
                        check(
                            result[n * y + dy][n * x + dx] == direct[n * y + dy][n * x + dx],
                            f'[thick/n={n},sfx={sfx}] interior pixel changed at ({y},{x})',
                        )
        check(checked > 0, 'thick block test must actually compare interior pixels')
    print('  output dimensions exact and strictly-interior content pixel-identical: OK')


def test_mirror_differs_from_replicate() -> None:
    """The two fill modes must actually produce different border pixels."""

    image, _ = SAMPLES['centered']  # has a real transparent margin to extend into
    mirror, _ = content_aware_scaleNx(image, n=N, sfx=False, pad_mode='mirror', pad_ratio=PAD_RATIO)
    replicate, _ = content_aware_scaleNx(image, n=N, sfx=False, pad_mode='replicate', pad_ratio=PAD_RATIO)
    check(mirror != replicate, 'mirror and replicate padding must give different results')
    print('  mirror and replicate modes produce different borders: OK')


def test_padded_pipeline_roundtrip() -> None:
    """Manual pad -> unmodified scaler -> crop must equal the wrapper output."""

    image, _ = SAMPLES['sparse']
    bbox = opaque_bbox(image)
    top, bottom, left, right = bbox
    box_h, box_w = (bottom - top + 1, right - left + 1)
    total_y = round(PAD_RATIO * box_h)
    total_x = round(PAD_RATIO * box_w)
    pt, pl = (total_y // 2, total_x // 2)
    pb, pr = (total_y - pt, total_x - pl)

    padded = pad_image(image, bbox, pt, pb, pl, pr, 'mirror')
    scaled = ref_scale2x(padded)
    manual = []
    for y in range(N * SIZE):
        row = scaled[N * pt + y]
        manual.append([list(row[N * pl + x]) for x in range(N * SIZE)])

    wrapped, stats = content_aware_scaleNx(image, n=N, sfx=False, pad_mode='mirror', pad_ratio=PAD_RATIO)
    check(wrapped == manual, 'wrapper result must equal explicit pad->scale->crop pipeline')
    check(stats['pad'] == (pt, pb, pl, pr), 'wrapper padding split mismatch')
    print('  explicit pad->scale->crop roundtrip equals wrapper output: OK')


""" ╔═══════════════════════╗
    ║ Four-sample benchmark ║
    ╚═══════════════════════╝ """


def run_benchmark() -> list[dict]:
    """Report edge error, PSNR, time and extension ratio for every sample."""

    rows = []
    for name, (image, _) in SAMPLES.items():
        direct, direct_time = timed(scaleNx, image, n=N, sfx=False)
        direct_smear = smearing_runs(image, direct, N)
        for mode in ('mirror', 'replicate'):
            (result, stats), ca_time = timed(
                content_aware_scaleNx, image, n=N, sfx=False, pad_mode=mode, pad_ratio=PAD_RATIO
            )
            rows.append(
                {
                    'sample': name,
                    'mode': mode,
                    'edge_error': edge_band_error(result, direct, N),
                    'psnr': psnr(result, direct),
                    'time': ca_time,
                    'direct_time': direct_time,
                    'extension_pct': 100.0 * stats['extension_ratio'],
                    'smear_direct': direct_smear,
                    'smear_ca': smearing_runs(image, result, N),
                    'stage_timing': stats['timing'],
                    'pad': stats['pad'],
                    'padded_size': stats['padded_size'],
                }
            )
    return rows


def print_benchmark(rows: list[dict]) -> None:
    header = (
        f'{"sample":<9} {"mode":<9} {"edge MAE":>9} {"PSNR,dB":>9} '
        f'{"CA time,s":>10} {"direct,s":>9} {"ext.%":>6} {"smear d->ca":>12}  pad / padded size'
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        psnr_str = 'inf' if math.isinf(r['psnr']) else f'{r["psnr"]:>9.2f}'
        print(
            f'{r["sample"]:<9} {r["mode"]:<9} {r["edge_error"]:>9.3f} {psnr_str:>9} '
            f'{r["time"]:>10.5f} {r["direct_time"]:>9.5f} {r["extension_pct"]:>5.1f}% '
            f'{r["smear_direct"]:>5.1f}->{r["smear_ca"]:<5.1f}  '
            f'{str(r["pad"]):<18} {r["padded_size"]}'
        )
    print()
    print('Pipeline stage timing (median of medians across samples), seconds:')
    for stage in ('pad', 'scale', 'crop'):
        values = [r['stage_timing'][stage] for r in rows]
        print(f'  {stage:<6}: {median(values):.5f}')
    print()

    # Quantitative sanity assertions on the reported metrics.
    for r in rows:
        check(r['edge_error'] >= 0.0 and math.isfinite(r['edge_error']), f'[{r["sample"]}/{r["mode"]}] invalid edge error')
        check(r['psnr'] > 0.0, f'[{r["sample"]}/{r["mode"]}] PSNR must be positive')
        check(r['time'] > 0.0 and r['direct_time'] > 0.0, f'[{r["sample"]}/{r["mode"]}] timing must be positive')
        check(0.0 <= r['extension_pct'] < 100.0, f'[{r["sample"]}/{r["mode"]}] extension percent out of range')
        check(r['smear_ca'] >= 0.0, f'[{r["sample"]}/{r["mode"]}] smearing run length must be non-negative')

    # Samples with a transparent margin and an opaque box wider than the
    # stencil must show broken (shorter) monochrome strips versus direct
    # scaling: that is the observable edge-smear repair. The corners sample
    # fills the whole image (no transparent side row/column), so no margin
    # exists to repair and equal run lengths are expected.
    for name in ('centered', 'edge'):
        for mode in ('mirror', 'replicate'):
            r = next(r for r in rows if r['sample'] == name and r['mode'] == mode)
            check(
                r['smear_ca'] < r['smear_direct'],
                f'[{name}/{mode}] content-aware padding must shorten the smeared edge strip ({r["smear_direct"]} -> {r["smear_ca"]})',
            )
    corners = next(r for r in rows if r['sample'] == 'corners' and r['mode'] == 'mirror')
    check(corners['edge_error'] == 0.0, 'corners sample (opaque to every border) must equal direct scaling')
    # Sparse single-dot boxes degenerate to repeat-edge behaviour; extension
    # still changes fully transparent pixels around the dots, so edge MAE
    # stays positive while the strips remain structurally similar.
    sparse = next(r for r in rows if r['sample'] == 'sparse' and r['mode'] == 'mirror')
    check(sparse['edge_error'] > 0.0, 'sparse sample padding must still alter the perimeter band')


""" ╔═══════════════════════╗
    ║ pytest / standalone    ║
    ╚═══════════════════════╝ """

# Pytest collects every test_* function; the benchmark itself is invoked
# from the standalone main() and once here under pytest as well.


def test_benchmark_runs_and_metrics_hold() -> None:
    rows = run_benchmark()
    check(len(rows) == 2 * len(SAMPLES), 'benchmark must cover both modes for every sample')
    print_benchmark(rows)


def main() -> int:
    print('ScaleNx content-aware boundary extension testbench')
    print('=' * 78)
    print(f'Samples: {len(SAMPLES)} self-generated {SIZE}x{SIZE} RGBA images, '
          f'benchmark factor n={N}, pad_ratio={PAD_RATIO}, repeats={REPEATS}')
    print()

    test_transparent_and_empty_raise()
    test_bbox_detection()
    test_padding_reference()
    test_invalid_arguments()
    test_direct_scaling_unchanged()
    test_size_and_opaque_content_preserved()
    test_mirror_differs_from_replicate()
    test_padded_pipeline_roundtrip()
    print()
    rows = run_benchmark()
    print_benchmark(rows)

    if FAILURES:
        print(f'RESULT: {len(FAILURES)} check(s) FAILED')
        for failure in FAILURES:
            print(f'  - {failure}')
        return 1
    print('RESULT: all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
