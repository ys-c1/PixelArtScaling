#!/usr/bin/env python3

"""
==============================
ScaleNx export cache testbench
==============================

Self-contained test and benchmark for the rescaling result cache
added to ``scalenx.scaleNx``.

Simulates repeated batch export (rescale + write PNM) of a small
self-generated image set in four scenarios:

1. ``duplicate`` - exactly the same image and parameters every time;
2. ``params``    - same image, cycling through all 4 parameter combos;
3. ``inputs``    - same parameters, different image every time;
4. ``mixed``     - pseudo-random mix of images and parameters.

For every scenario reports hit rate, export time (cached vs. forced-miss
baseline), cache memory footprint, and verifies that cached results and
exported files are pixel-by-pixel/byte-by-byte identical to recomputed
reference.

Also verifies cache configurability (capacity, eviction policy),
that cache capacity 0 raises ValueError, and that cached data is
immune to caller-side mutations.

No third-party dependencies; run as::

    python3 test_export_cache.py

Exit code 0 means all checks passed, 1 means at least one failure.

"""

__author__ = 'Ilya Razmanov'
__license__ = 'unlicense'

import sys
import tempfile
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pypnm import list2pnm
from scalenx import cache_clear, cache_info, scaleNx, set_cache_maxsize, set_cache_policy
from scalenx.scalenx import scale2x as ref_scale2x
from scalenx.scalenx import scale3x as ref_scale3x
from scalenx.scalenxsfx import scale2x as ref_scale2xsfx
from scalenx.scalenxsfx import scale3x as ref_scale3xsfx

MAXCOLORS = 255
SIZE = 64  # sample images are SIZE x SIZE x 3, small but non-trivial

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record a check result."""
    if not condition:
        FAILURES.append(message)
        print(f'  FAIL: {message}')


""" ╔═══════════════════════╗
    ║ Sample image factory  ║
    ╚═══════════════════════╝ """


def make_image(variant: int, size: int = SIZE) -> list[list[list[int]]]:
    """Generate deterministic small RGB test image, 8 variants."""

    rng = Random(1000 + variant)
    image = []
    for y in range(size):
        row = []
        for x in range(size):
            if variant == 0:  # gradient
                px = [(x * 255) // (size - 1), (y * 255) // (size - 1), ((x + y) * 255) // (2 * size - 2)]
            elif variant == 1:  # checkerboard
                v = 255 if (x // 4 + y // 4) % 2 else 0
                px = [v, 255 - v, v]
            elif variant == 2:  # diagonal stripes
                v = 255 if (x + y) % 8 < 4 else 32
                px = [v, v, 255 - v]
            elif variant == 3:  # circles-ish
                v = 255 if ((x - size // 2) ** 2 + (y - size // 2) ** 2) % 97 < 48 else 16
                px = [v, 128, 255 - v]
            elif variant == 4:  # random noise
                px = [rng.randint(0, 255) for _ in range(3)]
            elif variant == 5:  # flat with single-pixel details
                px = [200, 200, 200] if (x * y) % 113 else [255, 0, 0]
            elif variant == 6:  # horizontal bands
                px = [(y * 37) % 256, (y * 91) % 256, 64]
            else:  # vertical edge patterns
                px = [255, 255, 255] if (x % 7) in (0, 1) else [0, 0, (x * 53) % 256]
            row.append(px)
        image.append(row)
    return image


IMAGES = [make_image(v) for v in range(8)]

COMBOS = [(2, False), (2, True), (3, False), (3, True)]


def reference(image: list[list[list[int]]], n: int, sfx: bool) -> list[list[list[int]]]:
    """Compute reference result bypassing the cache entirely."""

    if sfx:
        return ref_scale2xsfx(image) if n == 2 else ref_scale3xsfx(image)
    return ref_scale2x(image) if n == 2 else ref_scale3x(image)


""" ╔═══════════════════════╗
    ║ Export pipeline       ║
    ╚═══════════════════════╝ """


def export(image: list[list[list[int]]], n: int, sfx: bool, out_path: Path) -> tuple[list[list[list[int]]], float]:
    """One export: rescale via public scaleNx, then write PNM file.

    Returns result image and time spent in `scaleNx` itself.

    """

    start = time.perf_counter()
    result = scaleNx(image, n=n, sfx=sfx)
    scale_time = time.perf_counter() - start
    list2pnm(str(out_path), result, MAXCOLORS)
    return result, scale_time


def run_scenario(name: str, plan: list[tuple[int, int, bool]], workdir: Path) -> dict:
    """Run one scenario twice: forced-miss baseline and cached.

    ``plan`` is a list of (image index, n, sfx) export jobs.

    """

    # ↓ Forced-miss baseline: cache cleared before every export,
    #   so every export is recomputed from scratch.
    base_dir = workdir / f'{name}_baseline'
    base_dir.mkdir()
    cache_clear()
    base_results = []
    base_scale = 0.0
    start = time.perf_counter()
    for i, (img_idx, n, sfx) in enumerate(plan):
        cache_clear()
        result, scale_time = export(IMAGES[img_idx], n, sfx, base_dir / f'{i:03d}.ppm')
        base_results.append(result)
        base_scale += scale_time
    baseline_time = time.perf_counter() - start

    # ↓ Cached run.
    cache_dir = workdir / f'{name}_cached'
    cache_dir.mkdir()
    cache_clear()
    cached_results = []
    cached_scale = 0.0
    start = time.perf_counter()
    for i, (img_idx, n, sfx) in enumerate(plan):
        result, scale_time = export(IMAGES[img_idx], n, sfx, cache_dir / f'{i:03d}.ppm')
        cached_results.append(result)
        cached_scale += scale_time
    cached_time = time.perf_counter() - start
    info = cache_info()

    # ↓ Consistency: cached results vs. forced-miss baseline vs. direct reference,
    #   in memory (pixel-by-pixel) and on disk (byte-by-byte).
    consistent = True
    for i, (img_idx, n, sfx) in enumerate(plan):
        ref = reference(IMAGES[img_idx], n, sfx)
        if cached_results[i] != ref:
            consistent = False
            check(False, f'[{name}] export #{i}: cached result differs from recomputed reference')
        if cached_results[i] != base_results[i]:
            consistent = False
            check(False, f'[{name}] export #{i}: cached result differs from forced-miss baseline')
        if (cache_dir / f'{i:03d}.ppm').read_bytes() != (base_dir / f'{i:03d}.ppm').read_bytes():
            consistent = False
            check(False, f'[{name}] export #{i}: exported file differs from baseline file')

    total = info['hits'] + info['misses']
    return {
        'name': name,
        'exports': total,
        'hits': info['hits'],
        'misses': info['misses'],
        'hit_rate': info['hits'] / total if total else 0.0,
        'baseline_time': baseline_time,
        'cached_time': cached_time,
        'baseline_scale': base_scale,
        'cached_scale': cached_scale,
        'cache_memory': info['memory'],
        'cache_size': info['size'],
        'consistent': consistent,
    }


""" ╔═══════════════════════╗
    ║ Configuration checks  ║
    ╚═══════════════════════╝ """


def raises_value_error(func, *args) -> bool:
    try:
        func(*args)
    except ValueError:
        return True
    return False


def check_configuration() -> None:
    """Verify capacity/policy configurability and error handling."""

    print('Configuration checks:')

    # ↓ Cache of size 0 (and other invalid capacities) must raise
    check(raises_value_error(set_cache_maxsize, 0), 'set_cache_maxsize(0) must raise ValueError')
    check(raises_value_error(set_cache_maxsize, -3), 'set_cache_maxsize(-3) must raise ValueError')
    check(raises_value_error(set_cache_maxsize, 2.5), 'set_cache_maxsize(2.5) must raise ValueError')
    check(raises_value_error(set_cache_maxsize, True), 'set_cache_maxsize(True) must raise ValueError')
    print('  cache capacity 0 and other invalid capacities raise ValueError: OK')

    # ↓ Unknown eviction policy must raise
    check(raises_value_error(set_cache_policy, 'random'), "set_cache_policy('random') must raise ValueError")
    set_cache_policy('lru')
    set_cache_policy('fifo')
    set_cache_policy('lru')
    print('  eviction policy configurable (lru/fifo), unknown policy raises ValueError: OK')

    # ↓ Capacity is enforced
    set_cache_maxsize(2)
    cache_clear()
    scaleNx(IMAGES[0], 2, False)
    scaleNx(IMAGES[1], 2, False)
    scaleNx(IMAGES[2], 2, False)
    check(cache_info()['size'] == 2, f'cache size must be capped at 2, got {cache_info()["size"]}')
    print('  capacity enforced (maxsize=2 keeps 2 entries): OK')

    # ↓ LRU: hit refreshes entry, so B (not A) is evicted.
    #   Sequence A B A(hit) C with maxsize=2 must give [A, C].
    #   Each probe is checked on a fresh sequence since a miss-probe
    #   re-inserts the probed entry and evicts another one.
    set_cache_policy('lru')

    def lru_sequence():
        cache_clear()
        scaleNx(IMAGES[0], 2, False)  # miss, cache: A
        scaleNx(IMAGES[1], 2, False)  # miss, cache: A B
        scaleNx(IMAGES[0], 2, False)  # hit,  cache: B A
        scaleNx(IMAGES[2], 2, False)  # miss, evicts B, cache: A C
        return cache_info()

    info = lru_sequence()
    scaleNx(IMAGES[0], 2, False)  # must be hit (A refreshed, retained)
    check(cache_info()['hits'] - info['hits'] == 1, 'LRU: refreshed entry A must survive, got a miss')
    info = lru_sequence()
    scaleNx(IMAGES[1], 2, False)  # must be miss (B evicted)
    check(cache_info()['misses'] - info['misses'] == 1, 'LRU: stale entry B must be evicted, got a hit')
    print('  LRU eviction: recently used entry survives, stale evicted: OK')

    # ↓ FIFO: hit does NOT refresh entry, so A is evicted.
    #   Sequence A B A(hit) C with maxsize=2 must give [B, C].
    set_cache_policy('fifo')

    def fifo_sequence():
        cache_clear()
        scaleNx(IMAGES[0], 2, False)  # miss, cache: A
        scaleNx(IMAGES[1], 2, False)  # miss, cache: A B
        scaleNx(IMAGES[0], 2, False)  # hit,  cache: A B (no refresh)
        scaleNx(IMAGES[2], 2, False)  # miss, evicts A, cache: B C
        return cache_info()

    info = fifo_sequence()
    scaleNx(IMAGES[1], 2, False)  # must be hit (B retained)
    check(cache_info()['hits'] - info['hits'] == 1, 'FIFO: entry B must survive, got a miss')
    info = fifo_sequence()
    scaleNx(IMAGES[0], 2, False)  # must be miss (A evicted despite the hit)
    check(cache_info()['misses'] - info['misses'] == 1, 'FIFO: entry A must be evicted regardless of hit, got a hit')
    print('  FIFO eviction: oldest entry evicted regardless of hits: OK')

    # ↓ Shrinking capacity evicts excess entries immediately
    set_cache_policy('lru')
    set_cache_maxsize(8)
    cache_clear()
    for i in range(8):
        scaleNx(IMAGES[i], 2, False)
    set_cache_maxsize(3)
    check(cache_info()['size'] == 3, f'shrinking maxsize must evict immediately, got size {cache_info()["size"]}')
    print('  shrinking capacity evicts excess entries immediately: OK')

    # ↓ Invalid method n must still raise, cached or not
    check(raises_value_error(scaleNx, IMAGES[0], 4, False), 'scaleNx(n=4) must raise ValueError')
    check(raises_value_error(scaleNx, IMAGES[0], 4, False), 'scaleNx(n=4) repeated must raise ValueError')
    print('  invalid method n still raises ValueError: OK')

    # ↓ Source mutation isolation: mutating the source image after
    #   caching must not corrupt the cached result (cached copy is
    #   independent even though raw results alias source pixels).
    #   Reference is recomputed from a fresh copy AFTER the mutation,
    #   since an uncached result aliases source pixels and would get
    #   contaminated by the mutation.
    set_cache_maxsize(16)
    cache_clear()
    probe = make_image(4, 12)
    scaleNx(probe, 3, True)
    probe[0][0][0] = (probe[0][0][0] + 1) % 256  # caller mutates source afterwards
    info = cache_info()
    second = scaleNx(make_image(4, 12), 3, True)  # same content as original probe
    check(cache_info()['hits'] - info['hits'] == 1, 'repeat export of unchanged input must be a cache hit')
    ref = reference(make_image(4, 12), 3, True)  # fresh uncontaminated recompute
    check(second == ref, 'cached result must be immune to source mutation after caching')
    print('  cached results immune to source mutation after caching: OK')

    set_cache_policy('lru')
    set_cache_maxsize(16)
    cache_clear()


""" ╔═══════════════════════╗
    ║ Main                  ║
    ╚═══════════════════════╝ """


def main() -> int:
    print('ScaleNx export cache testbench')
    print('=' * 78)

    check_configuration()
    print()

    rng = Random(42)
    scenarios = [
        ('duplicate', [(0, 2, False)] * 8),
        ('params', [(0, n, sfx) for _ in range(3) for (n, sfx) in COMBOS]),
        ('inputs', [(i, 2, False) for i in range(8)]),
        ('mixed', [(rng.randrange(4), *COMBOS[rng.randrange(4)]) for _ in range(24)]),
    ]

    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, plan in scenarios:
            reports.append(run_scenario(name, plan, Path(tmp)))

    header = f'{"scenario":<10} {"exports":>7} {"hits":>5} {"miss":>5} {"hit%":>7} | {"export base,s":>13} {"export cache,s":>14} {"speedup":>7} | {"scale base,s":>12} {"scale cache,s":>13} {"speedup":>7} | {"cache mem":>10} {"consistent":>10}'
    print(header)
    print('-' * len(header))
    for r in reports:
        speedup = r['baseline_time'] / r['cached_time'] if r['cached_time'] > 0 else float('inf')
        scale_speedup = r['baseline_scale'] / r['cached_scale'] if r['cached_scale'] > 0 else float('inf')
        print(
            f'{r["name"]:<10} {r["exports"]:>7} {r["hits"]:>5} {r["misses"]:>5} '
            f'{r["hit_rate"] * 100:>6.1f}% | {r["baseline_time"]:>13.4f} {r["cached_time"]:>14.4f} '
            f'{speedup:>6.1f}x | {r["baseline_scale"]:>12.4f} {r["cached_scale"]:>13.4f} '
            f'{scale_speedup:>6.1f}x | {r["cache_memory"]:>7d} B {str(r["consistent"]):>10}'
        )
    print()

    # ↓ Expected hit patterns per scenario
    by_name = {r['name']: r for r in reports}
    check(by_name['duplicate']['hits'] == 7 and by_name['duplicate']['misses'] == 1, 'duplicate scenario must give 7 hits / 1 miss')
    check(by_name['params']['hits'] == 8 and by_name['params']['misses'] == 4, 'params scenario must give 8 hits / 4 misses')
    check(by_name['inputs']['hits'] == 0 and by_name['inputs']['misses'] == 8, 'inputs scenario must give 0 hits / 8 misses')
    check(by_name['mixed']['hits'] > 0, 'mixed scenario must give some hits')
    for r in reports:
        check(r['consistent'], f'[{r["name"]}] cached export must be identical to recompute')

    if FAILURES:
        print(f'RESULT: {len(FAILURES)} check(s) FAILED')
        for failure in FAILURES:
            print(f'  - {failure}')
        return 1
    print('RESULT: all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
