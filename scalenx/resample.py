#!/usr/bin/env python3

"""
=========
Resample
=========

------------------------------------------------------------------------------
Table-driven general purpose image resampling in pure Python.
------------------------------------------------------------------------------

:Abstract: Current module comprise arbitrary-factor image resampling driven
    by a **kernel table**. Each resampling kernel is described by three
    things only:

    - **support radius** (``support``): half-width of the kernel, in source
      pixels, *i.e.* the farthest source sample the kernel ever reaches;
    - **weight function** (``weight``): kernel profile. For a *separable*
      kernel it is a 1-D function ``w(d)`` of the distance between the
      destination point and a source sample along one axis. For a
      *non-separable* kernel it is a 2-D function ``w(dy, dx)``;
    - **normalization mode** (``normalize``): how the weights used for one
      destination pixel are made to sum to 1:

        * ``'area'``  - divide by the theoretical integral of the kernel
          over its full support (constant per kernel, exact on interior
          samples);
        * ``'local'`` - divide by the actual sum of the (clamped) weights
          covering this destination pixel (recommended at image borders,
          where repeat-edge clamping truncates the footprint).

    Adding a kernel means appending one :class:`Kernel` entry to
    :data:`KERNELS` - the main loop in :func:`resample` is never modified.

Two arithmetic engines, selected by the table entry itself
-----------------------------------------------------------

A kernel whose profile is piecewise-polynomial over the rationals returns
:class:`fractions.Fraction` weights. It runs the **exact fixed-point
engine**: coefficients are converted once to integer numerators over one
shared (per-axis) denominator, so the whole convolution is integer
multiply-accumulate and the final rounding operates on the exact rational
result. The separable two-pass and a brute-force 2-D weighted sum then
agree to the last integer, including values sitting exactly on a
half-integer boundary - there is no floating-point summation order to
disagree about.

A kernel whose profile is transcendental (e.g. Lanczos, involving
``sin``) has no exact finite rational representation; it returns ``float``
weights and runs the **floating-point engine**, reducing each output
pixel with :func:`math.fsum`. The engine a kernel uses is declared in its
table entry (``exact=True``/``False``) and is itself part of the
"normalization/arithmetic" description of the kernel.

Separable vs non-separable
--------------------------

A kernel is **separable** when its 2-D response factorizes into a product
of two 1-D profiles, ``w2d(dy, dx) = w1d(dy) * w1d(dx)``. Such a kernel is
declared with a 1-D ``weight`` and ``separable=True``; the resampler then
runs two independent 1-D passes (vertical, then horizontal), which costs
``O(Yout * Xout * support)`` instead of
``O(Yout * Xout * support ** 2)``.

A kernel is **non-separable** when no such factorization exists (rotated
kernels, genuinely radial responses, *etc.*). It is declared with a 2-D
``weight`` and ``separable=False``; the resampler evaluates the full
2-D footprint around every destination pixel in a single pass.

Both kinds share the exact same table schema (radius, weight,
normalization, separability, arithmetic); the main loop dispatches on the
entry's flags alone.

Boundary policy
---------------

Source coordinates outside the image use "repeat edge" mode, matching the
rest of ScaleNx: every source index is clamped to
``[0, X - 1]`` / ``[0, Y - 1]``. Mapping is pixel-center aligned,
``src = (dst + 0.5) / factor - 0.5``, so image corners map onto source
corners at any factor.

All math uses :mod:`math`, :mod:`fractions` and built-ins only; no
third-party modules and no network access.

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.11.19.0'
__status__ = 'Production'
__all__ = ['Kernel', 'KERNELS', 'resample', 'get_kernel']

import math
from fractions import Fraction
from typing import Callable, Literal


class Kernel:
    """One table entry describing a resampling kernel.

    ----

    :param str name: unique kernel name, used as the table key;
    :param float support: support radius in source pixels;
    :param Callable weight: weight function, ``w(d)`` (separable) or
        ``w(dy, dx)`` (non-separable). Return :class:`Fraction` for an
        exact polynomial profile, ``float`` for a transcendental one;
    :param str normalize: normalization mode, ``'area'`` or ``'local'``;
    :param bool separable: ``True`` for a 1-D-factorizable kernel,
        ``False`` for a genuine 2-D kernel;
    :param bool exact: ``True`` when every weight is an exact rational;
        selects the fixed-point integer engine. ``False`` selects the
        floating-point engine;
    :param area: optional exact theoretical integral over the full support;
        when omitted it is measured numerically once;
    :raises ValueError: unknown normalization mode.

    """

    __slots__ = ('name', 'support', 'weight', 'normalize', 'separable', 'exact', '_area')

    def __init__(
        self,
        name: str,
        support: float,
        weight: Callable,
        normalize: Literal['area', 'local'] = 'local',
        separable: bool = True,
        exact: bool = True,
        area: Fraction | float | None = None,
    ):
        if normalize not in ('area', 'local'):
            raise ValueError(f'normalize must be "area" or "local", got `{normalize}`')
        self.name = name
        self.support = float(support)
        self.weight = weight
        self.normalize = normalize
        self.separable = separable
        self.exact = exact
        if area is None:
            area = self._integral()
        self._area = area if isinstance(area, Fraction) else Fraction(area)

    def _integral(self, steps: int = 4096) -> Fraction:
        """Numerical integral of the weight over its full support."""

        r = self.support
        if self.separable:
            h = r / steps
            total = 0.5 * (self.weight(-r) + self.weight(r))
            for i in range(1, 2 * steps):
                total += self.weight(-r + i * h)
            return Fraction(float(total) * h).limit_denominator(10 ** 12)
        n = 256
        h = (2.0 * r) / n
        total = 0.0
        for iy in range(n):
            dy = -r + (iy + 0.5) * h
            for ix in range(n):
                dx = -r + (ix + 0.5) * h
                total += self.weight(dy, dx)
        return Fraction(float(total) * h * h).limit_denominator(10 ** 12)

    def __repr__(self) -> str:
        kind = 'separable' if self.separable else 'non-separable'
        engine = 'exact' if self.exact else 'float'
        return f'Kernel({self.name!r}, support={self.support:g}, normalize={self.normalize!r}, {kind}, {engine})'


""" ╔═══════════════════════════╗
    ║ 1-D weight profile funcs  ║
    ╚═══════════════════════════╝
The piecewise-polynomial profiles take and return rational Fractions, so
their values over rational distances are exact.                            """


def _w_box(d) -> Fraction:
    """Box (nearest-neighbor / pixel averaging) profile, radius 0.5."""

    return Fraction(1) if Fraction(-1, 2) < d <= Fraction(1, 2) else Fraction(0)


def _w_tent(d) -> Fraction:
    """Tent (linear / bilinear) profile, radius 1."""

    ad = d if d >= 0 else -d
    return Fraction(1) - ad if ad < 1 else Fraction(0)


def _w_quadratic_bspline(d) -> Fraction:
    """Quadratic B-spline profile, radius 1.5."""

    ad = d if d >= 0 else -d
    if ad < Fraction(1, 2):
        return Fraction(3, 4) - ad * ad
    if ad < Fraction(3, 2):
        t = Fraction(3, 2) - ad
        return Fraction(1, 2) * t * t
    return Fraction(0)


def _w_catmull_rom(d, a: Fraction = Fraction(-1, 2)) -> Fraction:
    """Keys (Catmull-Rom, ``a=-1/2``) cubic profile, radius 2."""

    ad = d if d >= 0 else -d
    ad2 = ad * ad
    ad3 = ad2 * ad
    if ad < 1:
        return (a + 2) * ad3 - (a + 3) * ad2 + 1
    if ad < 2:
        return a * ad3 - 5 * a * ad2 + 8 * a * ad - 4 * a
    return Fraction(0)


def _w_lanczos2(d) -> float:
    """Lanczos-2 profile, radius 2 (transcendental - float by nature)."""

    ad = abs(float(d))
    if ad < 2e-16:
        return 1.0
    if ad < 2.0:
        fd = float(d)
        return (math.sin(math.pi * fd) / (math.pi * fd)) * (math.sin(math.pi * fd / 2.0) / (math.pi * fd / 2.0))
    return 0.0


# ↓ Demo kernel: sharpened tent - same 1 px radius as bilinear, but the
#   center tap is up-weighted 25%. One extra table entry, no loop changes.
def _w_tent_sharp(d) -> Fraction:
    """Sharpened tent profile, radius 1 (demo kernel)."""

    ad = d if d >= 0 else -d
    if ad < 1:
        return (Fraction(1) - ad) * (Fraction(5, 4) if ad < Fraction(1, 2) else Fraction(3, 4))
    return Fraction(0)


""" ╔═══════════════════════════╗
    ║ 2-D non-separable example ║
    ╚═══════════════════════════╝ """


def _w_radial_tent(dy, dx) -> float:
    """Radial tent, radius 1 - a genuinely 2-D, non-separable kernel.

    Weight depends on Euclidean distance ``sqrt(dy**2 + dx**2)``; there is
    no ``f(dy) * g(dx)`` factorization, so it must run the 2-D path.
    """

    r = math.hypot(float(dy), float(dx))
    return 1.0 - r if r < 1.0 else 0.0


""" ╔══════════════════════════════════════════╗
    ║ Kernel table - append entries here only  ║
    ╚══════════════════════════════════════════╝ """

KERNELS: dict[str, Kernel] = {
    k.name: k
    for k in (
        # Polynomial profiles are partition of unity: exact 1-D integral 1.
        Kernel('box', 0.5, _w_box, normalize='area', separable=True, exact=True, area=Fraction(1)),
        Kernel('tent', 1.0, _w_tent, normalize='area', separable=True, exact=True, area=Fraction(1)),
        Kernel('quadratic_bspline', 1.5, _w_quadratic_bspline, normalize='area', separable=True, exact=True, area=Fraction(1)),
        Kernel('catmull_rom', 2.0, _w_catmull_rom, normalize='local', separable=True, exact=True, area=Fraction(1)),
        Kernel('lanczos2', 2.0, _w_lanczos2, normalize='local', separable=True, exact=False, area=Fraction(1)),
        # ↓ Demo entry #1: new separable exact kernel, one line.
        Kernel('tent_sharp', 1.0, _w_tent_sharp, normalize='local', separable=True, exact=True),
        # ↓ Demo entry #2: non-separable float kernel, same schema.
        # Exact 2-D area of a radial tent over the radius-1 disk: pi / 3.
        Kernel('radial_tent', 1.0, _w_radial_tent, normalize='local', separable=False, exact=False, area=math.pi / 3.0),
    )
}
"""Registry of all resampling kernels, keyed by name."""


def get_kernel(name: str) -> Kernel:
    """Return the :class:`Kernel` table entry registered under ``name``.

    :raises KeyError: no kernel with such name.

    """

    try:
        return KERNELS[name]
    except KeyError:
        raise KeyError(f'Unknown kernel `{name}`; available: {", ".join(sorted(KERNELS))}') from None


""" ╒══════════════════════════╕
    │ Helpers shared by loops │
    ╰──────────────────────────╯ """


def _half_up(value) -> int:
    """Round half up (away from zero for positives), deterministic."""

    return math.floor(value + 0.5)


def _factor_rational(factor) -> Fraction:
    """Validate a factor and return it as an exact Fraction."""

    if isinstance(factor, bool) or not isinstance(factor, (int, float, Fraction)):
        raise TypeError(f'factor must be int or float, got {type(factor).__name__}')
    if factor <= 0:
        raise ValueError(f'factor must be a positive finite number, got {factor}')
    return Fraction(factor)


def _validate_image(image3d) -> tuple[int, int, int]:
    """Validate 3-D nested-list image and return ``(Y, X, Z)`` sizes."""

    if not isinstance(image3d, list) or not image3d or not isinstance(image3d[0], list) or not image3d[0]:
        raise ValueError('source_image must be a non-empty 3D nested list of pixels')
    pixel0 = image3d[0][0]
    if not isinstance(pixel0, (list, tuple)) or not pixel0:
        raise ValueError('source_image pixels must be non-empty lists or tuples of channel values')
    Y = len(image3d)
    X = len(image3d[0])
    Z = len(pixel0)
    for row in image3d:
        if len(row) != X:
            raise ValueError('source_image rows must be of equal length')
        for pixel in row:
            if len(pixel) != Z:
                raise ValueError('source_image pixels must have equal channel count')
    return Y, X, Z


def _centers(length_in: int, length_out: int, factor: Fraction) -> list[Fraction]:
    """Exact mapped source center for every destination position."""

    half = Fraction(1, 2)
    return [((Fraction(dst) + half) / factor - half) for dst in range(length_out)]


def _rational_taps(kernel: Kernel, length_in: int, center: Fraction) -> list[tuple[int, Fraction]]:
    """Exact normalized rational taps ``[(clamped src index, weight), ...]``."""

    r = kernel.support
    fc = float(center)
    taps = []
    wsum = Fraction(0)
    for src in range(math.ceil(fc - r), math.floor(fc + r) + 1):
        w = kernel.weight(Fraction(src) - center)
        if w == 0:
            continue
        idx = min(length_in - 1, max(0, src))
        taps.append((idx, w))
        wsum += w
    denom = kernel._area if kernel.normalize == 'area' else (wsum if wsum != 0 else Fraction(1))
    return [(idx, w / denom) for idx, w in taps]


def _float_centers(length_in: int, length_out: int, factor: Fraction) -> list[float]:
    """Float mapped source centers, one per destination position."""

    f = float(factor)
    return [((dst + 0.5) / f) - 0.5 for dst in range(length_out)]


def _float_taps_fast(kernel: Kernel, length_in: int, center: float) -> list[tuple[int, float]]:
    """Float taps from a float center - no rational arithmetic on the path."""

    r = kernel.support
    raw = []
    wsum = 0.0
    for src in range(math.ceil(center - r), math.floor(center + r) + 1):
        w = float(kernel.weight(src - center))
        if w == 0.0:
            continue
        idx = min(length_in - 1, max(0, src))
        raw.append((idx, w))
        wsum += w
    denom = float(kernel._area) if kernel.normalize == 'area' else (wsum if wsum != 0.0 else 1.0)
    return [(idx, w / denom) for idx, w in raw]


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _fixed_tables(kernel: Kernel, length_in: int, length_out: int, factor: Fraction):
    """Build exact integer coefficient table and its shared denominator."""

    rational_rows = []
    denoms = set()
    for center in _centers(length_in, length_out, factor):
        taps = _rational_taps(kernel, length_in, center)
        rational_rows.append(taps)
        for _, w in taps:
            denoms.add(w.denominator)
    common = 1
    for d in denoms:
        common = _lcm(common, d)
    int_rows = [[(idx, w.numerator * (common // w.denominator)) for idx, w in taps] for taps in rational_rows]
    return int_rows, common


""" ╒══════════════════════════════════════╕
    │ Exact fixed-point separable engine  │
    ╰──────────────────────────────────────╯ """


def _resample_separable_exact(image3d, kernel, fy: Fraction, fx: Fraction, Y: int, X: int, Z: int):
    """Two integer multiply-accumulate passes, one exact final rounding."""

    out_Y = max(1, _half_up(Y * fy))
    out_X = max(1, _half_up(X * fx))
    rows_y, denom_y = _fixed_tables(kernel, Y, out_Y, fy)
    rows_x, denom_x = _fixed_tables(kernel, X, out_X, fx)
    scale = denom_y * denom_x

    intermediate = [[[0] * Z for _ in range(X)] for _ in range(out_Y)]
    for oy, taps in enumerate(rows_y):
        dst = intermediate[oy]
        for x in range(X):
            acc = [0] * Z
            for sy, wn in taps:
                pixel = image3d[sy][x]
                for z in range(Z):
                    acc[z] += wn * pixel[z]
            dst[x] = acc

    out = []
    for oy in range(out_Y):
        mid = intermediate[oy]
        row = []
        for taps in rows_x:
            acc = [0] * Z
            for sx, wn in taps:
                pixel = mid[sx]
                for z in range(Z):
                    acc[z] += wn * pixel[z]
            rounded = [_half_up(Fraction(a, scale)) for a in acc]
            row.append(list(rounded))
        out.append(row)
    return out


""" ╒══════════════════════════════════╕
    │ Float separable engine (fsum)    │
    ╰──────────────────────────────────╯ """


def _resample_separable_float(image3d, kernel, fy: Fraction, fx: Fraction, Y: int, X: int, Z: int):
    """Two float passes with an unrounded intermediate, final fsum round."""

    out_Y = max(1, _half_up(Y * fy))
    out_X = max(1, _half_up(X * fx))
    table_y = [_float_taps_fast(kernel, Y, c) for c in _float_centers(Y, out_Y, fy)]
    table_x = [_float_taps_fast(kernel, X, c) for c in _float_centers(X, out_X, fx)]

    intermediate = []
    for taps in table_y:
        row = []
        for x in range(X):
            acc = [0.0] * Z
            for sy, w in taps:
                pixel = image3d[sy][x]
                for z in range(Z):
                    acc[z] += w * pixel[z]
            row.append(acc)  # unrounded float intermediate
        intermediate.append(row)

    out = []
    for mid in intermediate:
        row = []
        for taps in table_x:
            acc = [0.0] * Z
            for sx, w in taps:
                pixel = mid[sx]
                for z in range(Z):
                    acc[z] += w * pixel[z]
            # half-up rounding; per-channel accumulators hold few taps.
            rounded = [_half_up(v) for v in acc]
            row.append(list(rounded))
        out.append(row)
    return out


""" ╒══════════════════════════╕
    │ Non-separable 2-D pass  │
    ╰──────────────────────────╯ """


def _resample_2d(image3d, kernel, fy: Fraction, fx: Fraction, Y: int, X: int, Z: int):
    """Single-pass 2-D resample for kernels without 1-D factorization."""

    out_Y = max(1, _half_up(Y * fy))
    out_X = max(1, _half_up(X * fx))
    radius = kernel.support
    half = Fraction(1, 2)
    exact = kernel.exact
    out = []
    for oy in range(out_Y):
        cy = (Fraction(oy) + half) / fy - half
        fcy = float(cy)
        row = []
        for ox in range(out_X):
            cx = (Fraction(ox) + half) / fx - half
            fcx = float(cx)
            if exact:
                sums = [Fraction(0) for _ in range(Z)]
                wsum = Fraction(0)
                for sy in range(math.ceil(fcy - radius), math.floor(fcy + radius) + 1):
                    iy = min(Y - 1, max(0, sy))
                    dy = Fraction(sy) - cy
                    src_row = image3d[iy]
                    for sx in range(math.ceil(fcx - radius), math.floor(fcx + radius) + 1):
                        w = kernel.weight(dy, Fraction(sx) - cx)
                        if w == 0:
                            continue
                        ix = min(X - 1, max(0, sx))
                        pixel = src_row[ix]
                        for z in range(Z):
                            sums[z] += w * pixel[z]
                        wsum += w
                denom = kernel._area if kernel.normalize == 'area' else (wsum if wsum != 0 else Fraction(1))
                rounded = [_half_up(s / denom) for s in sums]
            else:
                terms = [[] for _ in range(Z)]
                wsum = 0.0
                for sy in range(math.ceil(fcy - radius), math.floor(fcy + radius) + 1):
                    iy = min(Y - 1, max(0, sy))
                    dy = Fraction(sy) - cy
                    src_row = image3d[iy]
                    for sx in range(math.ceil(fcx - radius), math.floor(fcx + radius) + 1):
                        w = float(kernel.weight(dy, Fraction(sx) - cx))
                        if w == 0.0:
                            continue
                        ix = min(X - 1, max(0, sx))
                        pixel = src_row[ix]
                        for z in range(Z):
                            terms[z].append(w * pixel[z])
                        wsum += w
                denom = float(kernel._area) if kernel.normalize == 'area' else (wsum if wsum != 0.0 else 1.0)
                rounded = [_half_up(math.fsum(t) / denom) for t in terms]
            row.append(list(rounded))
        out.append(row)
    return out


""" ╒══════════════════════╕
    │ Table-driven entry  │
    ╰──────────────────────╯ """


def resample(
    image3d: list[list[list[int]]],
    factor,
    kernel: str | Kernel = 'tent',
    factor_x=None,
) -> list[list[list[int]]]:
    """Resample ``image3d`` by ``factor`` using a kernel from :data:`KERNELS`.

    ----

    :param image3d: 3-D nested list (image) of lists (rows) of lists
        (pixels) of int (channel values); tuples of int are accepted too;
    :param factor: positive int or float scaling factor applied to both
        axes (or Y only when ``factor_x`` is given);
    :param kernel: kernel table key (``str``) or a :class:`Kernel` entry;
    :param factor_x: optional separate X-axis factor; when omitted,
        ``factor`` is used on both axes;
    :raises KeyError: unknown kernel name;
    :raises TypeError/ValueError: bad factor or malformed image;
    :return: resampled 3-D nested list, output pixels are plain lists
        regardless of the input pixel container;
    :rtype: list[list[list[int]]]

    """

    entry = get_kernel(kernel) if isinstance(kernel, str) else kernel
    if not isinstance(entry, Kernel):
        raise TypeError(f'kernel must be a str name or Kernel entry, got {type(entry).__name__}')

    factor_y = _factor_rational(factor)
    factor_x = factor_y if factor_x is None else _factor_rational(factor_x)

    Y, X, Z = _validate_image(image3d)

    if not entry.separable:
        return _resample_2d(image3d, entry, factor_y, factor_x, Y, X, Z)
    if entry.exact:
        return _resample_separable_exact(image3d, entry, factor_y, factor_x, Y, X, Z)
    return _resample_separable_float(image3d, entry, factor_y, factor_x, Y, X, Z)


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
