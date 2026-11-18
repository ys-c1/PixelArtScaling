#!/usr/bin/env python3

"""
=========
EdgeScale
=========

-------------------------------------------------
Edge-directed 2x image rescaling (rule-based,
xBR / Scale2x family) with configurable blending
-------------------------------------------------

:Abstract: Current module comprises **edgescale2x** image rescaling
    function, implemented in pure Python. It detects corners, diagonal
    edges and thin lines from the 4-neighbourhood and diagonal pixels
    of every source pixel, and reconstructs the upscaled edge using
    configurable blend weights instead of plain nearest-neighbour
    block doubling.

Usage
-----

::

    scaled_image = edgescale2x(source_image)

or, with explicit configuration::

    scaled_image = edgescale2x(
        source_image,
        weights={'edge': 1.0, 'diagonal': 0.5, 'thin': 0.0},
        smooth_diagonals=True,
        enabled=True,
    )

where:

- ``source_image``: input image as list of lists (rows) of lists (pixels)
        of int (channel values);
- ``weights``: optional dict overriding any of the default blend weights:

    - ``'edge'``: blend weight for corner/edge reconstruction
      (Scale2x-style corner cuts). ``1.0`` = full replacement (crisp
      xBR-like look), ``0.5`` = half-way anti-aliasing, ``0.0`` = off;
    - ``'diagonal'``: blend weight for outer staircase tip smoothing
      (only used when ``smooth_diagonals`` is True);
    - ``'thin'``: softening weight for 1 px thin lines.
      ``0.0`` (default) preserves thin lines exactly;

- ``smooth_diagonals``: enable/disable diagonal staircase tip smoothing;
- ``enabled``: when False, output is plain nearest-neighbour 2x,
        pixel-identical to block doubling;

- ``scaled_image``: output image as list of lists (rows) of lists (pixels)
        of int (channel values), 2x size of input in both dimensions.

Notes
-----

Processing is palette-based for speed: the source image may contain
at most 256 distinct colours (which is normally the case for pixel art).
Blended colours produced by weight tables are added to the palette
on the fly and do not count towards this limit.

----
The Developer site: `The Toad's Slimy Mudhole`_

.. _The Toad's Slimy Mudhole: https://dnyarri.github.io

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__credits__ = ['Andrea Mazzoleni', 'Ilya Razmanov']
__license__ = 'unlicense'
__version__ = '2026.11.19.1'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'

DEFAULT_WEIGHTS: dict[str, float] = {
    'edge': 0.75,     # corner/edge reconstruction (Scale2x-style corner cuts)
    'diagonal': 0.5,  # outer staircase tip smoothing
    'thin': 0.0,      # 1 px thin line softening (0 = preserve exactly)
}

_MAX_COLOURS = 256


def _make_mixer(weight: float, palette: list, pal_index: dict):
    """Create colour mixing function for a fixed blend weight.

    Returns function ``mix(i, j) -> int`` mapping two palette indices
    to the palette index of their blend. Blended colours are appended
    to ``palette``/``pal_index`` on first use and cached.

    """
    if weight <= 0.0:
        return lambda i, j: i
    if weight >= 1.0:
        return lambda i, j: j

    cache: dict[tuple[int, int], int] = {}

    def mix(i: int, j: int) -> int:
        if i == j:
            return i
        key = (i, j) if i < j else (j, i)
        index = cache.get(key)
        if index is None:
            ci, cj = palette[i], palette[j]
            new_colour = tuple(round(a + (b - a) * weight) for a, b in zip(ci, cj))
            index = pal_index.get(new_colour)
            if index is None:
                index = len(palette)
                pal_index[new_colour] = index
                palette.append(list(new_colour))
            cache[key] = index
        return index

    return mix


def edgescale2x(
    image3d: list[list[list[int]]],
    weights: dict[str, float] | None = None,
    smooth_diagonals: bool = True,
    enabled: bool = True,
) -> list[list[list[int]]]:
    """Edge-directed 2x image rescale with configurable rule weights.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values); coordinate system match Photoshop, *i.e.*
        origin is top left corner, channels order is LA or RGBA from 0 to top;
        at most 256 distinct colours.
    :type image3d: list[list[list[int]]]
    :param weights: optional dict overriding any of ``DEFAULT_WEIGHTS``
        (``'edge'``, ``'diagonal'``, ``'thin'``); each weight is clamped
        to [0.0, 1.0];
    :type weights: dict[str, float] | None
    :param bool smooth_diagonals: enable diagonal staircase tip smoothing;
    :param bool enabled: when False, return plain nearest-neighbour 2x
        (pixel-identical to block doubling);
    :raises ValueError: unknown weight name or more than 256 source colours;
    :return: 3D nested list of the same structure as input,
        rescaled in X and Y directions twice.
    :rtype: list[list[list[int]]]

    """

    # ↓ Merging user weights over defaults
    w_table = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(w_table)
        if unknown:
            raise ValueError(f'Unknown weight name(s): {sorted(unknown)}. Known: {sorted(w_table)}')
        w_table.update(weights)
    w_edge = min(1.0, max(0.0, float(w_table['edge'])))
    w_diag = min(1.0, max(0.0, float(w_table['diagonal'])))
    w_thin = min(1.0, max(0.0, float(w_table['thin'])))

    # ↓ Determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

    """ ╔════════════════════════════════════════════╗
        ║ Pass 1: palette build and index conversion ║
        ╚════════════════════════════════════════════╝ """

    pal_index: dict[tuple, int] = {}
    idx_rows: list[bytes] = []
    get_index = pal_index.get
    for row in image3d:
        try:
            # ↓ Fast path: all colours of the row are already registered
            idx_rows.append(bytes(map(pal_index.__getitem__, map(tuple, row))))
        except KeyError:
            # ↓ Slow path: register new colours on the fly
            idx = bytearray()
            for pixel in row:
                key = tuple(pixel)
                index = get_index(key)
                if index is None:
                    index = len(pal_index)
                    if index >= _MAX_COLOURS:
                        raise ValueError(f'EdgeScale is palette-based and supports at most {_MAX_COLOURS} distinct source colours.')
                    pal_index[key] = index
                idx.append(index)
            idx_rows.append(bytes(idx))

    # ↓ Palette as list of pixel lists (output format, shared references
    #   in the same manner as scale2x/scale3x reuse source pixel lists)
    palette: list[list[int]] = [[]] * len(pal_index)
    for key, index in pal_index.items():
        palette[index] = list(key)

    """ ╔════════════════════════════════════════════╗
        ║ Nearest-neighbour base rows (C-speed)      ║
        ╚════════════════════════════════════════════╝ """

    base_cache: dict[bytes, list] = {}

    def get_base_row(idx_row: bytes) -> list:
        """Nearest-neighbour doubled row for given index row (cached)."""
        base = base_cache.get(idx_row)
        if base is None:
            half = [palette[b] for b in idx_row]
            base = [None] * (2 * X)
            base[0::2] = half
            base[1::2] = half
            base_cache[idx_row] = base
        return base

    if not enabled:
        # ↓ Pure nearest-neighbour: doubled rows, each appended twice
        scaled_image: list[list[list[int]]] = []
        for idx_row in idx_rows:
            base = get_base_row(idx_row)
            scaled_image.append(base)
            scaled_image.append(base)
        return scaled_image

    """ ╔════════════════════════════════════════════╗
        ║ Pass 2: edge detection via big-int bitops  ║
        ║ A pixel is "edge" when its 3x3 window is   ║
        ║ not uniform; only edge pixels may deviate  ║
        ║ from nearest-neighbour output.             ║
        ╚════════════════════════════════════════════╝ """

    row_ints = [int.from_bytes(r, 'little') for r in idx_rows]
    byte_bit_mask = int.from_bytes(b'\x01' * X, 'little')  # bit 0 of every byte set

    edge_masks: list[int] = []
    for y in range(Y):
        im = row_ints[y]
        iu = row_ints[y - 1] if y else im
        idn = row_ints[y + 1] if y < Y - 1 else im
        diff = 0
        for other in (iu, im, idn):
            xor = im ^ other
            diff |= xor | (xor << 8) | (xor >> 8)
        # ↓ Collapse per-byte difference flags to bit 0 of each byte
        diff |= diff >> 4
        diff |= diff >> 2
        diff |= diff >> 1
        edge_masks.append(diff & byte_bit_mask)

    """ ╔════════════════════════════════════════════╗
        ║ Pass 3: rule kernel over edge pixels       ║
        ╚════════════════════════════════════════════╝ """

    mix_edge = _make_mixer(w_edge, palette, pal_index)
    mix_diag = _make_mixer(w_diag, palette, pal_index)
    mix_thin = _make_mixer(w_thin, palette, pal_index)

    def kernel(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> tuple[int, int, int, int]:
        """Edge-directed 2x kernel.

        Source around default pixel E::

            ┌───┬───┬───┐
            │ a │ b │ c │
            ├───┼───┼───┤
            │ d │ e │ f │
            ├───┼───┼───┤
            │ g │ h │ i │
            └───┴───┴───┘

        Result::

            ┌────┬────┐
            │ r1 │ r2 │
            ├────┼────┤
            │ r3 │ r4 │
            └────┴────┘

        """
        r1 = r2 = r3 = r4 = e

        # ↓ Thin 1 px lines: preserved (optionally softened by 'thin' weight)
        if (b == e and h == e and d == f and d != e) or (d == e and f == e and b == h and b != e):
            if w_thin > 0.0:
                soft = mix_thin(e, d if d != e else b)
                r1 = r2 = r3 = r4 = soft
            return r1, r2, r3, r4

        # ↓ Corners/edges (Scale2x-style): cut the corner toward
        #   the matching orthogonal neighbours
        if b != h and d != f:
            if d == b:
                r1 = mix_edge(e, d)
            if b == f:
                r2 = mix_edge(e, f)
            if d == h:
                r3 = mix_edge(e, d)
            if h == f:
                r4 = mix_edge(e, f)

        # ↓ Diagonal staircase outer tips: exactly one adjacent
        #   (orthogonal + its diagonal) neighbour pair differs from E
        if smooth_diagonals:
            if a != e and a == b and c == e and d == e and f == e and g == e and h == e and i == e:
                r1 = mix_diag(e, b)
            elif a != e and a == d and b == e and c == e and f == e and g == e and h == e and i == e:
                r1 = mix_diag(e, d)
            elif c != e and c == b and a == e and d == e and f == e and g == e and h == e and i == e:
                r2 = mix_diag(e, b)
            elif c != e and c == f and a == e and b == e and d == e and g == e and h == e and i == e:
                r2 = mix_diag(e, f)
            elif i != e and i == f and a == e and b == e and c == e and d == e and g == e and h == e:
                r4 = mix_diag(e, f)
            elif i != e and i == h and a == e and b == e and c == e and d == e and f == e and g == e:
                r4 = mix_diag(e, h)
            elif g != e and g == h and a == e and b == e and c == e and d == e and f == e and i == e:
                r3 = mix_diag(e, h)
            elif g != e and g == d and a == e and b == e and c == e and f == e and h == e and i == e:
                r3 = mix_diag(e, d)

            # ↓ Diagonal slope fills: a diagonal + orthogonal neighbour
            #   pair agrees against E, the opposite orthogonal equals E,
            #   and a flanking neighbour also differs from E (this excludes
            #   straight edges and isolated tips, handled above)
            if a != e:
                if a == b and d == e and (f != e or h != e):
                    r1 = mix_diag(e, b)
                elif a == d and b == e and (f != e or h != e):
                    r1 = mix_diag(e, d)
            if c != e:
                if c == b and f == e and (d != e or h != e):
                    r2 = mix_diag(e, b)
                elif c == f and b == e and (d != e or h != e):
                    r2 = mix_diag(e, f)
            if g != e:
                if g == d and h == e and (b != e or f != e):
                    r3 = mix_diag(e, d)
                elif g == h and d == e and (b != e or f != e):
                    r3 = mix_diag(e, h)
            if i != e:
                if i == f and h == e and (b != e or d != e):
                    r4 = mix_diag(e, f)
                elif i == h and f == e and (b != e or d != e):
                    r4 = mix_diag(e, h)

        return r1, r2, r3, r4

    kernel_cache: dict[bytes, tuple[int, int, int, int]] = {}
    Xm1 = X - 1

    scaled_image = []
    for y in range(Y):
        mask = edge_masks[y]
        base = get_base_row(idx_rows[y])
        if mask == 0:
            # ↓ No edge pixels: both output rows are the shared NN row
            scaled_image.append(base)
            scaled_image.append(base)
            continue

        row_top = base.copy()
        row_bot = base.copy()

        up_r = idx_rows[y - 1] if y else idx_rows[0]
        mid_r = idx_rows[y]
        dn_r = idx_rows[y + 1] if y < Y - 1 else idx_rows[Y - 1]

        while mask:
            lsb = mask & -mask
            x = (lsb.bit_length() - 1) >> 3
            mask ^= lsb

            if 0 < x < Xm1:
                key = up_r[x - 1 : x + 2] + mid_r[x - 1 : x + 2] + dn_r[x - 1 : x + 2]
            else:
                # ↓ Border pixel, "repeat edge" mode
                xl = x - 1 if x else 0
                xr = x + 1 if x < Xm1 else x
                key = bytes((up_r[xl], up_r[x], up_r[xr], mid_r[xl], mid_r[x], mid_r[xr], dn_r[xl], dn_r[x], dn_r[xr]))

            result = kernel_cache.get(key)
            if result is None:
                result = kernel(*key)
                kernel_cache[key] = result
            r1, r2, r3, r4 = result

            e = mid_r[x]
            x2 = x + x
            if r1 != e:
                row_top[x2] = palette[r1]
            if r2 != e:
                row_top[x2 + 1] = palette[r2]
            if r3 != e:
                row_bot[x2] = palette[r3]
            if r4 != e:
                row_bot[x2 + 1] = palette[r4]

        scaled_image.append(row_top)
        scaled_image.append(row_bot)

    return scaled_image


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import edgescale

        help(edgescale)
