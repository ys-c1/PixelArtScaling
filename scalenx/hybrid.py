#!/usr/bin/env python3

"""
==========
HybridScale
==========

------------------------------------------------------------------------------
Mixed nearest-neighbor / edge-directed interpolation scaling, including
non-integer (fractional) factors, in pure Python.
------------------------------------------------------------------------------

:Abstract: Current module comprise arbitrary-factor image rescaling,
    combining two methods:

    - nearest-neighbor upscaling, preserving hard pixel edges and introducing
      no new colors ("pixel feel");
    - edge-directed interpolation (EDI), filling non-integer transition
      pixels with intermediate values while avoiding blur across detected
      edges.

    The two components are mixed in a configurable proportion.

Usage
-----

::

    from scalenx.hybrid import hybrid_scale

    result_image = hybrid_scale(source_image, factor, interp_weight=None)

where:

- ``source_image``: input image as list of lists (rows) of lists (pixels)
    of int (channel values), same structure as the rest of ScaleNx;
- ``factor``: positive scaling factor, integer or float (e.g. ``1.5``,
    ``2.5``, ``3.25``); output size is ``round(Y * factor)`` per axis;
- ``interp_weight``: mixing proportion in range ``0..1``:

    - ``0``  - pure nearest-neighbor, no interpolation at all;
    - ``1``  - pure edge-directed interpolation;
    - values between - convex blend of the two, per output pixel;
    - ``None`` (default) - use the fractional part of ``factor``
      (``factor - floor(factor)``). Integer factors therefore default
      to pure nearest-neighbor.

All math is based on :mod:`math` and built-ins only; no third-party
modules are used.

Boundary policy
---------------

Source coordinates outside the image are resolved in "repeat edge" mode,
matching the rest of ScaleNx: every coordinate is clamped to
``[0, X - 1]`` / ``[0, Y - 1]``. Mapping is pixel-center aligned,
``src = (dst + 0.5) / factor - 0.5``, so image corners always map to
source corners at any factor.

Integer factors
---------------

For an integer factor (``int`` or ``float`` with zero fractional part)
each output block of ``factor x factor`` pixels is a verbatim copy of one
source pixel, independently of ``interp_weight``. The result is therefore
pixel-identical to plain block nearest-neighbor upscaling.

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.11.18.0'
__status__ = 'Production'
__all__ = ['nearest_neighbor_scale', 'edge_directed_scale', 'hybrid_scale']

import math


def _validate_factor(factor) -> float:
    """Validate scaling factor and return it as float."""

    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise TypeError(f'factor must be int or float, got {type(factor).__name__}')
    factor = float(factor)
    if not math.isfinite(factor):
        raise ValueError(f'factor must be finite, got {factor}')
    if factor <= 0:
        raise ValueError(f'factor must be positive, got {factor}')
    return factor


def _validate_image(image3d) -> tuple[int, int, int]:
    """Validate 3D nested list image and return ``(Y, X, Z)`` sizes."""

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


def _half_up(value: float) -> int:
    """Round half up (away from zero for positives), deterministic.

    Plain ``round`` uses banker's rounding, which would map e.g.
    ``round(0.5)`` to 0; image scaling expects half-up instead.
    """

    return math.floor(value + 0.5)


def nearest_neighbor_scale(image3d: list[list[list[int]]], factor) -> list[list[list[int]]]:
    """Pure nearest-neighbor (pixel block) rescale by arbitrary factor.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param factor: positive int or float scaling factor;
    :raises ValueError: factor zero, negative, non-finite, or empty image;
    :raises TypeError: factor of non-numeric type;
    :return: 3D nested list of the same structure as input, resampled with
        nearest-neighbor only, *i.e.* no new channel values are introduced.
    :rtype: list[list[list[int]]]

    """

    factor = _validate_factor(factor)
    Y, X, _ = _validate_image(image3d)

    out_X = max(1, _half_up(X * factor))
    out_Y = max(1, _half_up(Y * factor))

    # ↓ precomputed source index per destination index, repeat-edge clamped.
    #   Pixel-center mapping src = (dst + 0.5) / factor - 0.5, then the
    #   nearest source pixel is floor(src + 0.5) = floor((dst + 0.5) / factor).
    map_x = [min(X - 1, max(0, math.floor((x + 0.5) / factor))) for x in range(out_X)]
    map_y = [min(Y - 1, max(0, math.floor((y + 0.5) / factor))) for y in range(out_Y)]

    scaled_image: list[list[list[int]]] = []
    for sy in map_y:
        src_row = image3d[sy]
        out_row = []
        for sx in map_x:
            out_row.append(list(src_row[sx]))  # copy, never alias source pixels
        scaled_image.append(out_row)

    return scaled_image


def edge_directed_scale(image3d: list[list[list[int]]], factor) -> list[list[list[int]]]:
    """Edge-directed interpolation (EDI) rescale by arbitrary factor.
    ----

    For each destination pixel a 2x2 source neighborhood around the mapped
    point is used. Horizontal and vertical local variation (sum of absolute
    channel differences) are compared:

    - small vertical variation means a horizontal edge -> interpolate
      vertically between top and bottom rows;
    - small horizontal variation means a vertical edge -> interpolate
      horizontally between left and right columns;
    - equal variation (including flat areas and exact corners) -> bilinear.

    Source bounds use repeat-edge clamping, so no coordinate ever leaves
    the image.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param factor: positive int or float scaling factor;
    :raises ValueError: factor zero, negative, non-finite, or empty image;
    :raises TypeError: factor of non-numeric type;
    :return: 3D nested list of the same structure as input.
    :rtype: list[list[list[int]]]

    """

    factor = _validate_factor(factor)
    Y, X, Z = _validate_image(image3d)

    out_X = max(1, _half_up(X * factor))
    out_Y = max(1, _half_up(Y * factor))

    scaled_image: list[list[list[int]]] = []

    for oy in range(out_Y):
        fy = (oy + 0.5) / factor - 0.5
        y0 = min(Y - 1, max(0, math.floor(fy)))
        y1 = min(Y - 1, y0 + 1)
        ty = min(1.0, max(0.0, fy - y0))
        top_row = image3d[y0]
        bottom_row = image3d[y1]

        out_row = []
        for ox in range(out_X):
            fx = (ox + 0.5) / factor - 0.5
            x0 = min(X - 1, max(0, math.floor(fx)))
            x1 = min(X - 1, x0 + 1)
            tx = min(1.0, max(0.0, fx - x0))

            p00 = top_row[x0]      # top-left
            p01 = top_row[x1]      # top-right
            p10 = bottom_row[x0]   # bottom-left
            p11 = bottom_row[x1]   # bottom-right

            # ↓ local variation along each edge, L1 over channels
            var_horizontal = sum(abs(p00[c] - p01[c]) + abs(p10[c] - p11[c]) for c in range(Z))
            var_vertical = sum(abs(p00[c] - p10[c]) + abs(p01[c] - p11[c]) for c in range(Z))

            pixel = []
            if var_vertical < var_horizontal:
                # horizontal edge: average the two rows, then blend vertically
                for c in range(Z):
                    top = 0.5 * (p00[c] + p01[c])
                    bottom = 0.5 * (p10[c] + p11[c])
                    pixel.append(_half_up(top + (bottom - top) * ty))
            elif var_horizontal < var_vertical:
                # vertical edge: average the two columns, then blend horizontally
                for c in range(Z):
                    left = 0.5 * (p00[c] + p10[c])
                    right = 0.5 * (p01[c] + p11[c])
                    pixel.append(_half_up(left + (right - left) * tx))
            else:
                # tie (flat area / diagonal ambiguity / corner): bilinear
                for c in range(Z):
                    top = p00[c] + (p01[c] - p00[c]) * tx
                    bottom = p10[c] + (p11[c] - p10[c]) * tx
                    pixel.append(_half_up(top + (bottom - top) * ty))
            out_row.append(pixel)
        scaled_image.append(out_row)

    return scaled_image


def hybrid_scale(image3d: list[list[list[int]]], factor, interp_weight=None) -> list[list[list[int]]]:
    """Mixed nearest-neighbor / edge-directed interpolation rescale.
    ----

    Integer factor fast path produces verbatim block copies, pixel-identical
    to :func:`nearest_neighbor_scale`. For fractional factors the
    nearest-neighbor result (hard pixel feel) and the edge-directed result
    (smooth non-integer transitions) are blended per channel with convex
    weight ``interp_weight``:

        ``out = (1 - w) * nearest + w * edge_directed``

    where ``w = interp_weight``; default ``w`` is the fractional part of
    ``factor``, so the interpolation share grows with the non-integer
    remainder and integer factors stay purely nearest-neighbor.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param factor: positive int or float scaling factor, e.g. ``1.5``,
        ``2.5``, ``3.25``;
    :param interp_weight: ``None`` or float in range ``0..1``; ``0`` means
        pure nearest-neighbor, ``1`` pure edge-directed interpolation;
    :raises ValueError: factor zero, negative or non-finite; weight outside
        ``0..1``; empty image;
    :raises TypeError: factor of non-numeric type;
    :return: 3D nested list of the same structure as input, size
        ``round(Y * factor)`` x ``round(X * factor)``.
    :rtype: list[list[list[int]]]

    """

    factor = _validate_factor(factor)
    Y, X, Z = _validate_image(image3d)

    # ↓ integer factors: exact block replication, interpolation never runs
    if float(factor).is_integer():
        return nearest_neighbor_scale(image3d, factor)

    if interp_weight is None:
        interp_weight = factor - math.floor(factor)

    if isinstance(interp_weight, bool) or not isinstance(interp_weight, (int, float)):
        raise TypeError(f'interp_weight must be None, int or float, got {type(interp_weight).__name__}')
    interp_weight = float(interp_weight)
    if not (0.0 <= interp_weight <= 1.0):
        raise ValueError(f'interp_weight must be in range 0..1, got {interp_weight}')

    nn_image = nearest_neighbor_scale(image3d, factor)
    edi_image = edge_directed_scale(image3d, factor)

    if interp_weight == 0.0:
        return nn_image
    if interp_weight == 1.0:
        return edi_image

    w = interp_weight
    scaled_image: list[list[list[int]]] = []
    for nn_row, edi_row in zip(nn_image, edi_image):
        out_row = []
        for nn_pixel, edi_pixel in zip(nn_row, edi_row):
            # convex blend stays within source value range, half-up rounded
            out_row.append([_half_up(nn_pixel[c] + (edi_pixel[c] - nn_pixel[c]) * w) for c in range(Z)])
        scaled_image.append(out_row)

    return scaled_image


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
