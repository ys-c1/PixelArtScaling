#!/usr/bin/env python3

"""
==========================
Content-aware edge padding
==========================

:Abstract: Current module comprise **content-aware boundary extension**
    wrapper around the Scale2x / Scale3x rescaling functions, implemented
    in pure Python without any third-party dependency.

Problem
-------

The plain ScaleNx algorithms use a *repeat edge* mode at the image border:
when the stencil kernel reaches a fully transparent pixel outside the
opaque content, it has no real neighbour and reuses the edge pixel,
so after upscaling the texture edge gets smeared into a row of blurred
(*"stretched"*) pixels.

Solution
--------

``content_aware_scaleNx``:

1. detects the bounding box of the opaque (non-zero alpha) content;
2. extends the image outside that bounding box by a configurable amount,
   filling the extension with either a mirror reflection or a replicated
   copy of the border content, so every stencil kernel around the real
   edge gets real neighbours;
3. upscales the extended image with the unmodified ScaleNx function;
4. crops the scaled extension back, returning an image of exactly the
   same size as a direct rescaling of the source.

Images without an alpha channel (``L``, ``RGB``) are fully opaque by
definition, so their bounding box is the whole image. Fully transparent
images (alpha channel present but no opaque pixel) raise ``ValueError``.

Usage
-----

::

    from scalenx.contentaware import content_aware_scaleNx
    result, stats = content_aware_scaleNx(source_image, n=2, sfx=False,
                                          pad_mode='mirror', pad_ratio=0.25)

The direct rescaling output is never altered: with ``pad_ratio=0`` the
function returns the ScaleNx result pixel-by-pixel identical to a direct
call, and the underlying ScaleNx modules are reused unchanged.

References
----------

See package ``scalenx`` documentation for the Scale2x, Scale3x,
Scale2xSFX and Scale3xSFX algorithms themselves.

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2026 Ilya Razmanov'
__credits__ = ['Ilya Razmanov']
__license__ = 'unlicense'
__version__ = '2026.11.18.1'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'

from time import perf_counter
from typing import Literal

""" ╔══════════════════════════════════════════════╗
    ║ Opaque content bounding box detection         ║
    ╚══════════════════════════════════════════════╝ """


def opaque_bbox(image3d: list[list[list[int]]]) -> tuple[int, int, int, int]:
    """Return bounding box ``(top, bottom, left, right)`` of opaque content.

    Bounding box limits are inclusive indices of the outermost pixels
    whose alpha channel is non-zero. For images without an alpha channel
    (1-channel ``L`` or 3-channel ``RGB``) every pixel is opaque and the
    whole image box is returned.

    :param image3d: 3D nested list (image) of lists (rows) of lists
        (pixels) of int (channel values), channels order LA or RGBA;
    :type image3d: list[list[list[int]]]
    :raises ValueError: if the image has an alpha channel but no opaque
        pixel (fully transparent image), or the image is empty.
    :return: ``(top, bottom, left, right)`` inclusive bounding box.
    :rtype: tuple[int, int, int, int]

    """

    Y = len(image3d)
    if Y == 0 or len(image3d[0]) == 0:
        raise ValueError('Cannot detect opaque content in an empty image.')

    X = len(image3d[0])
    channels = len(image3d[0][0])
    has_alpha = channels in (2, 4)  # LA or RGBA; L and RGB are fully opaque

    if not has_alpha:
        return 0, Y - 1, 0, X - 1

    top = Y
    bottom = -1
    left = X
    right = -1

    for y in range(Y):
        row = image3d[y]
        for x in range(X):
            if row[x][-1] > 0:  # alpha channel is always the last one
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y
                if x < left:
                    left = x
                if x > right:
                    right = x

    if bottom < 0:  # not a single opaque pixel found
        raise ValueError('No opaque content found: image is fully transparent, nothing to extend and scale.')

    return top, bottom, left, right


""" ╔══════════════════════════════════════════════╗
    ║ Extension padding: mirror / replicate fill    ║
    ╚══════════════════════════════════════════════╝ """


def _replicate_index(coord: int, limit: int) -> int:
    """Clamp out-of-range coordinate to the nearest border index."""

    if coord < 0:
        return 0
    if coord >= limit:
        return limit - 1
    return coord


def _mirror_index(coord: int, limit: int) -> int:
    """Reflect out-of-range coordinate back into ``0..limit-1``.

    Repeats the reflection (period ``2 * limit``) so any coordinate,
    however far outside, is mapped back into the source range.
    Edge pixels themselves are not duplicated ("reflect" mode).
    """

    period = 2 * limit
    coord = coord % period
    if coord >= limit:
        coord = period - 1 - coord
    return coord


def pad_image(
    image3d: list[list[list[int]]],
    bbox: tuple[int, int, int, int],
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    pad_mode: Literal['mirror', 'replicate'] = 'mirror',
) -> list[list[list[int]]]:
    """Build a new image extended around ``bbox`` with mirror/replicate fill.

    Only the rectangle of the source inside the opaque bounding box is
    used as extension material, matching the "extend outside the content
    boundary" semantics.

    :param image3d: source 3D nested list;
    :param bbox: ``(top, bottom, left, right)`` inclusive opaque box;
    :param pad_top, pad_bottom, pad_left, pad_right: non-negative extension
        width in pixels on each side;
    :param pad_mode: ``'mirror'`` reflects border content,
        ``'replicate'`` copies the nearest border row/column;
    :raises ValueError: unknown ``pad_mode``.
    :return: new padded 3D nested list (independent copy).
    :rtype: list[list[list[int]]]

    """

    if pad_mode not in ('mirror', 'replicate'):
        raise ValueError(f"Unknown pad_mode `{pad_mode}`.\nOnly 'mirror' and 'replicate' padding modes are supported.")

    top, bottom, left, right = bbox
    box_h = bottom - top + 1
    box_w = right - left + 1

    map_index = _mirror_index if pad_mode == 'mirror' else _replicate_index

    Y, X = (len(image3d), len(image3d[0]))
    new_Y = Y + pad_top + pad_bottom
    new_X = X + pad_left + pad_right

    padded: list[list[list[int]]] = []
    for new_y in range(new_Y):
        # Coordinate inside the source image coordinate system.
        src_y = new_y - pad_top
        # Restrict the extension material to the opaque bounding box.
        box_y = map_index(src_y - top, box_h)
        source_row = image3d[top + box_y]
        new_row: list[list[int]] = []
        for new_x in range(new_X):
            src_x = new_x - pad_left
            box_x = map_index(src_x - left, box_w)
            new_row.append(list(source_row[left + box_x]))  # fresh pixel copy
        padded.append(new_row)

    return padded


""" ╔══════════════════════════════════════════════╗
    ║ Content-aware ScaleNx                         ║
    ╚══════════════════════════════════════════════╝ """


def content_aware_scaleNx(
    source_image: list[list[list[int]]],
    n: Literal[2, 3] = 2,
    sfx: bool = False,
    pad_mode: Literal['mirror', 'replicate'] = 'mirror',
    pad_ratio: float = 0.25,
) -> tuple[list[list[list[int]]], dict]:
    """Upscale with content-aware boundary extension.

    Pipeline: detect opaque bounding box -> extend outside the box with
    ``pad_mode`` fill by ``pad_ratio`` of the box size -> run the
    unmodified ScaleNx rescaling on the extended image -> crop the
    extension back, yielding exactly ``n`` times the source dimensions.

    :param source_image: source image 3D nested list; coordinate system
        match Photoshop, *i.e.* origin is top left corner, channels
        order is LA or RGBA from 0 to top;
    :type source_image: list[list[list[int]]]
    :param int n: ``2`` or ``3``, choice between Scale2* and Scale3*;
    :param bool sfx: choice between original ScaleNx (``False``) and
        improved ScaleNxSFX (``True``);
    :param str pad_mode: ``'mirror'`` (reflect border content) or
        ``'replicate'`` (copy the nearest border row/column);
    :param float pad_ratio: extension width on each side as a fraction
        of the opaque box width/height, ``0.0`` means no extension
        (direct rescaling, pixel-by-pixel identical output);
    :raises ValueError: invalid ``n``, unknown ``pad_mode``,
        non-numeric or negative ``pad_ratio``, or a fully transparent
        source without opaque content;
    :return: ``(result_image, stats)`` where ``result_image`` is the
        cropped scaled image and ``stats`` is a dict with the bounding
        box, per-side padding, padded image size, extension pixel ratio
        (fraction of padded pixels in the extended image) and timing
        of each pipeline stage;
    :rtype: tuple[list[list[list[int]]], dict]

    """

    # Imports are local so the module stays importable even if the package
    # layout changes, and no public symbol of scalenx is redefined.
    from .scalenx import scale2x, scale3x
    from .scalenxsfx import scale2x as scale2xsfx
    from .scalenxsfx import scale3x as scale3xsfx

    if n not in (2, 3):
        raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
    if pad_mode not in ('mirror', 'replicate'):
        raise ValueError(f"Unknown pad_mode `{pad_mode}`.\nOnly 'mirror' and 'replicate' padding modes are supported.")
    if isinstance(pad_ratio, bool) or not isinstance(pad_ratio, (int, float)):
        raise ValueError(f'pad_ratio must be a non-negative number, got `{pad_ratio!r}`.')
    if pad_ratio < 0:
        raise ValueError(f'pad_ratio must be non-negative, got `{pad_ratio}`.')

    Y, X = (len(source_image), len(source_image[0]))

    # ↓ Stage 1: opaque content bounding box (raises on empty/transparent input)
    top, bottom, left, right = opaque_bbox(source_image)
    box_h = bottom - top + 1
    box_w = right - left + 1

    # ↓ Stage 2: extension width per side, proportional to the box size.
    #   Asymmetric split keeps the total equal to the rounded 2*ratio size.
    total_pad_y = round(pad_ratio * box_h)
    total_pad_x = round(pad_ratio * box_w)
    pad_top = total_pad_y // 2
    pad_bottom = total_pad_y - pad_top
    pad_left = total_pad_x // 2
    pad_right = total_pad_x - pad_left

    if total_pad_y == 0 and total_pad_x == 0:
        # No extension requested: fall back to the direct, unmodified
        # rescaling so the output is guaranteed pixel-by-pixel identical.
        if sfx:
            result = scale2xsfx(source_image) if n == 2 else scale3xsfx(source_image)
        else:
            result = scale2x(source_image) if n == 2 else scale3x(source_image)
        stats = {
            'n': n,
            'sfx': sfx,
            'pad_mode': pad_mode,
            'pad_ratio': float(pad_ratio),
            'bbox': (top, bottom, left, right),
            'pad': (0, 0, 0, 0),
            'padded_size': (Y, X),
            'extension_ratio': 0.0,
            'timing': {'pad': 0.0, 'scale': 0.0, 'crop': 0.0},
        }
        return result, stats

    t0 = perf_counter()
    padded = pad_image(source_image, (top, bottom, left, right), pad_top, pad_bottom, pad_left, pad_right, pad_mode)
    t1 = perf_counter()
    padded_Y, padded_X = (len(padded), len(padded[0]))

    # ↓ Stage 3: unmodified ScaleNx rescaling of the extended image
    if sfx:
        scaled_padded = scale2xsfx(padded) if n == 2 else scale3xsfx(padded)
    else:
        scaled_padded = scale2x(padded) if n == 2 else scale3x(padded)
    t2 = perf_counter()

    # ↓ Stage 4: crop the scaled extension back. Integer padding makes the
    #   mapping exact, so the result size equals the direct n*x rescaling.
    out_Y, out_X = (n * Y, n * X)
    off_y, off_x = (n * pad_top, n * pad_left)
    result: list[list[list[int]]] = []
    for y in range(out_Y):
        source_row = scaled_padded[off_y + y]
        result.append([list(source_row[off_x + x]) for x in range(out_X)])
    t3 = perf_counter()

    extension_pixels = padded_Y * padded_X - Y * X
    stats = {
        'n': n,
        'sfx': sfx,
        'pad_mode': pad_mode,
        'pad_ratio': float(pad_ratio),
        'bbox': (top, bottom, left, right),
        'pad': (pad_top, pad_bottom, pad_left, pad_right),
        'padded_size': (padded_Y, padded_X),
        'extension_ratio': extension_pixels / (padded_Y * padded_X),
        'timing': {'pad': t1 - t0, 'scale': t2 - t1, 'crop': t3 - t2},
    }
    return result, stats


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
