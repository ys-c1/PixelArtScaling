"""
=======
ScaleNx
=======

------------------------------------------------------------------------------
Scale2x, Scale3x, Scale2xSFX and Scale3xSFX image rescaling for Python >=3.10.
------------------------------------------------------------------------------

:Abstract: Current module comprise **Scale2x**, **Scale3x**, **Scale2xSFX**
    and **Scale3xSFX** image rescaling functions, implemented in pure Python.

    Scale2x, Scale3x and Scale4x algorithms `[1]`_ were initially developed by
    `Andrea Mazzoleni`_ in 2001 AD for the sole purpose of upscaling
    low resolution screen output without introducing intermediate colors
    (*i.e.* blur) for old DOS games emulators `[2]`_ and
    similar narrow niche software.

    Later on *ca.* 2014 AD significantly improved Scale2xSFX and Scale3xSFX
    algorithms `[3]`_, providing better diagonals rendering, were introduced
    for the same very specific purpose.

    While screen scalers, based on algorithms above, are numerous,
    general purpose image rescaling wants still seem to be unsupplied.

    Due to severe demand for general purpose ScaleNx library,
    and apparent lack thereof, current general purpose pure Python
    `ScaleNx`_ implementation was developed.

Usage
-----

::

    from scalenx import scaleNx
    result_image = scaleNx(source_image, n, sfx)

where:

- **``source_image``**: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
- **``n``**: choice between Scale2* and Scale3* methods:
    - ``n=2``: Scale2x or Scale2xSFX;
    - ``n=3``: Scale3x or Scale3xSFX;

- **``sfx``**: choice between original ScaleNx and improved ScaleNxSFX methods:
    - ``sfx=False``: Scale2x or Scale3x;
    - ``sfx=True``: Scale2xSFX or Scale3xSFX.

.. note:: Function name **``scaleNx``** must contain capital **N**
    to avoid confusion with legacy file/module names.

Compatibility info
------------------

Current implementation is proven to work with CPython 3.10 and above,
and supposed to do so with any other Python version
understanding or ignoring type hints *etc.*

Copyright and redistribution
----------------------------

Current Python `ScaleNx`_ implementation is developed by
Ilya Razmanov (hereinafter referred to as "the Developer"), based on
algorithm descriptions `[1]`_, `[3]`_ by corresponding originators.

Changes introduced by the Developer for the purpose of
speed-up are entirely on his conscience.

Current implementation may be freely used, redistributed and
improved at will by anyone.
Sharing useful modifications with the Developer and lesser species
is your sacred duty to the Universe.

References
----------

`[1]`_. Original description of Scale2x and Scale3x algorithms by `Andrea Mazzoleni`_.

`[2]`_. DOSBox - DOS emulator, using Scale2x and Scale3x screen upscaling.

`[3]`_. Original Scale2xSFX and Scale3xSFX proposal, archived copy.

.. _[1]: https://www.scale2x.it/algorithm

.. _[2]: https://www.dosbox.com/

.. _[3]: https://web.archive.org/web/20160527015550/https://libretro.com/forums/archive/index.php?t-1655.html

.. _Andrea Mazzoleni: https://www.scale2x.it/authors

Resources
---------

The Developer's site: `The Toad's Slimy Mudhole`_;

.. _The Toad's Slimy Mudhole: https://dnyarri.github.io

`ScaleNx`_ explanations and illustrations page for current implementation;

.. _ScaleNx: https://dnyarri.github.io/scalenx.html

ScaleNx source repositories: `ScaleNx@Github`_, `ScaleNx@Gitflic`_;

.. _ScaleNx@Github: https://github.com/Dnyarri/PixelArtScaling

.. _ScaleNx@Gitflic: https://gitflic.ru/project/dnyarri/pixelartscaling

`Changelog`_ for current implementation.

.. _Changelog: https://github.com/Dnyarri/PixelArtScaling/blob/main/CHANGELOG.md

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__credits__ = 'Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.8.6.312'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'
__all__ = ['scaleNx', 'set_cache_maxsize', 'set_cache_policy', 'cache_clear', 'cache_info']

from collections import OrderedDict
from hashlib import sha256
from sys import getsizeof
from typing import Literal

from .scalenx import scale2x, scale3x
from .scalenxsfx import scale2x as scale2xsfx
from .scalenxsfx import scale3x as scale3xsfx

""" ╔═══════════════════════════════════════╗
    ║ Export result cache                   ║
    ║ ------------------------------------- ║
    ║ Repeatedly exporting the same image   ║
    ║ with the same parameters no longer    ║
    ║ recomputes the rescaling: results are ║
    ║ cached by (input hash, params hash).  ║
    ╚═══════════════════════════════════════╝ """

_RESULT_CACHE: OrderedDict[tuple[int, bool, str], list[list[list[int]]]] = OrderedDict()
"""Cached rescaling results, ordered for eviction. Keys are ``(n, sfx, input hash)``."""

_CACHE_MAXSIZE = 16
"""Maximal number of cached results. Configurable via `set_cache_maxsize`."""

_CACHE_POLICY = 'lru'
"""Eviction policy, ``'lru'`` or ``'fifo'``. Configurable via `set_cache_policy`."""

_CACHE_HITS = 0
"""Number of cache hits since last `cache_clear`."""

_CACHE_MISSES = 0
"""Number of cache misses since last `cache_clear`."""

_CACHE_POLICIES = ('lru', 'fifo')
"""Supported eviction policies."""


def _image_hash(image3d: list[list[list[int]]]) -> str:
    """Return content hash of ``image3d``, unambiguous for nested lists of int."""

    return sha256(repr(image3d).encode('utf-8')).hexdigest()


def _copy_image(image3d: list[list[list[int]]]) -> list[list[list[int]]]:
    """Fast deep copy of uniform image nested list; much faster than ``copy.deepcopy`` here."""

    return [[pixel[:] for pixel in row] for row in image3d]


def _deep_sizeof(obj, _seen: set | None = None) -> int:
    """Recursively measure memory footprint of nested lists/tuples, in bytes."""

    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)
    size = getsizeof(obj)
    if isinstance(obj, (list, tuple)):
        size += sum(_deep_sizeof(item, _seen) for item in obj)
    return size


def _evict() -> None:
    """Evict oldest/least recently used entries until cache fits `maxsize`."""

    while len(_RESULT_CACHE) > _CACHE_MAXSIZE:
        _RESULT_CACHE.popitem(last=False)  # front is oldest (FIFO) / least recently used (LRU)


def set_cache_maxsize(maxsize: int) -> None:
    """Set maximal number of cached rescaling results.

    :param int maxsize: new cache capacity, must be a positive integer.
        Excess entries are evicted immediately according to current policy.
    :raises ValueError: if ``maxsize`` is not a positive integer
        (in particular, cache of size 0 is not allowed).

    """

    global _CACHE_MAXSIZE
    if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
        raise ValueError(f'Cache maxsize must be a positive integer, got `{maxsize}`.\nCache of size 0 is not allowed; use `cache_clear()` to empty the cache instead.')
    _CACHE_MAXSIZE = maxsize
    _evict()


def set_cache_policy(policy: Literal['lru', 'fifo']) -> None:
    """Set cache eviction policy.

    :param str policy: ``'lru'`` (evict least recently used entry)
        or ``'fifo'`` (evict oldest entry).
    :raises ValueError: if ``policy`` is not one of supported policies.

    """

    global _CACHE_POLICY
    if policy not in _CACHE_POLICIES:
        raise ValueError(f'Unknown cache policy `{policy}`.\nOnly {_CACHE_POLICIES} policies are supported.')
    _CACHE_POLICY = policy


def cache_clear() -> None:
    """Empty the result cache and reset hit/miss statistics."""

    global _CACHE_HITS, _CACHE_MISSES
    _RESULT_CACHE.clear()
    _CACHE_HITS = _CACHE_MISSES = 0


def cache_info() -> dict:
    """Return current cache state and statistics as dict:

    ``maxsize`` - configured capacity;
    ``policy`` - configured eviction policy;
    ``size`` - number of cached results;
    ``hits``, ``misses`` - statistics since last `cache_clear`;
    ``memory`` - approximate memory held by cached results, in bytes.

    """

    return {
        'maxsize': _CACHE_MAXSIZE,
        'policy': _CACHE_POLICY,
        'size': len(_RESULT_CACHE),
        'hits': _CACHE_HITS,
        'misses': _CACHE_MISSES,
        'memory': _deep_sizeof(list(_RESULT_CACHE.values())),
    }


def scaleNx(source_image: list[list[list[int]]], n: Literal[2, 3], sfx: bool) -> list[list[list[int]]]:
    """ScaleNx image rescaling, configurable via ``n`` and ``sfx`` options.
    ----

    Results are cached by ``(n, sfx, source image content hash)`` key:
    repeated rescaling of the same image with the same parameters
    reuses the cached result instead of recomputing it.
    Cache capacity and eviction policy are configurable via
    `set_cache_maxsize` and `set_cache_policy`; see also
    `cache_info` and `cache_clear`.

    .. note:: Cached results are shared objects: repeated calls with
        the same input return the same nested list object.
        Treat the returned image as read-only; mutating it corrupts
        the cache. This matches the fact that rescaling results
        may alias source image pixels.

    :param source_image: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
    :type source_image: list[list[list[int]]]
    :param int n: ``2`` or ``3``, choice between Scale2* and Scale3* methods;
    :param bool sfx: choice between ScaleNx and ScaleNxSFX methods.
    :raises ValueError: Attempt to use nonexistent method ``n``.
    :return: rescaled image of the same type as ``source_image``.
    :rtype: list[list[list[int]]]

    """

    global _CACHE_HITS, _CACHE_MISSES

    key = (n, sfx, _image_hash(source_image))

    if key in _RESULT_CACHE:
        _CACHE_HITS += 1
        if _CACHE_POLICY == 'lru':
            _RESULT_CACHE.move_to_end(key)
        # ↓ Direct reuse of cached result; copying it would cost
        #   more than recomputing, defeating the cache purpose
        return _RESULT_CACHE[key]

    _CACHE_MISSES += 1

    if sfx:
        if n == 2:
            result = scale2xsfx(source_image)
        elif n == 3:
            result = scale3xsfx(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
    else:
        if n == 2:
            result = scale2x(source_image)
        elif n == 3:
            result = scale3x(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')

    # ↓ Cached copy must be independent since result pixels
    #   may alias source pixels, and source may be mutated later
    _RESULT_CACHE[key] = _copy_image(result)
    _evict()

    return result
