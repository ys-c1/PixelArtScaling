#!/usr/bin/env python3
"""
Small self-contained demo for the table-driven resampler.

Run with the bundled stdlib only::

    python3 -I tests/demo_kernel_table.py

It shows:
  1. the built-in kernel table and how a kernel entry encodes
     support radius + weight function + normalization (+ separability);
  2. adding a brand-new kernel by appending ONE table entry, with no
     change to the main resampling loop;
  3. the difference between a separable kernel (two 1-D passes) and a
     non-separable kernel (one 2-D pass), on a tiny synthetic image.
"""

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scalenx.resample import KERNELS, Kernel, get_kernel, resample  # noqa: E402


def show(title, image3d):
    print(f'  {title} ({len(image3d)}x{len(image3d[0])}):')
    for row in image3d:
        print('   ', [px[0] for px in row])


def main() -> None:
    # A tiny 1-channel synthetic gradient (no external files needed).
    src = [[[0], [100]], [[200], [255]]]
    print('Kernel table entries (name -> radius, normalization, separable, engine):')
    for name, k in KERNELS.items():
        print(f'  {name:18s} r={k.support:g}  norm={k.normalize:5s}  '
              f'separable={k.separable}  exact={k.exact}')

    print('\n[Separable] tent/bilinear, factor 2 (two 1-D passes):')
    show('source', src)
    show('tent 2x', resample(src, 2, 'tent'))

    print('\n[Non-separable] radial_tent, factor 2 (one 2-D pass):')
    show('radial_tent 2x', resample(src, 2, 'radial_tent'))

    # ---------------------------------------------------------------
    # Add a NEW kernel with a single table entry. The main loop in
    # resample() is not touched; it already dispatches on the entry.
    # ---------------------------------------------------------------
    def _w_flat_top(d) -> Fraction:
        """Flat-top trapezoid, radius 1.5 - a simple distinct demo profile."""

        ad = d if d >= 0 else -d
        if ad < Fraction(1, 2):
            return Fraction(1)
        if ad < Fraction(3, 2):
            return (Fraction(3, 2) - ad)
        return Fraction(0)

    KERNELS['flat_top'] = Kernel(
        'flat_top',
        support=1.5,
        weight=_w_flat_top,
        normalize='local',
        separable=True,
        exact=True,
    )
    print('\n[New kernel] one appended entry, no loop edit:')
    print(' ', get_kernel('flat_top'))
    show('flat_top 2x', resample(src, 2, 'flat_top'))

    print('\nDemo finished.')


if __name__ == '__main__':
    main()
