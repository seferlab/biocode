"""Port of ``Histogram.scala``.

Binning rule (kept identical so graph "shape" scores match the Scala code):

* bin width ``h = (hi - lo) / nbins`` if ``drange`` is given, otherwise
  ``(max - min) / nbins``;
* a value ``v`` goes to bin ``max(0, floor((v - min(vals) - 1e-10) / h))`` --
  note that the offset is the *data* minimum even when ``drange`` is given;
* with ``normed=True`` every value adds ``1 / (h * n)`` instead of 1.

The Scala version throws if a value falls past the last bin; here such values
are put in the last bin instead.
"""

from __future__ import annotations

import math

import numpy as np


class Histogram:
    def __init__(self, vals, drange=None, nbins=10, normed=False):
        vals = [float(v) for v in vals]
        if not vals:
            raise ValueError("Histogram of an empty sequence")
        self.nbins = nbins
        min_val, max_val = min(vals), max(vals)
        if drange is not None:
            self.h = (drange[1] - drange[0]) / nbins
        else:
            self.h = (max_val - min_val) / nbins
        inv_norm = 1.0 / (self.h * len(vals)) if normed else 1.0
        hist = np.zeros(nbins)
        b = 1e-10
        h = self.h
        for v in vals:
            if h > 0:
                i = max(0, int(math.floor((v - min_val - b) / h)))
            else:
                i = 0
            hist[min(i, nbins - 1)] += inv_norm
        self.hist = hist
