"""Power-law fit of a degree sequence (Clauset, Shalizi & Newman 2009).

``GrowGraph.analyzeDegreeDist`` in the Scala code piped the degrees to an
external helper, ``src/main/python/EvaluatePLFit.py``, which is not part of the
repository.  This module provides the same output -- ``alpha, xmin, L, p,
gof`` -- with a self-contained implementation:

* discrete maximum-likelihood exponent for every candidate ``xmin``
  (Hurwitz zeta normalisation),
* ``xmin`` chosen by the smallest Kolmogorov-Smirnov distance (``gof``),
* semi-parametric bootstrap p-value (bounded by a time budget).
"""

from __future__ import annotations

import math
import random as _random
import time

import numpy as np

# Bernoulli numbers B2, B4, ... for the Euler-Maclaurin tail
_B2K = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6]


def hurwitz_zeta(s, q, n_terms=20):
    """zeta(s, q) = sum_{k>=0} (q + k)^-s  for s > 1, q > 0."""
    k = np.arange(n_terms, dtype=float)
    total = float(np.sum((q + k) ** -s))
    a = q + n_terms
    total += a ** (1 - s) / (s - 1) + 0.5 * a ** -s
    fact = s * a ** (-s - 1)              # s (s+1) ... / a^(s+2j-1)
    for j, b in enumerate(_B2K, start=1):
        total += b / math.factorial(2 * j) * fact
        fact *= (s + 2 * j - 1) * (s + 2 * j) / (a * a)
    return total


def _fit_alpha(tail, xmin, lo=1.01, hi=6.0, tol=1e-5):
    n = len(tail)
    slog = float(np.sum(np.log(tail)))

    def nll(a):
        return n * math.log(hurwitz_zeta(a, xmin)) + a * slog

    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    alpha = (a + b) / 2
    return alpha, -nll(alpha)


def _ks(tail, xmin, alpha):
    xs, counts = np.unique(tail, return_counts=True)
    n = len(tail)
    emp_ge = 1.0 - (np.cumsum(counts) - counts) / n          # empirical P(X >= x)
    z0 = hurwitz_zeta(alpha, xmin)
    fit_ge = np.array([hurwitz_zeta(alpha, x) / z0 for x in xs])
    return float(np.max(np.abs(emp_ge - fit_ge)))


def fit(data, min_tail=10):
    """Return ``(alpha, xmin, loglik, ks)`` for positive integer ``data``."""
    x = np.sort(np.asarray(data, dtype=float))
    x = x[x >= 1]
    cands = np.unique(x)
    best = None
    for xmin in cands:
        tail = x[x >= xmin]
        if len(tail) < min_tail:
            break
        alpha, ll = _fit_alpha(tail, xmin)
        d = _ks(tail, xmin, alpha)
        if best is None or d < best[3]:
            best = (alpha, float(xmin), ll, d)
    if best is None:
        raise ValueError("not enough data for a power-law fit")
    return best


def _synthetic(x, alpha, xmin, rng):
    n = len(x)
    below = x[x < xmin]
    n_tail = sum(1 for _ in range(n) if rng.random() >= len(below) / n) if len(below) else n
    out = [below[int(rng.random() * len(below))] for _ in range(n - n_tail)] if len(below) else []
    for _ in range(n_tail):
        r = rng.random()
        out.append(math.floor((xmin - 0.5) * (1 - r) ** (-1 / (alpha - 1)) + 0.5))
    return np.array(out, dtype=float)


def plfit(data, n_boot=100, timeout=None, rng=None):
    """Fit and bootstrap.  Returns ``(alpha, xmin, L, p, gof)``.

    ``p`` is the fraction of synthetic data sets fitting worse than the data;
    if ``timeout`` (seconds) runs out, ``p`` uses the replicates finished so far
    (NaN if none).
    """
    rng = rng or _random.Random()
    x = np.asarray(data, dtype=float)
    alpha, xmin, ll, gof = fit(x)
    t_end = None if timeout is None else time.monotonic() + timeout
    worse = done = 0
    for _ in range(n_boot):
        if t_end is not None and time.monotonic() > t_end:
            break
        try:
            d = fit(_synthetic(x, alpha, xmin, rng))[3]
        except ValueError:
            continue
        done += 1
        if d >= gof:
            worse += 1
    p = worse / done if done else float("nan")
    return alpha, xmin, ll, p, gof
