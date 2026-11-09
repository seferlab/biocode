"""Classical multidimensional scaling (port of ``MDS.scala``).

As in the Scala code the matrix ``D`` is double-centred directly
(``B = -1/2 J D J``).  Pass squared distances for textbook classical MDS.

The Scala version selected eigenvector *rows* and unsorted eigenvalues when
building the embedding; this port uses the sorted eigenvalues and the matching
eigenvector columns, which is what the method intends.
"""

from __future__ import annotations

import numpy as np


def classical_mds(D, k):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D @ J
    B = (B + B.T) / 2.0
    vals, vecs = np.linalg.eigh(B)
    s = np.where(vals < 1e-10, 0.0, vals)
    order = np.argsort(s)[::-1]
    s, vecs = s[order], vecs[:, order]
    sq = s ** 2
    error = float(sq.sum() - sq[:k].sum())
    embedding = vecs[:, :k] @ np.diag(np.sqrt(s[:k]))
    return embedding, error


classicalMDS = classical_mds
