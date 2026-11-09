"""Global (Needleman-Wunsch style) alignment of two strings
(port of ``GlobalAlignment.scala``).

Costs: match -1, mismatch +1, gap +1; the score is minimised.
"""

from __future__ import annotations


class GlobalAlignment:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.table = None
        self.traceback = None

    def align(self):
        a, b = self.a, self.b
        n, m = len(a), len(b)
        table = [[0.0] * (m + 1) for _ in range(n + 1)]
        tb = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            table[i][0] = float(i)
        for j in range(1, m + 1):
            table[0][j] = float(j)
        for i in range(1, n + 1):
            ai = a[i - 1]
            prev, cur, tbi = table[i - 1], table[i], tb[i]
            for j in range(1, m + 1):
                c0 = (-1 if ai == b[j - 1] else 1) + prev[j - 1]   # extend
                c1 = 1 + prev[j]                                     # gap in b
                c2 = 1 + cur[j - 1]                                  # gap in a
                best = min(c0, c1, c2)
                cur[j] = best
                tbi[j] = 0 if best == c0 else (1 if best == c1 else 2)
        self.table, self.traceback = table, tb
        return self

    @property
    def score(self):
        return self.table[len(self.a)][len(self.b)]

    def alignment(self):
        """The two aligned strings (gaps as ``-``)."""
        a, b, tb = self.a, self.b, self.traceback
        i, j = len(a), len(b)
        aa, bb = [], []
        while i != 0 and j != 0:
            t = tb[i][j]
            if t == 0:
                aa.append(a[i - 1]); bb.append(b[j - 1]); i -= 1; j -= 1
            elif t == 1:
                aa.append(a[i - 1]); bb.append("-"); i -= 1
            else:
                aa.append("-"); bb.append(b[j - 1]); j -= 1
        return "".join(reversed(aa)), "".join(reversed(bb))

    def print_alignment(self):
        x, y = self.alignment()
        print(x)
        print(y)


def alignment_score(a, b):
    """Score only, with O(len(b)) memory."""
    m = len(b)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            c0 = (-1 if ai == b[j - 1] else 1) + prev[j - 1]
            c1 = 1 + prev[j]
            c2 = 1 + cur[j - 1]
            cur[j] = c0 if c0 <= c1 and c0 <= c2 else (c1 if c1 <= c2 else c2)
        prev = cur
    return prev[m]
