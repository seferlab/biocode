"""Tiny console progress bar (stands in for ``net.robpatro.utils.console.ProgressBar``)."""

from __future__ import annotations

import sys


class ProgressBar:
    def __init__(self, total, mark="*", width=50, stream=None):
        self.total = max(int(total), 1)
        self.mark = mark
        self.width = width
        self.stream = stream or sys.stderr
        self._last = -1

    def update(self, i):
        frac = min(max(i / self.total, 0.0), 1.0)
        filled = int(frac * self.width)
        if filled == self._last:
            return
        self._last = filled
        bar = self.mark * filled + " " * (self.width - filled)
        self.stream.write("\r[%s] %3d%%" % (bar, int(frac * 100)))
        self.stream.flush()

    def done(self):
        self._last = -1
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()
