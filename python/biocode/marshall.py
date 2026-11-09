"""Object (de)serialisation (port of ``Marshall.scala`` and ``FileUtils.scala``).

The Scala code used Java serialisation for the evolved population; the Python
port uses :mod:`pickle`.  Only load population files you produced yourself --
unpickling runs code from the file.
"""

from __future__ import annotations

import pickle

from .program import HEADER


def write_to_file(fname, obj):
    with open(fname, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def is_ascii(fname):
    """True if ``fname`` is a text GrowCode program (starts with ``; GROWCODE``)."""
    with open(fname, "rb") as f:
        return f.read(len(HEADER)) == HEADER.encode()


def is_binary(fname):
    return not is_ascii(fname)


writeToFile = write_to_file
isAscii = is_ascii
isBinary = is_binary
