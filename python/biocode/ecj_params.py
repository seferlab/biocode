"""Reader for ECJ ``.params`` files.

Supports what the BioCode example files use: ``key = value`` lines, ``#``
comments, CRLF line endings and ``parent.N = file`` includes (resolved
relative to the including file).  Later definitions override earlier ones and
a file overrides its parents, as in ECJ.
"""

from __future__ import annotations

import os


class ParameterDatabase:
    def __init__(self, values=None, path=None):
        self.values = dict(values or {})
        self.path = path

    @classmethod
    def from_file(cls, path):
        values = {}
        cls._load(os.path.abspath(path), values, set())
        return cls(values, os.path.abspath(path))

    @classmethod
    def _load(cls, path, values, seen):
        if path in seen:
            return
        seen.add(path)
        own = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                else:
                    parts = line.split(None, 1)
                    k, v = parts[0], (parts[1] if len(parts) > 1 else "")
                own[k.strip()] = v.strip()
        # parents first, so this file's values win
        i = 0
        while "parent.%d" % i in own:
            ppath = own["parent.%d" % i]
            if not os.path.isabs(ppath):
                ppath = os.path.join(os.path.dirname(path), ppath)
            cls._load(os.path.abspath(ppath), values, seen)
            i += 1
        values.update(own)

    # -- access ---------------------------------------------------------
    def set(self, key, value):
        self.values[key] = str(value)

    def exists(self, key):
        return key in self.values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_string(self, key, default=None):
        return self.values.get(key, default)

    def get_int(self, key, default=None):
        v = self.values.get(key)
        return default if v is None else int(v)

    def get_float(self, key, default=None):
        v = self.values.get(key)
        return default if v is None else float(v)

    def get_bool(self, key, default=False):
        v = self.values.get(key)
        if v is None:
            return default
        return v.lower() in ("true", "t", "yes", "1")

    def first(self, keys, default=None):
        """Value of the first key in ``keys`` that is defined."""
        for k in keys:
            if k in self.values:
                return self.values[k]
        return default

    def resolve_path(self, value):
        """ECJ convention: ``$file`` is relative to the parameter file's folder,
        anything else is relative to the current directory."""
        if value is None:
            return None
        if value.startswith("$"):
            base = os.path.dirname(self.path) if self.path else os.getcwd()
            return os.path.join(base, value[1:])
        return value
