"""GrowCode instruction set (port of ``Operation.scala``).

Every instruction is a small immutable value object with

* ``Str``       -- the text written to ``.grw`` program files,
* ``short_str`` -- a compact one/two character form,
* ``apply(m)``  -- executes the instruction on a :class:`~biocode.machine.Machine`.

Instructions compare equal when they have the same type and parameters (Scala
case-class semantics).  Real-valued parameters are stored as 32-bit floats,
as in the Scala code, so programs print the same way.
"""

from __future__ import annotations

import random as _random

import numpy as np


def f32(x):
    """Round a Python float to single precision (Scala ``Float``)."""
    return float(np.float32(x))


def fmt_f32(x):
    """Shortest text that identifies the float32 value (like Java's Float.toString)."""
    return str(np.float32(x))


class MachineOperation:
    __slots__ = ()
    name = ""          # token used in program files
    short = ""         # short string

    # -- value semantics -----------------------------------------------------
    def params(self):
        return ()

    def __eq__(self, other):
        return type(self) is type(other) and self.params() == other.params()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self).__name__,) + self.params())

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(map(repr, self.params())))

    def __reduce__(self):
        return (type(self), self.params())

    # -- text forms ---------------------------------------------------------
    @property
    def Str(self):
        return self.name

    @property
    def short_str(self):
        return self.short

    def __str__(self):
        return self.Str

    def apply(self, m):
        raise NotImplementedError

    def reset(self):
        pass


class NoOp(MachineOperation):
    __slots__ = ()
    name, short = "noop", "."

    def apply(self, m):
        m.op_nop()


class CopyOp(MachineOperation):
    __slots__ = ()
    name, short = "copy", ":"

    def apply(self, m):
        m.op_copy()


class DetachOp(MachineOperation):
    __slots__ = ()
    name, short = "detach", "D"

    def apply(self, m):
        m.op_detach()


class AttachOp(MachineOperation):
    __slots__ = ()
    name, short = "attach", "A"

    def apply(self, m):
        m.op_attach()


class RonPaulOp(MachineOperation):
    """Clears the influence (label) memory.  Written as ``clearinf``."""
    __slots__ = ()
    name, short = "clearinf", "C"

    def apply(self, m):
        m.op_ron_paul()


class SaveOp(MachineOperation):
    __slots__ = ()
    name, short = "save", "S"

    def apply(self, m):
        m.op_save()


class LoadOp(MachineOperation):
    __slots__ = ()
    name, short = "load", "L"

    def apply(self, m):
        m.op_load()


class SwapOp(MachineOperation):
    __slots__ = ()
    name, short = "swap", "W"

    def apply(self, m):
        m.op_swap()


class CreateOp(MachineOperation):
    __slots__ = ()
    name, short = "create", "/"

    def apply(self, m):
        m.op_create()


class REdgeOp(MachineOperation):
    __slots__ = ()
    name, short = "redge", "E"

    def apply(self, m):
        m.op_redge()


class RNodeOp(MachineOperation):
    __slots__ = ()
    name, short = "rnode", "V"

    def apply(self, m):
        m.op_rnode()


class NeighOp(MachineOperation):
    __slots__ = ()
    name, short = "neigh", "N"

    def apply(self, m):
        m.op_neigh()


class NewOp(MachineOperation):
    __slots__ = ()
    name, short = "new", "*"

    def apply(self, m):
        m.op_new()


class ClearOp(MachineOperation):
    __slots__ = ()
    name, short = "clear", "-"

    def apply(self, m):
        m.op_clear()


class LoopOp(MachineOperation):
    """End-of-program marker; behaves as a no-op (as in the Scala code)."""
    __slots__ = ()
    name, short = "loop", "]"

    def apply(self, m):
        m.op_nop()


class TruthOp(MachineOperation):
    __slots__ = ("p1", "p2", "p3", "p4")
    short = "T"

    def __init__(self, p1, p2, p3, p4):
        self.p1, self.p2, self.p3, self.p4 = f32(p1), f32(p2), f32(p3), f32(p4)

    def params(self):
        return (self.p1, self.p2, self.p3, self.p4)

    @property
    def Str(self):
        return "truth " + " ".join(fmt_f32(p) for p in self.params())

    def apply(self, m):
        m.op_truth(self.p1, self.p2, self.p3, self.p4)


class StepOp(MachineOperation):
    __slots__ = ("d", "_d")

    def __init__(self, d):
        self.d = int(d)
        self._d = self.d

    def params(self):
        return (self.d,)

    @property
    def short_str(self):
        return "[%d" % self._d

    @property
    def Str(self):
        return "step %d" % self._d

    def apply(self, m):
        m.op_step(self)

    def reset(self):
        self.d = self._d


class RewindOp(MachineOperation):
    __slots__ = ("n_instr", "n_loop")

    def __init__(self, n_instr, n_loop):
        self.n_instr, self.n_loop = int(n_instr), int(n_loop)

    def params(self):
        return (self.n_instr, self.n_loop)

    @property
    def short_str(self):
        return "<-(%d,%d)" % (self.n_instr, self.n_loop)

    @property
    def Str(self):
        return "rewind %d,%d" % (self.n_instr, self.n_loop)

    def apply(self, m):
        m.op_rewind(self.n_instr, self.n_loop)


class WalkOp(MachineOperation):
    __slots__ = ("d",)
    short = "~"

    def __init__(self, d):
        self.d = int(d)

    def params(self):
        return (self.d,)

    @property
    def Str(self):
        return "walk %d" % self.d

    def apply(self, m):
        m.op_walk(self.d)


class SetOp(MachineOperation):
    __slots__ = ("d",)

    def __init__(self, d):
        self.d = int(d)

    def params(self):
        return (self.d,)

    @property
    def short_str(self):
        return ">(%d)" % self.d

    @property
    def Str(self):
        return "set %d" % self.d

    def apply(self, m):
        m.op_set(self.d)


class InflOp(MachineOperation):
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = f32(p)

    def params(self):
        return (self.p,)

    @property
    def short_str(self):
        return "!(%f)" % self.p

    @property
    def Str(self):
        return "infl " + fmt_f32(self.p)

    def apply(self, m):
        m.op_infl(self.p)


class ScoinOp(MachineOperation):
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = f32(p)

    def params(self):
        return (self.p,)

    @property
    def short_str(self):
        return "@(%f)" % self.p

    @property
    def Str(self):
        return "scoin " + fmt_f32(self.p)

    def apply(self, m):
        m.op_scoin(self.p)


# ---------------------------------------------------------------------------
# Parsing and random generation (companion object ``MachineOperation``)
# ---------------------------------------------------------------------------

_NULLARY = {
    "noop": NoOp, "copy": CopyOp, "detach": DetachOp, "attach": AttachOp,
    "clearinf": RonPaulOp,
    "ronpaul": RonPaulOp,   # older programs (e.g. YeastPPIBest.grw) use this name
    "save": SaveOp, "load": LoadOp, "swap": SwapOp, "create": CreateOp,
    "redge": REdgeOp, "rnode": RNodeOp, "neigh": NeighOp, "loop": LoopOp,
    "new": NewOp, "clear": ClearOp,
}


def from_token_list(toks):
    """Build an instruction from the tokens of one program line."""
    toks = [t.lower() for t in toks]
    try:
        head, rest = toks[0], toks[1:]
        if head in _NULLARY and not rest:
            return _NULLARY[head]()
        if head == "truth":
            return TruthOp(*(float(x) for x in rest[:4]))
        if head == "step":
            return StepOp(int(rest[0]))
        if head == "rewind":
            a, b = rest[0].split(",")[:2]
            return RewindOp(int(a), int(b))
        if head == "walk":
            return WalkOp(int(rest[0]))
        if head == "set":
            return SetOp(int(rest[0]))
        if head == "infl":
            return InflOp(float(rest[0]))
        if head == "scoin":
            return ScoinOp(float(rest[0]))
    except (IndexError, ValueError):
        pass
    raise ValueError("Failed to parse cmd: %s" % " ".join(toks))


# The instructions the genetic search draws from (``opArray`` in Scala).
# NoOp, CopyOp, NeighOp, TruthOp, StepOp, WalkOp and LoopOp are deliberately
# left out, exactly as in the original.
OP_ARRAY = (
    DetachOp, RonPaulOp, AttachOp, SaveOp, LoadOp, SwapOp, CreateOp,
    REdgeOp, RNodeOp, RewindOp, SetOp, InflOp, NewOp, ClearOp, ScoinOp,
)


def random_operation(rng=_random):
    """A uniformly chosen instruction with random parameters."""
    cls = OP_ARRAY[rng.randrange(len(OP_ARRAY))]
    if cls is TruthOp:
        return TruthOp(rng.random(), rng.random(), rng.random(), rng.random())
    if cls is StepOp:
        return StepOp(rng.randrange(10) + 1)
    if cls is RewindOp:
        return RewindOp(rng.randrange(10) + 1, rng.randrange(10) + 1)
    if cls is WalkOp:
        return WalkOp(rng.randrange(10) + 1)
    if cls is SetOp:
        return SetOp(rng.randrange(10) + 1)
    if cls is InflOp:
        return InflOp(rng.random())
    if cls is ScoinOp:
        return ScoinOp(rng.random())
    return cls()
