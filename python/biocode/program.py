"""GrowCode programs (port of ``Program.scala``).

Program file format (``.grw``)::

    ; GROWCODE
    rnode
    infl 0.7862678
    rewind 3,1
    ...

Everything after ``;`` on a line is a comment.  As in the Scala code,
:meth:`Program.from_file` wraps the instructions in ``step 1`` ... ``loop``
(both are effectively no-ops).
"""

from __future__ import annotations

import os

from .operations import LoopOp, StepOp, from_token_list

HEADER = "; GROWCODE"


class Program:
    def __init__(self, pcode):
        self.pcode = list(pcode)
        self.properties = {}

    @property
    def shortStr(self):
        """Newline separated instruction text (what Scala's ``shortStr`` returns)."""
        return "\n".join(op.Str for op in self.pcode)

    short_str = shortStr

    @property
    def compact_str(self):
        """One/two characters per instruction (Scala's commented-out ``shortStr``)."""
        return "".join(op.short_str for op in self.pcode)

    def __len__(self):
        return len(self.pcode)

    @property
    def length(self):
        return len(self.pcode)

    def instr(self, pc):
        return self.pcode[pc]

    def __repr__(self):
        return "Program(%d ops)" % len(self.pcode)

    # -- I/O ---------------------------------------------------------------
    @staticmethod
    def parse_lines(lines, wrap=True, verbose=False):
        code = [StepOp(1)] if wrap else []
        for raw in lines:
            cidx = raw.find(";")
            line = (raw[:cidx] if cidx != -1 else raw).lower().strip()
            if line:
                toks = line.split()
                if verbose:
                    print(toks)
                code.append(from_token_list(toks))
        if wrap:
            code.append(LoopOp())
        return Program(code)

    @staticmethod
    def from_file(filename, wrap=True, verbose=False):
        with open(filename) as fh:
            return Program.parse_lines(fh, wrap=wrap, verbose=verbose)

    def to_text(self):
        return HEADER + "\n" + "\n".join(op.Str for op in self.pcode)

    def to_file(self, filename):
        """Write the program in ``.grw`` text form (creates parent folders)."""
        d = os.path.dirname(filename)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(filename, "w") as f:
            f.write(self.to_text())
