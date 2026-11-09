#!/usr/bin/env bash
# Python version of ../runexample.sh -- run it from anywhere; it works in the
# project root so the paths inside the .params files resolve.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
JOBS="${JOBS:-4}"   # worker processes for fitness evaluation

# Evolve an ensemble of programs for a Barabasi-Albert network (k=4)
python3 -m biocode learn --cfg examples/PrefAttach/PrefAttachExp.params --jobs "$JOBS"

# Analyze the ensemble and report the 'best' program
python3 -m biocode grow -g examples/PrefAttach/PrefAttach.edg -i 498 -o examples/PrefAttach/PrefAttach.grw examples/PrefAttach/PrefAttach.grw

# Evolve an ensemble of programs for the PPI network
#python3 -m biocode learn --cfg examples/YeastPPI/YeastPPIExp.params --jobs "$JOBS"

# Analyze the ensemble and report the 'best' program
#python3 -m biocode grow -g examples/YeastPPI/YeastPPI.edg -i 2645 -o examples/YeastPPI/YeastPPIBest.grw examples/YeastPPI/YeastPPI.grw
