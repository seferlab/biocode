# Evolve an ensemble of programs for a Barabasi-Albert network (k=4)
sbt 'runMain LearnGraph --cfg examples/PrefAttach/PrefAttachExp.params'

# Analyze the ensemble and report the 'best' program
sbt 'runMain GrowCode.GrowGraph -g examples/PrefAttach/PrefAttach.edg -i 498 -o examples/PrefAttach/PrefAttach.grw examples/PrefAttach/PrefAttach.grw'

# Evolve an ensemble of programs for the PPI network
#sbt 'runMain LearnGraph --cfg examples/YeastPPI/YeastPPIExp.params'

# Analyze the ensemble and report the 'best' program
#sbt 'runMain GrowCode.GrowGraph -g examples/YeastPPI/YeastPPI.edg -i 2645 -o examples/YeastPPI/YeastPPIBest.grw examples/YeastPPI/YeastPPI.grw'

