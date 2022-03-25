from utils import run
from States import InitStates, cpmcutoff, ApplyCPM, ComputeNormFactors, LinearReg, SSE_MeanLogCount
# , ApplyCPM, ComputeNormFactors, cpmcutoff, LinearReg, SSE_MeanLogCount

# from States.InitStates import LocalMean, GlobalMean
# from States.ApplyCPM import ApplyCPM, AggregateGeneNames
# from States.ComputeNormFactors import ComputeNormFactors, AggregateLibSizes
# from States.cpmcutoff import CPMCutOff, CutOffAggregation
# from States.LinearReg import LinearRegression, AggregateRegression
# from States.SSE_MeanLogCount import SSE, AggregateSSE

if __name__ == '__main__':
    run()
