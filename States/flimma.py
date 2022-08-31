from FeatureCloud.app.engine.app import Role, AppState, app_state
from .InitStates import LocalMean, GlobalMean
from .cpmcutoff import CPMCutOff, CutOffAggregation
from .ApplyCPM import ApplyCPM, AggregateGeneNames
from .ComputeNormFactors import ComputeNormFactors, AggregateLibSizes
from .LinearReg import LinearRegression, AggregateRegression
from .SSE_MeanLogCount import SSE, AggregateSSE, WriteResults


@app_state(name='initial', role=Role.BOTH, app_name='flimma')
class B1(LocalMean):
    def register(self):
        self.register_transition('Gene filter', Role.COORDINATOR, 'Gather local gene names')
        self.register_transition('Local CPM Cutoff', Role.PARTICIPANT, 'Wait for shared genes')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Gene filter'
        return 'Local CPM Cutoff'


@app_state('Gene filter', Role.COORDINATOR)
class C1(GlobalMean):
    def register(self):
        self.register_transition('Local CPM Cutoff', Role.COORDINATOR, 'Broadcast shared genes')

    def run(self) -> str or None:
        super().run()
        return 'Local CPM Cutoff'


@app_state('Local CPM Cutoff', Role.BOTH)
class B2(CPMCutOff):
    def register(self):
        self.register_transition('Median cutoff aggregation', Role.COORDINATOR, 'Gather local cutoffs')
        self.register_transition('Apply CPM Cutoff', Role.PARTICIPANT, 'Wait for global CPM cutoff')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Median cutoff aggregation'
        return 'Apply CPM Cutoff'


@app_state('Median cutoff aggregation', Role.COORDINATOR)
class C2(CutOffAggregation):
    def register(self):
        self.register_transition('Apply CPM Cutoff', Role.COORDINATOR, 'Broadcast global CPM cutoff')

    def run(self) -> str or None:
        super().run()
        return 'Apply CPM Cutoff'


@app_state('Apply CPM Cutoff', Role.BOTH)
class B3(ApplyCPM):
    def register(self):
        self.register_transition('Aggregate Gene Names', Role.COORDINATOR, 'Gather genes above the cutoff')
        self.register_transition('Compute Norm Factors', Role.PARTICIPANT, 'Wait for shared genes above the cutoff')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Aggregate Gene Names'
        return 'Compute Norm Factors'


@app_state('Aggregate Gene Names', Role.COORDINATOR)
class C3(AggregateGeneNames):
    def register(self):
        self.register_transition('Compute Norm Factors', Role.COORDINATOR, 'Broadcast shared genes above the cutoff')

    def run(self) -> str or None:
        super().run()
        return 'Compute Norm Factors'


@app_state('Compute Norm Factors', Role.BOTH)
class B4(ComputeNormFactors):
    def register(self):
        self.register_transition('UQ norm factor aggregation', Role.COORDINATOR, 'Gather upper local norm factors')
        self.register_transition('Linear Regression', Role.PARTICIPANT, 'Wait for global UQ factor')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'UQ norm factor aggregation'
        return 'Linear Regression'


@app_state('UQ norm factor aggregation', Role.COORDINATOR)
class C4(AggregateLibSizes):
    def register(self):
        self.register_transition('Linear Regression', Role.COORDINATOR, 'Broadcast global UQ factor')

    def run(self) -> str or None:
        super().run()
        return 'Linear Regression'


@app_state('Linear Regression', Role.BOTH)
class B5(LinearRegression):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Aggregate Regression Parameters', Role.COORDINATOR, "Gather intercepts and slopes")
        self.register_transition('SSE', Role.PARTICIPANT, "Wait for Beta")

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Aggregate Regression Parameters'
        return 'SSE'


@app_state('Aggregate Regression Parameters', Role.COORDINATOR)
class C5(AggregateRegression):
    def register(self):
        self.register_transition('SSE', Role.COORDINATOR, "Broadcast Beta")

    def run(self) -> str or None:
        super().run()
        return 'SSE'


@app_state('SSE', Role.BOTH)
class B6(SSE):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Aggregate SSE', Role.COORDINATOR, "Gather local SSE params")
        self.register_transition('Linear Regression', Role.PARTICIPANT, "Wait for lowess")
        self.register_transition('Write Results', Role.PARTICIPANT, "Wait for global gene expression analysis")

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            self.weighted = not self.weighted
            return 'Aggregate SSE'
        if not self.weighted:
            self.weighted = not self.weighted
            return 'Linear Regression'
        return 'Write Results'


@app_state('Aggregate SSE', Role.COORDINATOR)
class C6(AggregateSSE):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Linear Regression', Role.COORDINATOR, "Broadcast lowess")
        self.register_transition('terminal', Role.COORDINATOR, "Broadcast Gene expression analysis")

    def run(self) -> str or None:
        super().run()
        if not self.weighted:
            return 'Linear Regression'
        return 'terminal'


@app_state('Write Results', Role.PARTICIPANT)
class P1(WriteResults):

    def register(self):
        self.register_transition('terminal', Role.PARTICIPANT, "Terminate app execution")

    def run(self) -> str or None:
        super().run()
        return 'terminal'
