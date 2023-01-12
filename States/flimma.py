"""
    FeatureCloud Flimma Application
    Copyright 2023 Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
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
        self.register_transition('Gene filter', Role.COORDINATOR, label='Gather local gene names')
        self.register_transition('Local CPM Cutoff', Role.PARTICIPANT, label='Wait for shared genes')

    def run(self) -> str or None:
        self.store("weighted", False)
        super().run()
        if self.is_coordinator:
            return 'Gene filter'
        return 'Local CPM Cutoff'


@app_state('Gene filter', Role.COORDINATOR)
class C1(GlobalMean):
    def register(self):
        self.register_transition('Local CPM Cutoff', Role.COORDINATOR, label='Broadcast shared genes')

    def run(self) -> str or None:
        super().run()
        return 'Local CPM Cutoff'


@app_state('Local CPM Cutoff', Role.BOTH)
class B2(CPMCutOff):
    def register(self):
        self.register_transition('Median cutoff aggregation', Role.COORDINATOR, label='Gather local cutoffs')
        self.register_transition('Apply CPM Cutoff', Role.PARTICIPANT, label='Wait for global CPM cutoff')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Median cutoff aggregation'
        return 'Apply CPM Cutoff'


@app_state('Median cutoff aggregation', Role.COORDINATOR)
class C2(CutOffAggregation):
    def register(self):
        self.register_transition('Apply CPM Cutoff', Role.COORDINATOR, label='Broadcast global CPM cutoff')

    def run(self) -> str or None:
        super().run()
        return 'Apply CPM Cutoff'


@app_state('Apply CPM Cutoff', Role.BOTH)
class B3(ApplyCPM):
    def register(self):
        self.register_transition('Aggregate Gene Names', Role.COORDINATOR, label='Gather genes above the cutoff')
        self.register_transition('Compute Norm Factors', Role.PARTICIPANT,
                                 label='Wait for shared genes above the cutoff')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'Aggregate Gene Names'
        return 'Compute Norm Factors'


@app_state('Aggregate Gene Names', Role.COORDINATOR)
class C3(AggregateGeneNames):
    def register(self):
        self.register_transition('Compute Norm Factors', Role.COORDINATOR,
                                 label='Broadcast shared genes above the cutoff')

    def run(self) -> str or None:
        super().run()
        return 'Compute Norm Factors'


@app_state('Compute Norm Factors', Role.BOTH)
class B4(ComputeNormFactors):
    def register(self):
        self.register_transition('UQ norm factor aggregation', Role.COORDINATOR,
                                 label='Gather upper local norm factors')
        self.register_transition('Linear Regression', Role.PARTICIPANT, label='Wait for global UQ factor')

    def run(self) -> str or None:
        super().run()
        if self.is_coordinator:
            return 'UQ norm factor aggregation'
        return 'Linear Regression'


@app_state('UQ norm factor aggregation', Role.COORDINATOR)
class C4(AggregateLibSizes):
    def register(self):
        self.register_transition('Linear Regression', Role.COORDINATOR, label='Broadcast global UQ factor')

    def run(self) -> str or None:
        super().run()
        return 'Linear Regression'


@app_state('Linear Regression', Role.BOTH)
class B5(LinearRegression):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Aggregate Regression Parameters', Role.COORDINATOR,
                                 label="Gather intercepts and slopes")
        self.register_transition('SSE', Role.PARTICIPANT, label="Wait for Beta")

    def run(self) -> str or None:
        self.weighted = self.load("weighted")
        super().run()
        if self.is_coordinator:
            return 'Aggregate Regression Parameters'
        return 'SSE'


@app_state('Aggregate Regression Parameters', Role.COORDINATOR)
class C5(AggregateRegression):
    def register(self):
        self.register_transition('SSE', Role.COORDINATOR, label="Broadcast Beta")

    def run(self) -> str or None:
        super().run()
        return 'SSE'


@app_state('SSE', Role.BOTH)
class B6(SSE):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Aggregate SSE', Role.COORDINATOR, label="Gather local SSE params")
        self.register_transition('Linear Regression', Role.PARTICIPANT, label="Wait for lowess")
        self.register_transition('Write Results', Role.PARTICIPANT, label="Wait for global gene expression analysis")

    def run(self) -> str or None:
        self.weighted = self.load("weighted")
        super().run()
        # self.weighted = self.load("weighted")
        self.weighted = not self.weighted
        self.store("weighted", self.weighted)
        if self.is_coordinator:
            return 'Aggregate SSE'
        if self.weighted:
            return 'Linear Regression'
        return 'Write Results'


@app_state('Aggregate SSE', Role.COORDINATOR)
class C6(AggregateSSE):
    def __init__(self):
        super().__init__()

    def register(self):
        self.register_transition('Linear Regression', Role.COORDINATOR, label="Broadcast lowess")
        self.register_transition('Write Results', Role.COORDINATOR, label="Broadcast Gene expression analysis")

    def run(self) -> str or None:
        self.weighted = self.load("weighted")
        super().run()
        self.store("weighted", self.weighted)
        if self.weighted:
            return 'Linear Regression'
        return 'Write Results'


@app_state('Write Results', Role.BOTH)
class P1(WriteResults):

    def register(self):
        self.register_transition('terminal', Role.BOTH, label="Terminate app execution")

    def run(self) -> str or None:
        super().run()
        return 'terminal'
