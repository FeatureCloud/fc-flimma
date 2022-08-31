"""
    FeatureCloud Flimma Application
    Copyright 2022 Mohammad Bakhtiari. All Rights Reserved.
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
from FeatureCloud.app.engine.app import AppState, LogLevel, State, SMPCOperation

ACKNOWLEDGE = "FC_ACK"


class AckState(AppState):
    def _send_data_to_coordinator(self, data, send_to_self=True, use_smpc=False, get_ack=False):
        super(AckState, self).send_data_to_coordinator(data, send_to_self, use_smpc)
        if get_ack and not self.is_coordinator:
            ack = self.await_data()
            if not ack == ACKNOWLEDGE:
                self.log(f"Wrong Acknowledge code: {ack}", LogLevel.FATAL)
                self.update(state=State.ERROR)

    def _gather_data(self, is_json: bool = False, ack: bool = False):
        data = super(AckState, self).gather_data()
        if ack:
            self.broadcast_data(data=ACKNOWLEDGE, send_to_self=False)
        return data

    def _aggregate_data(self, operation: SMPCOperation = SMPCOperation.ADD, use_smpc: bool = False, ack: bool = False):
        data = super(AckState, self).aggregate_data(operation, use_smpc)
        if ack:
            self.broadcast_data(data=ACKNOWLEDGE, send_to_self=False)
        return data

    def send_multiple_data(self, data: list, params: list):
        for d, p in zip(data, params):
            self._send_data_to_coordinator(data=d, **p, get_ack=True)

    def instant_aggregate(self, name: str, data: list, use_smpc: bool = False, operation: SMPCOperation = SMPCOperation.ADD):
        self._send_data_to_coordinator(data=data, use_smpc=use_smpc, get_ack=True)
        if self.is_coordinator:
            aggregated_data = self._aggregate_data(operation=operation, use_smpc=use_smpc, ack=True)
            self.store(name, aggregated_data)

    def instant_gather(self, name: str, data: list, is_json: bool = False):
        self._send_data_to_coordinator(data=data, use_smpc=False, get_ack=True)
        if self.is_coordinator:
            gathered_data = self._gather_data(is_json=is_json, ack=True)
            self.store(name, gathered_data)
