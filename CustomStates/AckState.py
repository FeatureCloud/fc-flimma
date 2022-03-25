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
    def send_data_to_coordinator(self, data, send_to_self=True, use_smpc=False, get_ack=False):
        super(AckState, self).send_data_to_coordinator(data, send_to_self, use_smpc)
        if get_ack and not self.is_coordinator:
            ack = self.await_data()
            if not ack == ACKNOWLEDGE:
                self.log(f"Wrong Acknowledge code: {ack}", LogLevel.FATAL)
                self.update(state=State.ERROR)

    def gather_data(self, is_json=False, ack=False):
        data = super(AckState, self).gather_data()
        if ack:
            self.broadcast_data(data=ACKNOWLEDGE, send_to_self=False)
        return data

    def aggregate_data(self, operation: SMPCOperation, use_smpc=False, ack=False):
        data = super(AckState, self).aggregate_data(operation, use_smpc)
        if ack:
            self.broadcast_data(data=ACKNOWLEDGE, send_to_self=False)
        return data