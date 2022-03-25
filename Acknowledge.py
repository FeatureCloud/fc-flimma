"""
    FeatureCloud Flimma Application
    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.
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
from time import sleep
ACKNOWLEDGE = "FC_ACK"


def get_acknowledge(state):
    pass
    # if state.is_coordinator:
    #     return True
    # ack = state.await_data(n=1, unwrap=True, is_json=False)[0]
    # if ack == ACKNOWLEDGE:
    #     print("************GOT Acknowledged***********")
    #     return True
    # print(f"The acknowledge message {ack} is not valid!"
    #       f"The correct massage is {ACKNOWLEDGE}")
    # return False


def acknowledge(state, client_id):
    pass
    # if client_id != state.id:
    #     state.send_data_to_participant(data=[ACKNOWLEDGE], destination=client_id, flush=True)
    #     print("************Acknowledged***********")
    # flush(state)

def flush(state):
    while state._app.status_available():
        sleep(1)

# def fetch(state):
#     while state._app.status_available():
#         sleep(5)