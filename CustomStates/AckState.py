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