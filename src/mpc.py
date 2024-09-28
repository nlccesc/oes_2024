# src/mpc.py

class MPC:
    def __init__(self, model, state_machine, horizon=5):
        """
        Initialize the MPC controller.
        
        :param model: The trained machine learning model to predict next states.
        :param state_machine: The state machine to validate state transitions.
        :param horizon: The prediction horizon (number of steps to predict ahead).
        """
        self.model = model
        self.state_machine = state_machine
        self.horizon = horizon

    def predict_next_state(self, current_state):
        """
        Predict the next state using the machine learning model.
        
        :param current_state: The current state input for the model.
        :return: Predicted next state.
        """
        return self.model.predict([current_state])[0]  # Assuming model returns a list/array

    def generate_predictions(self, initial_state):
        """
        Generate a sequence of future states based on the MPC horizon.
        
        :param initial_state: The initial state from which to start predictions.
        :return: List of predicted states.
        """
        predictions = []
        current_state = initial_state

        for _ in range(self.horizon):
            next_state = self.predict_next_state(current_state)
            predictions.append(next_state)
            current_state = next_state

        return predictions

    def validate_predictions(self, predictions):
        """
        Validate the sequence of MPC predictions using the state machine.
        
        :param predictions: List of predicted states to validate.
        :return: Validated list of states according to state machine rules.
        """
        return self.state_machine.validate_mpc_predictions(predictions)
