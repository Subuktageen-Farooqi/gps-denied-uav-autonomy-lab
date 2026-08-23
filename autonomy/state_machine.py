from enum import Enum


class State(str, Enum):
    SEARCH="SEARCH"; ACQUIRE="ACQUIRE"; TRACK="TRACK"; PREDICT="PREDICT"; REACQUIRE="REACQUIRE"


class AutonomyFSM:
    def __init__(self, prediction_timeout=3.0):
        self.state, self.time_in_state = State.SEARCH, 0.0
        self.hits, self.prediction_timeout = 0, prediction_timeout

    def _set(self, state):
        if state != self.state: self.state, self.time_in_state, self.hits = state, 0.0, 0

    def update(self, detected, dt):
        self.time_in_state += dt
        if self.state == State.SEARCH and detected: self._set(State.ACQUIRE); self.hits=1
        elif self.state == State.ACQUIRE:
            self.hits = self.hits+1 if detected else 0
            if self.hits >= 3: self._set(State.TRACK)
            elif self.time_in_state > 2 and not detected: self._set(State.SEARCH)
        elif self.state == State.TRACK and not detected: self._set(State.PREDICT)
        elif self.state == State.PREDICT:
            if detected: self._set(State.REACQUIRE); self.hits=1
            elif self.time_in_state > self.prediction_timeout: self._set(State.SEARCH)
        elif self.state == State.REACQUIRE:
            self.hits = self.hits+1 if detected else 0
            if self.hits >= 2: self._set(State.TRACK)
            elif not detected: self._set(State.PREDICT)
        return self.state

