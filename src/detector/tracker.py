# detector/tracker.py
from collections import deque
from collections import defaultdict

class VehicleState:
    OUTSIDE = "outside"
    ENTER = "enter"
    INSIDE = "inside"
    EXIT = "exit"

class ObjectFSM:
    def __init__(self):
        self.states = defaultdict(lambda: VehicleState.OUTSIDE)
        self.prev_inside = defaultdict(lambda: False)

    def update(self, obj_id, inside_now):
        inside_prev = self.prev_inside[obj_id]
        state_prev = self.states[obj_id]

        # ENTER: False -> True
        if not inside_prev and inside_now:
            self.states[obj_id] = VehicleState.ENTER

        # INSIDE: True -> True (after ENTER)
        elif inside_prev and inside_now:
            self.states[obj_id] = VehicleState.INSIDE

        # EXIT: True -> False
        elif inside_prev and not inside_now:
            self.states[obj_id] = VehicleState.EXIT
        # OUTSIDE: False -> False
        else:
            self.states[obj_id] = VehicleState.OUTSIDE

        # update memory
        self.prev_inside[obj_id] = inside_now

        return self.states[obj_id]
    
class TrackState:
    def __init__(self, history_len):
        self.positions = deque(maxlen=history_len)
        self.state = VehicleState.OUTSIDE

    def add(self, cx, cy):
        self.positions.append((cx, cy))

    def ready(self):
        return len(self.positions) == self.positions.maxlen