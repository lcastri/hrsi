from enum import Enum


class action_strategy(Enum):
    RANDOM = 0
    KEYBOARD = 1
    
class head_movements(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
class torso_movements(Enum):
    UP = 0
    DOWN = 1

NODE_NAME = 'action_manager'
NODE_RATE = 10 #0.33 #Hz

J_HEAD_1 = "head_1_joint"
J_HEAD_2 = "head_2_joint"
J_HEAD_1_LEFT = 1.2
J_HEAD_1_RIGHT = -1.2
J_HEAD_1_TIME = 1.5
J_HEAD_2_UP = 0.79
J_HEAD_2_DOWN = -1
J_HEAD_2_TIME = 1.5

J_TORSO = "torso_lift_joint"
J_TORSO_UP = 0.35
J_TORSO_DOWN = 0.025
J_TORSO_TIME = 1.5