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
NODE_RATE = 10 #Hz

J_HEAD_1 = "head_1_joint"
J_HEAD_2 = "head_2_joint"
J_HEAD_1_TARGETLEFT = 1.2
J_HEAD_1_TARGETRIGHT = -1.2
J_HEAD_1_TARGETTIME = 1.5
J_HEAD_2_TARGETUP = 0.7
J_HEAD_2_TARGETDOWN = -1
J_HEAD_2_TARGETTIME = 1.5

J_TORSO = "torso_lift_joint"
J_TORSO_TARGETUP = 0.35
J_TORSO_TARGETDOWN = 0.025
J_TORSO_TARGETTIME = 1.5