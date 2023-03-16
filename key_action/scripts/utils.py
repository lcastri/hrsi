from constants import *


def print_node_info():
    """
    Print node's name, rate and other info
    """
    print("Node name : " + NODE_NAME)
    print("Node rate : " + str(NODE_RATE) + "Hz")
    
    
def print_instruction():
    msg = """
        Reading from the keyboard and Publishing to String!
        ---------------------------
        Pre-defined actions:
            a - Head left and right 
            b - Head up and down
            c - Torso up and down
            d - Increase the velocity by 10% 
            e - Decrease the velocity by 10%
        ---------------------------

        Press q to quit.
        """
    print(msg)