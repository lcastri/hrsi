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
        """
    for k in KEY_MAP: msg = msg + "\n\t" + k + " - " + KEY_MAP[k]
        
    msg = msg + """
        ---------------------------

        Press q to quit.
        """
    print(msg)