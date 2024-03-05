import numpy as np


def get_direction(start_node, end_node):
    direction = end_node.node - start_node.node
    distance = np.linalg.norm(direction)
    return direction / (distance + 1e-8), distance