import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mazelab import Motion


def xy_to_flatten_idx(array, x, y):
    M, N = array.shape
    return x*N + y


def flatten_idx_to_xy(array, idx):
    M, N = array.shape
    x = idx//N
    y = idx%N
    return np.array([x, y])


def make_graph(impassible_array, motions):
    M, N = impassible_array.shape
    free_idx = list(zip(*np.where(np.logical_not(impassible_array))))
    row = []
    col = []
    for idx in free_idx:
        node_idx = xy_to_flatten_idx(impassible_array, idx[0], idx[1])
        for motion in motions:
            next_idx = [idx[0] + motion[0], idx[1] + motion[1]]
            if (next_idx[0] >= 0 and next_idx[0] < M and next_idx[1] >= 0 and next_idx[1] < N) and not impassible_array[next_idx[0], next_idx[1]]:
                row.append(node_idx)
                col.append(xy_to_flatten_idx(impassible_array, next_idx[0], next_idx[1]))
    data = [1]*len(row)
    graph = csr_matrix((data, (row, col)), shape=(M*N, M*N))
    
    return graph


def get_actions(impassible_array, motions, predecessors, start_idx, goal_idx):
    start_idx = xy_to_flatten_idx(impassible_array, *start_idx)
    goal_idx = xy_to_flatten_idx(impassible_array, *goal_idx)
    actions = []
    while goal_idx != start_idx:
        if predecessors[goal_idx] == -9999:
            return None
        action = flatten_idx_to_xy(impassible_array, goal_idx) - flatten_idx_to_xy(impassible_array, predecessors[goal_idx])
        for i, motion in enumerate(motions):
            if np.allclose(action, motion):
                action_idx = i
        actions.append(action_idx)
        goal_idx = predecessors[goal_idx]
    return actions[::-1]


def dijkstra_solver(impassible_array, motion, start_idx, goal_idx):
    impassible_array = np.asarray(impassible_array)
    assert impassible_array.dtype == np.bool
    assert isinstance(motion, Motion)
    motions = [list(m.values())[0] for m in motion.motions]
    
    graph = make_graph(impassible_array, motions)
    dist_matrix, predecessors = dijkstra(csgraph=graph, indices=xy_to_flatten_idx(impassible_array, *start_idx), return_predecessors=True)
    actions = get_actions(impassible_array, motions, predecessors, start_idx, goal_idx)
    return actions
