########################################################################################
#  Modified from https://github.com/aimacode/aima-python/blob/master/search.ipynb
########################################################################################


import heapq

import numpy as np


class Node(object):
    """Each Node in a search tree consists of:
            1. Current state
            2. A pointer to previous node
            3. The action leads to current state from previous one
            4. Path cost from the root to current state. 
    """
    
    def __init__(self, state, prev_node=None, action=None, step_cost=1):
        """Create a Node given current state, previous node, action and one-step cost"""
        self.state = state
        self.prev_node = prev_node
        self.action = action
        if self.prev_node is None:  # at root
            self.path_cost = 0
        else:
            self.path_cost = self.prev_node.path_cost + step_cost  # accumulate cost
            
    def __repr__(self):
        """String representation of the node."""
        return 'State: {}, Path cost: {}'.format(self.state, self.path_cost)
    
    def __lt__(self, node):
        """Operator of less than"""
        return self.path_cost < node.path_cost
    
    def next_node(self, maze, action):
        """Returns a Node with next state by taking an action from current state"""
        next_state = env._next_state(self.state, action)
        step_cost = maze.step_cost(self.state, action, next_state)
        
        return Node(next_state, self, action, step_cost)


class Frontier(object):
    """Frontier (a Priority Queue) ordered by a cost function. """
    
    def __init__(self, initial_node, f):
        """Initialize a Frontier with intial Node with a cost function"""
        self.heap = []
        self.state_nodes = {}  # Track all state-node pairs in Frontier 
        self.f = f
        
        self.add(initial_node)
        
    def add(self, node):
        """Add node to the Frontier"""
        heapq.heappush(self.heap, [self.f(node), node])
        self.state_nodes[tuple(node.state)] = node  # Track state-node
        
    def pop(self):
        """Remove and return the Node with minimum f value."""
        f_value, node = heapq.heappop(self.heap)
        self.state_nodes.pop(tuple(node.state))  # Remove state-node
        
        return node
    
    def replace(self, node):
        """Replace a previous node by this node with the same state."""
        for i, (f_value, old_node) in enumerate(self.heap):
            if node.state == old_node.state:
                self.heap[i] = [self.f(node), node]
                heapq._siftdown(self.heap, 0, i)
                
                self.state_nodes[tuple(node.state)] = node  # Update state-node

    def __contains__(self, state):
        return state in self.state_nodes
    
    def __getitem__(self, state):
        """Return the node with state"""
        return self.state_nodes[tuple(state)]

    def __len__(self):
        return len(self.heap)
    
    
class AstarSolver(object):
    """A* solver for the maze"""
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        
        # Solve it
        self.solution_node = self._astar_search(self._heuristic)
    
    def solvable(self):
        """Return True if there exists a valid solution by the definition of this maze."""
        return self.solution_node is not None
    
    def get_actions(self):
        """Return the solution of sequence of actions"""
        node = self.solution_node
        
        actions = []
        while node.prev_node:
            actions.append(node.action)
            node = node.prev_node
        return actions[::-1]
            
    def get_states(self):
        """Return the solution of trajectory of states"""
        node = self.solution_node
        
        states = [node.state]
        while node.prev_node:
            node = node.prev_node
            states.append(node.state)
        return states[::-1]
    
    def _astar_search(self, h):
        """A* search
        
        Args:
            h: A heuristic function
        """
        f = lambda node: node.path_cost + h(node.state)
        frontier = Frontier(Node(self.env.state), f)
        explored = set()

        while frontier:
            node = frontier.pop()
            if tuple(node.state) == tuple(self.goal):  # goal test
                return node
            explored.add(tuple(node.state))

            for action in self.env.all_actions:
                child_state = self.env._next_state(node.state, action)
                child = Node(child_state, 
                             prev_node=node, 
                             action=action, 
                             step_cost=self._step_cost(node.state, action, child_state))
                chlid_state = tuple(child.state)
                if chlid_state not in explored and chlid_state not in frontier:
                    frontier.add(child)
                elif chlid_state in frontier and frontier[chlid_state] < child:
                    frontier.replace(child)

        return None  # No solution found

    def _heuristic(self, state):
        """Heuristic function for maze: Euclidean distance to the goal"""
        return np.linalg.norm(np.array(state) - np.array(self.goal))
    
    def _step_cost(self, state, action, next_state):
        """Return a cost that a given action leads a state to a next_state"""
        return 1  # Simple maze: uniform cost for each step in the path.