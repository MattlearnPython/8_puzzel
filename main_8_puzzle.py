import queue as Q
import time
import resource
import sys
import math
from QueueAndStack import Queue, Stack

import copy
'''
global path_to_goal, cost_of_path, nodes_expanded, search_depth
global max_search_depth, running_time, max_ram_usage
'''

## The Class that Represents the Puzzle
class PuzzleState(object):
    """docstring for PuzzleState
    1.config: a tuple that contains n^2 numbers and represents a state
    2.n: n-puzzle
    """
    
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []
        
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = int(i / self.n)
                self.blank_col = int(i % self.n)
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
                
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
                
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
                
            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
                
        return self.children
    
    def calculate_cost(self, num_steps):
        """calculate cost of the state"""
        self.cost = calculate_total_heuristic(self.config, self.n) + num_steps
    
    def set_cost(self, cost):
        self.cost = cost
# Function that Writes to output.txt    
def determine_path_and_depth(state):
    path_to_goal = []
    search_depth = 0
    while True:
        if state.action == 'Initial':
            break
        else:
            search_depth += 1
            path_to_goal.append(state.action)
            state = state.parent 
    path_to_goal = path_to_goal[::-1] 
    
    return path_to_goal, search_depth    

def writeOutput(state, explored, frontier, running_time):
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # Calculate the output   
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----           
    path_to_goal, search_depth = determine_path_and_depth(state)
    cost_of_path = len(path_to_goal)
    nodes_expanded = len(explored) - 1 #Initial state should not be counted in.                    
    ram_usage_deno = 1024 ** 2
    max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / ram_usage_deno
    
    '''
    # Update maximum search depth;    
    max_search_depth = 0
    for state_unexplored in frontier.queue:
        tmp, search_depth_cur = determine_path_and_depth(state_unexplored)
        if search_depth_cur >= max_search_depth:
            max_search_depth = search_depth_cur
    for state_explored in explored:
        tmp, search_depth_cur = determine_path_and_depth(state_explored)
        if search_depth_cur >= max_search_depth:
            max_search_depth = search_depth_cur
    '''
      
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # Print the output   
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    print("Success! Check the output.txt!")        
    file = open('output.txt', 'w')      
    file.write( 'path_to_goal: ' + repr(path_to_goal) + '\n' )
    file.write( 'cost_of_path: ' + repr(cost_of_path) + '\n' )
    file.write( 'nodes_expanded : ' + repr(nodes_expanded) + '\n' )
    file.write( 'search_depth: ' + repr(search_depth) + '\n' )
    #file.write( 'max_search_depth: ' + repr(max_search_depth) + '\n' )
    file.write( 'running_time: ' + repr(running_time) + '\n' )
    file.write( 'max_ram_usage: ' + repr(max_ram_usage) )  
    file.close()   
    #print("Failure! Cannot find a feasible path.")  

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 1. Breadth first search
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
def bfs_search(initial_state): 
    start_time = time.time()
    """BFS search"""    
    frontier = Q.Queue()
    frontier.put(initial_state)
    frontier_config = set()
    frontier_config.add(initial_state.config)
    explored = set()
    
    while not frontier.empty():
        state = frontier.get()
        frontier_config.remove(state.config)
        explored.add(state.config)      
        if test_goal(state):
            running_time = time.time() - start_time
            writeOutput(state, explored, frontier, running_time)
            break
        else:
            state.expand()               
            for neighbor in state.children:
                if neighbor.config not in frontier_config and neighbor.config not in explored:
                        frontier.put(neighbor)
                        frontier_config.add(neighbor.config)
    return None

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 2. Depth first search
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====  
def dfs_search(initial_state): 
    start_time = time.time()
    """DFS search"""    
    frontier = Q.LifoQueue()
    frontier.put(initial_state)
    frontier_config = set()
    frontier_config.add(initial_state.config)
    explored = set()
    
    while not frontier.empty():
        state = frontier.get()
        frontier_config.remove(state.config)
        explored.add(state.config)
        #print('# expanded nodes:' + repr(len(explored)) + '\n')        
        if test_goal(state):
            running_time = time.time() - start_time
            writeOutput(state, explored, frontier, running_time)
            break
        else:
            state.expand()            
  
            for neighbor in state.children[::-1]:
                if neighbor.config not in frontier_config and neighbor.config not in explored:
                        frontier.put(neighbor)
                        frontier_config.add(neighbor.config)
    return None

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# 3. A star search
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====  
def A_star_search(initial_state): 
    start_time = time.time() 
    """ A* search """  
    num_step = 0 #Cost to reach new node 
    frontier = []
    initial_state.calculate_cost(num_step)
    frontier.append(initial_state)
    
    frontier_config = set()
    frontier_config.add(initial_state.config)
    explored = set()
    
    while not len(frontier) == 0:
    #for i in range(1):
        state = frontier.pop()
        frontier_config.remove(state.config)
        explored.add(state.config)
        
        if test_goal(state):
            running_time = time.time() - start_time
            writeOutput(state, explored, frontier, running_time)
            break
        else:
            state.expand()
            num_step += 1 
            for neighbor in state.children:
                if neighbor.config not in frontier_config and neighbor.config not in explored:
                        neighbor.calculate_cost(num_step)
                        frontier.append(neighbor)
                        frontier_config.add(neighbor.config)
                elif neighbor.config in frontier_config:
                        neighbor.calculate_cost(num_step)
                        for item in frontier:
                            if item.config == neighbor.config:
                                if neighbor.cost < item.cost:
                                    item.set_cost(neighbor.cost)
                                
        frontier.sort(key = lambda state: state.cost, reverse = True)
        
    return None

def calculate_total_heuristic(config, n):
    """calculate the total estimated cost of a state"""
    total_heuristic = 0
    for idx in range(n ** 2):
        value = config[idx]
        dist = calculate_manhattan_dist(idx, value, n)
        total_heuristic += dist
    
    return total_heuristic

def calculate_manhattan_dist(idx, value, n):
    """calculatet the manhattan distance of a tile"""
    idx_row = idx // n
    idx_col = idx % n
    value_pos_row = value // n
    value_pos_col = value % n
    manhattan_dist = abs(value_pos_row - idx_row) + abs(value_pos_col - idx_col)
    
    return manhattan_dist
    
def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    goal_state_config = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return goal_state_config == list(puzzle_state.config)

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")

if __name__ == '__main__': 
    sm = "ast"
    begin_state = [1, 2, 5, 3, 4, 0, 6, 7, 8]
    #begin_state = [1, 2, 0, 3, 4, 5, 6, 7, 8]
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size) 
    
    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")
        
  
    
    
        