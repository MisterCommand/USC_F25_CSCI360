import itertools
import numpy as np
import matplotlib.pyplot as plt
import time

# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
import numpy as np

class TextbookStack(object):
    """ A class that tracks the """
    def __init__(self, initial_order, initial_orientations):
        assert len(initial_order) == len(initial_orientations)
        self.num_books = len(initial_order)
        
        for i, a in enumerate(initial_orientations):
            assert i in initial_order
            assert a == 1 or a == 0

        self.order = np.array(initial_order)
        self.orientations = np.array(initial_orientations)

    def flip_stack(self, position):
        assert position <= self.num_books
        
        self.order[:position] = self.order[:position][::-1]
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[::-1]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx != self.order[idx]) or (front_matter != 1):
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)
    
    def __eq__(self, other):
        assert isinstance(other, TextbookStack), "equality comparison can only ba made with other __TextbookStacks__"
        return all(self.order == other.order) and all(self.orientations == other.orientations)

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack

def breadth_first_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #

    queue = deque([(stack, [])]) # queue to store the state of the stack to be explored and the sequence of flips
    visited = {(tuple(stack.order), tuple(stack.orientations))} # set to store visited states to avoid loops
    count = 0

    while queue:

        current_stack = queue.popleft() # FIFO
        count += 1

        # If the stack deque is ordered, return the flip sequence and node count
        if current_stack[0].check_ordered():
            return current_stack[1], count

        # for each flip in the sequence
        for flip in range(1, current_stack[0].num_books + 1):
            new_stack = current_stack[0].copy()
            new_stack.flip_stack(flip)
            new_sequence = current_stack[1] + [flip]
            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                queue.append((new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))

    return current_stack[1], count # <- return the flip sequence and the number of nodes traversed
    # ---------------------------- #


def depth_first_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #

    queue = deque([(stack, [])]) # queue to store the state of the stack to be explored and the sequence of flips
    visited = {(tuple(stack.order), tuple(stack.orientations))} # set to store visited states to avoid loops
    count = 0

    while queue:
        current_stack = queue.pop() # LIFO
        count += 1

        if current_stack[0].check_ordered():
            return current_stack[1], count

        # for each flip in the sequence
        for flip in range(1, current_stack[0].num_books + 1):
            new_stack = current_stack[0].copy()
            new_stack.flip_stack(flip)
            new_sequence = current_stack[1] + [flip]

            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                queue.append((new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))

    return (current_stack[1], count) # <- return the flip sequence and the number of nodes traversed
    # ---------------------------- #


# Generate all possible states of the stack given n
def generate_all_states(n):
    order = np.arange(n) # 0 to n - 1

    # All possible orders and orientations
    all_orders = list(itertools.permutations(order))
    all_orientations = list(itertools.product([0, 1], repeat=n))

    # All possible states
    all_states = []
    for order in all_orders:
        for orientations in all_orientations:
            all_states.append(TextbookStack(order, orientations))

    # Check if every state is unique
    in_set = set({})
    for state in all_states:
        in_set.add((tuple(state.order), tuple(state.orientations)))

    return all_states

# Count the average number of flips to order the stack and the average number of nodes traversed by the search algorithm
def count_average_flips_and_nodes(algo, n):
    all_states = generate_all_states(n)
    print("Number of possible states: ", len(all_states))

    # Iterate through all states
    flips_list = []
    nodes_list = []
    for state in all_states:
        if algo == "bfs":
            flips, nodes = breadth_first_search(state)
            print(flips, nodes)
            # print("Progress: ", len(flips_list) / len(all_states))
            flips_list.append(len(flips))
            nodes_list.append(nodes)

        elif algo == "dfs":
            flips, nodes = depth_first_search(state)
            print(flips, nodes)
            # print("Progress: ", len(flips_list) / len(all_states))
            flips_list.append(len(flips))
            nodes_list.append(nodes)


    return {
        "flips": np.mean(flips_list),
        "nodes": np.mean(nodes_list)
    }

# ------------------------------------------------------------ #
# The following code is generated by AI to plot the results
# ------------------------------------------------------------ #
def run_experiments_and_plot():
    """Run BFS and DFS experiments for n=1 to 5 and plot the results"""
    n_values = range(1, 6)  # 1 to 5
    
    bfs_mean_flips = []
    bfs_mean_nodes = []
    dfs_mean_flips = []
    dfs_mean_nodes = []
    
    print("Running experiments for n = 1 to 5...")
    
    for n in n_values:
        print(f"\n=== Running experiments for n = {n} ===")
        
        # Run BFS experiment
        print(f"Running BFS for n = {n}...")
        start_time = time.time()
        bfs_results = count_average_flips_and_nodes("bfs", n)
        bfs_time = time.time() - start_time
        
        bfs_mean_flips.append(bfs_results["flips"])
        bfs_mean_nodes.append(bfs_results["nodes"])
        print(f"BFS completed in {bfs_time:.2f} seconds")
        print(f"BFS Results - Mean flips: {bfs_results['flips']:.2f}, Mean nodes: {bfs_results['nodes']:.2f}")
        
        # Run DFS experiment (with timeout for larger n)
        print(f"Running DFS for n = {n}...")
        start_time = time.time()
        
        dfs_results = count_average_flips_and_nodes("dfs", n)
        dfs_mean_flips.append(dfs_results["flips"])
        dfs_mean_nodes.append(dfs_results["nodes"])
        dfs_time = time.time() - start_time
        print(f"DFS completed in {dfs_time:.2f} seconds")
        print(f"DFS Results - Mean flips: {dfs_results['flips']:.2f}, Mean nodes: {dfs_results['nodes']:.2f}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean number of flips
    ax1.plot(n_values, bfs_mean_flips, 'bo-', label='BFS', linewidth=2, markersize=8)
    valid_dfs_n = [n for n, flips in zip(n_values, dfs_mean_flips) if not np.isnan(flips)]
    valid_dfs_flips = [flips for flips in dfs_mean_flips if not np.isnan(flips)]
    ax1.plot(valid_dfs_n, valid_dfs_flips, 'ro-', label='DFS', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Books (n)')
    ax1.set_ylabel('Mean Number of Flips')
    ax1.set_title('Mean Number of Flips vs Number of Books')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)
    
    # Plot 2: Mean number of nodes explored
    ax2.plot(n_values, bfs_mean_nodes, 'bo-', label='BFS', linewidth=2, markersize=8)
    valid_dfs_nodes = [nodes for nodes in dfs_mean_nodes if not np.isnan(nodes)]
    ax2.plot(valid_dfs_n, valid_dfs_nodes, 'ro-', label='DFS', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Books (n)')
    ax2.set_ylabel('Mean Number of Nodes Explored')
    ax2.set_title('Mean Number of Nodes Explored vs Number of Books')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_values)
    
    plt.tight_layout()
    plt.savefig('bfs_vs_dfs_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'n':<3} | {'BFS Flips':<10} | {'BFS Nodes':<10} | {'DFS Flips':<10} | {'DFS Nodes':<10}")
    print("-" * 55)
    for i, n in enumerate(n_values):
        bfs_f = f"{bfs_mean_flips[i]:.2f}" if not np.isnan(bfs_mean_flips[i]) else "N/A"
        bfs_n = f"{bfs_mean_nodes[i]:.2f}" if not np.isnan(bfs_mean_nodes[i]) else "N/A"
        dfs_f = f"{dfs_mean_flips[i]:.2f}" if not np.isnan(dfs_mean_flips[i]) else "N/A"
        dfs_n = f"{dfs_mean_nodes[i]:.2f}" if not np.isnan(dfs_mean_nodes[i]) else "N/A"
        print(f"{n:<3} | {bfs_f:<10} | {bfs_n:<10} | {dfs_f:<10} | {dfs_n:<10}")

    
run_experiments_and_plot()