# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
from heapq import heappush, heappop
import queue
import itertools

import numpy as np
import time


class TextbookStack(object):
    """A class that tracks the"""

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
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[
            ::-1
        ]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx != self.order[idx]) or (front_matter != 1):
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)

    def __eq__(self, other):
        assert isinstance(
            other, TextbookStack
        ), "equality comparison can only ba made with other __TextbookStacks__"
        return all(self.order == other.order) and all(
            self.orientations == other.orientations
        )

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack


def a_star_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #
    priority_queue = []
    visited = set() # set to store visited states to avoid loops

    def heuristic(s):
        # (a) If the pair of books are not adjacent in the ordered stack, regardless of being face up or face down.
        # (b) If the pair has a book facing up and one facing down.
        # (c) If the pair is wrongly ordered, but with correct orientations (both facing up)
        # (d) If the pair is correctly ordered, but with wrong orientations (both facing down)
        count = 0

        for i in range(len(s.order) - 1):
            if abs(s.order[i] - s.order[i + 1]) != 1:
                # (a)
                count += 1
            elif s.orientations[i] != s.orientations[i + 1]:
                # (b)
                count += 1
            elif s.order[i] > s.order[i + 1] and s.orientations[i] == 1:
                # (c)
                count += 1
            elif s.order[i] < s.order[i + 1] and s.orientations[i] == 0:
                # (d)
                count += 1

        return count
    
    # Push the initial state onto the queue
    heappush(priority_queue, (0, 0, stack, [])) # (f, tie_breaker, stack, flip_sequence), f doesn't matter for the first element
    visited.add((tuple(stack.order), tuple(stack.orientations)))
    current_stack = (stack, []) # tuple of (TextbookStack, flip_sequence)
    tie_breaker = 1 # Compare this if same f value

    # Main loop
    while priority_queue:
        # Dequeue lowest f
        f_cost, _, current_stack, flip_sequence = heappop(priority_queue)

        # Goal check
        if current_stack.check_ordered():
            return flip_sequence

        # Expand neighbors
        for flip in range(1, current_stack.num_books + 1):
            new_stack = current_stack.copy()
            new_stack.flip_stack(flip)
            new_sequence = flip_sequence + [flip]

            # Calculate f
            f = len(new_sequence) + heuristic(new_stack)

            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                heappush(priority_queue, (f, tie_breaker, new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))
                tie_breaker += 1

    return [] # return empty sequence if no solution found
    # ---------------------------- #


def weighted_a_star_search(stack, epsilon=None, N=1):
    # Weighted A* is extra credit (also returns number of nodes visited)

    flip_sequence = []
    # --- v ADD YOUR CODE HERE v --- #

    priority_queue = []
    visited = set() # set to store visited states to avoid loops

    def heuristic(s):
        # (a) If the pair of books are not adjacent in the ordered stack, regardless of being face up or face down.
        # (b) If the pair has a book facing up and one facing down.
        # (c) If the pair is wrongly ordered, but with correct orientations (both facing up)
        # (d) If the pair is correctly ordered, but with wrong orientations (both facing down)
        count = 0

        for i in range(len(s.order) - 1):
            if abs(s.order[i] - s.order[i + 1]) != 1:
                # (a)
                count += 1
            elif s.orientations[i] != s.orientations[i + 1]:
                # (b)
                count += 1
            elif s.order[i] > s.order[i + 1] and s.orientations[i] == 1:
                # (c)
                count += 1
            elif s.order[i] < s.order[i + 1] and s.orientations[i] == 0:
                # (d)
                count += 1

        return count
    
    # Push the initial state onto the queue
    heappush(priority_queue, (0, 0, stack, [])) # (f, tie_breaker, stack, flip_sequence), f doesn't matter for the first element
    visited.add((tuple(stack.order), tuple(stack.orientations)))
    current_stack = (stack, []) # tuple of (TextbookStack, flip_sequence)
    tie_breaker = 1 # Compare this if same f value
    no_nodes_visited = 1 # Count of nodes visited

    # Main loop
    while priority_queue:
        # Dequeue lowest f
        f_cost, _, current_stack, flip_sequence = heappop(priority_queue)

        # Goal check
        if current_stack.check_ordered():
            return flip_sequence, no_nodes_visited

        # Expand neighbors
        for flip in range(1, current_stack.num_books + 1):
            new_stack = current_stack.copy()
            new_stack.flip_stack(flip)
            new_sequence = flip_sequence + [flip]
            no_nodes_visited += 1

            # Calculate f (WEIGHTED)
            w = 1 + epsilon - (epsilon * len(new_sequence) / N) if epsilon is not None else 1
            f = len(new_sequence) + w * heuristic(new_stack)

            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                heappush(priority_queue, (f, tie_breaker, new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))
                tie_breaker += 1

    return [], no_nodes_visited # return empty sequence if no solution found

    # ---------------------------- #


if __name__ == "__main__":
    test = TextbookStack(initial_order=[3, 2, 1, 0], initial_orientations=[0, 0, 0, 0])
    output_sequence = a_star_search(test)
    correct_sequence = int(output_sequence == [4])

    new_stack = apply_sequence(test, output_sequence)
    stack_ordered = new_stack.check_ordered()

    print(f"Stack is {'' if stack_ordered else 'not '}ordered")
    print(f"Comparing output to expected traces  - \t{'PASSED' if correct_sequence else 'FAILED'}")


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

# Run test for A* and Weighted A* for all permutations of textbooks with size m
def run_test(m, epsilon=None):
    all_states = generate_all_states(m)
    total_start_time = time.time()

    # Iterate through all states
    flips_list = []
    nodes_visited = []
    for state in all_states:
        flips, no_nodes_visited = weighted_a_star_search(state, epsilon=epsilon, N=2 * m) # If espsilon is None, runs A*
        # print(flips)
        # print("Progress: ", len(flips_list) / len(all_states))
        flips_list.append(len(flips))
        nodes_visited.append(no_nodes_visited)

    total_end_time = time.time()
    total_running_time = total_end_time - total_start_time

    print(f"Average number of nodes visited to order the stack of {m} books when epsilon is {epsilon}: ", np.mean(nodes_visited))
    print(f"Average number of flips to order the stack of {m} books when epsilon is {epsilon}: ", np.mean(flips_list))
    print(f"Total running time for {m} books with epsilon {epsilon}: {total_running_time} seconds")

# run_test(1)
# run_test(2)
# run_test(3)
# run_test(4)
# run_test(5)
# run_test(6)
# run_test(7)
# run_test(8)
# run_test(1, 1)
# run_test(2, 1)
# run_test(3, 1)
# run_test(4, 1)
# run_test(5, 1)
# run_test(6, 1)
# run_test(7, 1)
# run_test(8, 1)