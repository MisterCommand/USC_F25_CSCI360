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

        # If the stack deque is ordered, return the flip sequence
        if current_stack[0].check_ordered():
            return current_stack[1]

        # for each flip in the sequence
        for flip in range(1, current_stack[0].num_books + 1):
            new_stack = current_stack[0].copy()
            new_stack.flip_stack(flip)
            new_sequence = current_stack[1] + [flip]
            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                queue.append((new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))

    return current_stack[1]
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
            return current_stack[1]

        # for each flip in the sequence
        for flip in range(1, current_stack[0].num_books + 1):
            new_stack = current_stack[0].copy()
            new_stack.flip_stack(flip)
            new_sequence = current_stack[1] + [flip]

            # if the new stack is not visited, add it to the queue and visited set
            if (tuple(new_stack.order), tuple(new_stack.orientations)) not in visited:
                queue.append((new_stack, new_sequence))
                visited.add((tuple(new_stack.order), tuple(new_stack.orientations)))

    return current_stack[1]
    # ---------------------------- #