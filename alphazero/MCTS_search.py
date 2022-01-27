import numpy as np
import chess
from input_representation import board_to_planes
from output_representation import legal_moves_to_flat_planes, planes_to_move_probabilities
from net import AlphaZeroResNet
import torch

class DummyNode():
    # A dummy node to serve as parent for the root, holding its statistics
    def __init__(self):
        self.parent = None
        self.child_total_values = np.array([0], dtype=np.float32)
        self.child_visits = np.array([0], dtype=np.float32)
        self.child_priors = np.array([0], dtype=np.float32)
        self.legal_moves = None



class Node():
    def __init__(self, board, move = 0, parent = DummyNode()):
        self.board = board
        self.color = board.turn
        self.move = move # Index of the move, NOTE: This is just a number, not the uci string of the move!
        self.is_expanded = False
        self.parent = parent
        self.legal_moves = [m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN] # Filter underpromotions
        self.children = dict()
        self.checkmate_idx = None
        self.child_total_values = np.full([len(self.legal_moves)], 0, dtype=np.float32)
        self.child_visits = np.zeros([len(self.legal_moves)], dtype=np.float32)
        self.child_priors = np.zeros([len(self.legal_moves)], dtype=np.float32)


    @property
    def total_value(self):
        return self.parent.child_total_values[self.move]


    @total_value.setter
    def total_value(self, statistic):
        self.parent.child_total_values[self.move] = statistic


    @property
    def visit_count(self):
        return self.parent.child_visits[self.move]


    @visit_count.setter
    def visit_count(self, count):
        self.parent.child_visits[self.move] = count


    def child_Q(self):
        return self.child_total_values / (1 + self.child_visits)


    def child_U(self):
        return np.sqrt(self.visit_count) * (
            self.child_priors / (1 + self.child_visits))


    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())


    def most_visited_child(self):
        return self.children[np.argmax(self.child_visits)]


    def __str__(self):
        color = 'White' if self.color == 1 else 'Black'
        move = 'Starting Position' if self.parent.legal_moves is None else self.parent.legal_moves[self.move]
        return f'\n #### {color} #### \n Value: {self.total_value / self.visit_count} \n Visit Count: {self.visit_count} \n Number of Children: {len(self.legal_moves)} \n Last Move: {move} \n Checkmate Index: {self.checkmate_idx} \n'



class MCTS():
    def __init__(self, net, expansion_budget = 22, print_step = None):
        self.expansion_budget = expansion_budget
        self.root = None
        self.print_step = print_step
        self.net = net
        self.last_moves = [None] * 8


    def search(self, board):
        self.root = Node(board)
        for i in range(self.expansion_budget):
            # Select
            leaf = self.traverse(self.root)
            # Get NN prediction
            input = board_to_planes(leaf.board, self.last_moves)
            input = input[None, :] # Add dimension for NN input
            value, priors = self.net(input)
            # Backprop result
            self.backpropagate(value.detach().numpy(), leaf)
            # Expand
            leaf.is_expanded = True
            priors = priors.detach().numpy() # Transform torch tensor to numpy array
            priors = priors.reshape((64, 8, 8))
            leaf.child_priors = planes_to_move_probabilities(priors, leaf.legal_moves, leaf.color)
            # Update last moves
            if leaf != self.root:
                move = leaf.parent.legal_moves[leaf.move]
                self.last_moves.append(move)
                self.last_moves = self.last_moves[-8:]
            # Print step
            if self.print_step is not None and i % self.print_step == 0:
                print(f'----- Expansion Step: {i} -----')
        return self.root.most_visited_child()


    def traverse(self, node):
        current = node
        while current.is_expanded:
            if current.checkmate_idx is not None:
                current = current.children[current.checkmate_idx]
                break
            else:
                current = self.maybe_add_child(current, current.best_child())

        # If the leaf is a terminal position, update its flags and return it
        if current.board.is_checkmate():
            current.parent.checkmate_idx = current.move

        return current


    def maybe_add_child(self, node, move):
        if move not in node.children:
            board = node.board.copy()
            board.push(node.legal_moves[move])
            node.children[move] = Node(
                board, move = move, parent = node)
        return node.children[move]


    def backpropagate(self, value, node):
        if node.parent is None:
            return
        node.visit_count += 1
        node.total_value += value
        self.backpropagate(-value, node.parent)
