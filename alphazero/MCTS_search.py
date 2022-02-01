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
        self.last_moves = [None] * 8



class Node():
    def __init__(self, board, move = 0, parent = DummyNode()):
        self.board = board
        self.color = board.turn
        self.move = move # Index of the move, NOTE: This is just a number, not the uci string of the move!
        self.is_expanded = False
        self.is_terminal = False
        self.parent = parent
        self.legal_moves = [m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN] # Filter underpromotions
        self.children = dict()
        self.checkmate_idx = None
        self.child_total_values = np.full([len(self.legal_moves)], 0, dtype=np.float32)
        self.child_visits = np.zeros([len(self.legal_moves)], dtype=np.float32)
        self.child_priors = np.zeros([len(self.legal_moves)], dtype=np.float32)

        # Update last_moves
        self.last_moves = parent.last_moves.copy()
        if self.parent.legal_moves is not None:
            move = self.parent.legal_moves[self.move]
            self.last_moves.append(move)
            self.last_moves = self.last_moves[-8:]


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
        return np.sqrt(np.sum(self.child_visits)) * (
            self.child_priors / (1 + self.child_visits))


    def best_child(self):
        try:
            index = np.argmax(self.child_Q() + 2 * self.child_U())
        except:
            print(self)
            print('')
            print(self.board)
            print('Legal Moves')
            print(self.legal_moves)
            print('Outcome')
            print(self.board.outcome())
            exit()
        return index


    def most_visited_child(self):
        return self.children[np.argmax(self.child_visits)]


    def sample_child(self):
        probs = self.child_visits / np.sum(self.child_visits)
        if self.board.fullmove_number < 5:
            index = np.random.choice(np.arange(len(probs)), p=probs)
            return self.children[index]
        else:
            return self.children[np.argmax(probs)]


    def __str__(self):
        color = 'White' if self.color == 1 else 'Black'
        move = 'Starting Position' if self.parent.legal_moves is None else self.parent.legal_moves[self.move]
        return f'\n #### {color} #### \n Value: {self.total_value / self.visit_count} \n Visit Count: {self.visit_count} \n Number of Children: {len(self.legal_moves)} \n Last Move: {move} \n Checkmate Index: {self.checkmate_idx} \n'



class MCTS():
    def __init__(self, net, expansion_budget = 22, device=torch.device('cpu'), play_mode=False, noise_alpha=0.3, noise_epsilon=0.25):
        self.expansion_budget = expansion_budget
        self.root = None
        self.net = net
        self.device = device
        self.play_mode = play_mode
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon


    def search(self, board=None, use_exisitng_node=None):
        if use_exisitng_node is not None:
            self.root = use_exisitng_node
        else:
            self.root = Node(board)

        for i in range(self.expansion_budget):
            # Select
            leaf = self.traverse(self.root)
            # Get NN prediction
            input = board_to_planes(leaf.board, leaf.last_moves).to(self.device) # Add dimension for NN input
            value, priors = self.net(input)
            # Backprop result
            self.backpropagate(value.detach().cpu().numpy(), leaf)
            # Expand
            leaf.is_expanded = True
            priors = priors.detach().cpu().numpy()[0] # Transform torch tensor to numpy array
            priors = priors.reshape((64, 8, 8))
            leaf.child_priors = planes_to_move_probabilities(priors, leaf.legal_moves, leaf.color)
            if not self.play_mode:
                leaf.child_priors = (1-self.noise_epsilon) * leaf.child_priors + self.noise_epsilon * np.random.dirichlet([self.noise_alpha] * len(leaf.child_priors))

        if self.play_mode:
            return self.root.most_visited_child()
        else:
            return self.root.sample_child()


    def traverse(self, node):
        current = node
        while current.is_expanded:
            if current.checkmate_idx is not None:
                return current.children[current.checkmate_idx]
            elif current.is_terminal:
                break
            else:
                current = self.maybe_add_child(current, current.best_child())

        # If the leaf is a terminal position, update its flags and return it
        outcome = current.board.outcome()
        if outcome is not None:
            current.is_terminal = True
            if outcome.termination == 1: # If it is a checkmate
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
        while node.parent is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent
