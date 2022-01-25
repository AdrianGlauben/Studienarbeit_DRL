################################################################
#### https://www.moderndescartes.com/essays/deep_dive_mcts/ ####
################################################################
import collections
import math
import numpy as np
import chess
import time

INIT_STATS = 10e+11
INIT_VISITS = 10e+9

class DummyNode():
    # A dummy node to serve as parent for the root, holding its statistics
    def __init__(self):
        self.parent = None
        self.child_stats = np.array([[INIT_STATS, INIT_STATS]], dtype=np.float32)
        self.child_visits = np.array([INIT_VISITS], dtype=np.float32)
        self.legal_moves = None

        @property
        def stats(self):
            return self.parent.child_stats[self.move]

        @stats.setter
        def stats(self, statistic):
            self.parent.child_stats[self.move] = statistic

        @property
        def visit_count(self):
            return self.parent.child_visits[self.move]

        @visit_count.setter
        def visit_count(self, count):
            self.parent.child_visits[self.move] = count



class Node():
    def __init__(self, board, move = 0, parent = DummyNode()):
        self.board = board
        self.color = 1 if board.turn else 0
        self.move = move # Index of the move, NOTE: This is just a number, not the uci string of the move!
        self.is_expanded = False
        self.parent = parent
        self.legal_moves = list(board.legal_moves)
        self.children = dict()
        self.checkmate_idx = None
        # Initializing these arrays whith arbitrary but very large numbers will make sure, that all children are expanded
        self.child_stats = np.full([len(self.legal_moves), 2], INIT_STATS, dtype=np.float32)
        self.child_visits = np.full([len(self.legal_moves)], INIT_VISITS, dtype=np.float32)

    @property
    def stats(self):
        return self.parent.child_stats[self.move]

    @stats.setter
    def stats(self, statistic):
        self.parent.child_stats[self.move] = statistic

    @property
    def visit_count(self):
        return self.parent.child_visits[self.move]

    @visit_count.setter
    def visit_count(self, count):
        self.parent.child_visits[self.move] = count

    def children_win_ratio(self):
        return self.child_stats[:,self.color] / self.child_visits

    def children_exploration_factor(self):
        return np.sqrt(np.log(self.visit_count) / self.child_visits)

    def best_child(self, c):
        return np.argmax(self.children_win_ratio() + c * self.children_exploration_factor())

    def most_visited_child(self):
        return self.children[np.argmax(self.child_visits)]

    def __str__(self):
        color = 'White' if self.color == 1 else 'Black'
        move = 'Starting Position' if self.parent.legal_moves is None else self.parent.legal_moves[self.move]
        return f'\n #### {color} #### \n Statistics: {self.stats} \n Visit Count: {self.visit_count} \n Number of Children: {len(self.legal_moves)} \n Last Move: {move} \n Checkmate Index: {self.checkmate_idx} \n'



class MCTS():
    def __init__(self, expansion_budget = 22, rollout_budget = 1, c = np.sqrt(2), print_step = None):
        self.expansion_budget = expansion_budget
        self.rollout_budget = rollout_budget
        self.c = c
        self.print_step = print_step
        self.root = None

    def search(self, board):
        self.root = Node(board)
        for i in range(self.expansion_budget):
            leaf = self.traverse(self.root)
            rollout_result = self.rollout(leaf)
            self.backpropagate(rollout_result[:2], np.sum(rollout_result), leaf)
            leaf.is_expanded = True

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
                current = self.maybe_add_child(current, current.best_child(self.c))

        # If the leaf is a terminal position, update its flags and return it
        if current.board.is_checkmate():
            current.parent.checkmate_idx = current.move

        return current

    def rollout(self, node):
        result = np.array([0, 0, 0]) # [Blacks wins, Whites wins, Draws]

        for _ in range(self.rollout_budget):
            board = node.board.copy()
            winner = self.rollout_policy(board)
            if winner is not None:
                winner = 1 if winner else 0
                result[winner] += 1
            else:
                result[2] += 1
        return result

    def rollout_policy(self, board):
        # Random games
        while not board.is_game_over():
            move = np.random.choice([m for m in board.legal_moves])
            board.push(move)
        return board.outcome().winner

    def backpropagate(self, stats, visits, node):
        if node.parent is None:
            return

        if node.visit_count == INIT_VISITS:
            # If the nodes stats are not yet initialized, initialize them.
            node.stats = stats
            node.visit_count = visits
        else:
            node.stats += stats
            node.visit_count += visits

        self.backpropagate(stats, visits, node.parent)

    def maybe_add_child(self, node, move):
        if move not in node.children:
            board = node.board.copy()
            board.push(node.legal_moves[move])
            node.children[move] = Node(
                board, move = move, parent = node)
        return node.children[move]



class MC_RAVE(MCTS):
    def __init__(self, expansion_budget = 22, rollout_budget = 1, c = np.sqrt(2), print_step = None):
        super().__init__(expansion_budget, rollout_budget, c, print_step)
        self.tranpositions = dict()




#### DO NOT USE, IMPLEMENTATION IS INCORRECT ####
class MCTS_TT(MCTS):
    def __init__(self, expansion_budget = 22, rollout_budget = 1, c = np.sqrt(2), print_step = None):
        super().__init__(expansion_budget, rollout_budget, c, print_step)
        self.tranpositions = dict()

    def maybe_add_child(self, node, move):
        if move not in node.children:
            board = node.board.copy()
            board.push(node.legal_moves[move])

            board_id = board.board_fen() + ' ' + str(node.color) # Gets the board identifying part of the fen representation
            new_node = Node(board, move = move, parent = node)
            if board_id in self.tranpositions:
                # Update new node with known Statistics
                self.use_transposition_stats(self.tranpositions[board_id], new_node)
            else:
                self.tranpositions[board_id] = new_node

            node.children[move] = new_node
        return node.children[move]

    def use_transposition_stats(self, known_node, new_node):
        new_node.children = known_node.children
        new_node.checkmate_idx = known_node.checkmate_idx
        new_node.child_stats = known_node.child_stats
        new_node.child_visits = known_node.child_visits
        self.backpropagate(known_node.stats, known_node.visit_count, new_node)
