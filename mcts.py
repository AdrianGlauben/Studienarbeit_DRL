import numpy as np
import chess
from psutil import cpu_count
import multiprocessing as mp

class Node:
    def __init__(self, board, parent = None, color = chess.WHITE, root = False):
        self.board = board
        self.parent = parent
        self.color = color # The color to move
        self.visit_count = 0
        self.win_count = 0
        self.children = list()
        self.is_expanded = True if root else False
        self.is_terminal = False
        self.is_checkmate = False
        if root == True:
            self.make_children()


    def is_root(self):
        # Checks if this node is the root node
        if self.parent is not None:
            return False
        else:
            return True


    def is_fully_expanded(self):
        # Checks if this node is fully expanded
        # Which is the case if all children are expanded
        if not self.children:
            return False
        else:
            children_status = [child.is_expanded for child in self.children]
            return all(children_status) # True if all in list are True, False otherwise


    def best_child(self):
        # Returns the best child of this node by evaluating their visit_count
        child_values = [child.visit_count for child in self.children]
        return self.children[np.argmax(child_values)]


    def make_children(self):
        # Generates all legal moves in this node
        # and safes the corresponding nodes in self.children
        for m in self.board.legal_moves:
            board = self.board.copy()
            board.push_san(board.san(m))
            self.children.append(Node(board, parent = self, color = not self.color))


    def __str__(self):
        # Prints the statistics of this node
        color = "White" if self.color else "Black"
        return f" \n #### {color} #### \n Visit Count: {self.visit_count} \n Win Count: {self.win_count} \n Number of Children: {len(self.children)} \n Expanded: {self.is_expanded}"

class Vanilla_MCTS:
    def __init__(self, expansion_budget = 100, rollout_budget = 1, c = 1, print_budget = False):
        self.print_budget = print_budget
        self.expansion_budget = expansion_budget
        self.rollout_budget = rollout_budget
        self.c = c


    def search(self, root):
        # Uses the given budgets to perform a search and return a move suggestion
        budget = self.expansion_budget
        while budget != 0:
            # Select a leaf of the current game tree
            leaf = self.traverse(root)

            if leaf.is_terminal:
                # If the leaf is an ending position dont perfrom rollouts
                result = [0, 0, 0]
                if leaf.is_checkmate:
                    result[not leaf.color] = self.rollout_budget
                else:
                    result[2] = self.rollout_budget
            else:
                # Perform a rollout for the leaf and backpropagate the result
                leaf.make_children()
                leaf.is_expanded = True

                result = self.rollout(leaf)

            # Backpropagate the results
            self.backpropagate(leaf, result)

            # Budget decrementation and printing
            budget -= 1
            if self.print_budget and budget % 50 == 0:
                print(f" \n ----- Expansion Budget: {budget} -----")

        return root.best_child()


    def traverse(self, node):
        # Selection Phase:
        # Traverses the currently fully expanded nodes and returns a promising leaf to expand
        while node.is_fully_expanded():
            children_uct = [self.uct(child) for child in node.children]
            node = node.children[np.argmax(children_uct)]

        outcome = node.board.outcome()

        if outcome is not None:
            node.is_terminal = True
            if outcome.winner is not None:
                node.is_checkmate == True
            return node
        else:
            unvisited_children = [child for child in node.children if child.is_expanded == False]
            return np.random.choice(unvisited_children)


    def rollout(self, node):
        # Simulation/Rollout Phase:
        # Uses the rollout policy to make moves until the end of the game and returns the results

        result = [0, 0, 0]
        budget = self.rollout_budget

        while budget != 0:
            board = node.board.copy()
            winner = self.rollout_policy(board)
            if winner == chess.WHITE:
                result[0] += 1
            elif winner == chess.BLACK:
                result[1] += 1
            else:
                result[2] += 1
            budget -= 1

        return result


    def rollout_policy(self, board):
        # Random games
        game_over = False

        while not game_over:
            outcome = board.outcome()
            if outcome is not None:
                game_over = True
                break
            move = board.san(np.random.choice([m for m in board.legal_moves]))
            board.push_san(move)

        return outcome.winner


    def backpropagate(self, node, result):
        # Backpropagation Phase:
        # Backpropagates the results from the rollout through the path taken
        node.visit_count += np.sum(result)
        if node.color == chess.WHITE:
            node.win_count += result[0]
        elif node.color == chess.BLACK:
            node.win_count += result[1]

        if node.is_root():
            return

        self.backpropagate(node.parent, result)


    def uct(self, node):
        # UCT Selection Rule
        return node.win_count / node.visit_count + self.c * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)



class Parallel_MCTS(Vanilla_MCTS):
    def __init__(self, expansion_budget = 100, rollout_budget = 1, c = 1, print_budget = False, no_of_cpus = cpu_count(logical=False)):
        super().__init__(expansion_budget, rollout_budget, c, print_budget)
        self.no_of_cpus = no_of_cpus

    def search(self, root):
        # Uses the given budgets to perform a search and return a move suggestion
        budget = self.expansion_budget
        while budget != 0:
            # Select a leaf of the current game tree
            leaf = self.traverse(root)

            # When the leaf is an ending position...
            if self.game_end_found:
                self.game_end_found = False
                return root.best_child()

            # Perform a rollout for the leaf and backpropagate the result
            leaf.make_children()
            leaf.is_expanded = True

            ########## Parallel Processing ############
            args = [leaf for i in range(self.no_of_cpus)]
            with mp.Pool(self.no_of_cpus) as pool:
                mp_result = pool.map(self.rollout, args)
            rollout_result = np.sum(mp_result, axis=0)

            self.backpropagate(leaf, rollout_result)

            # Budget decrementation and printing
            budget -= 1
            if self.print_budget and budget % 50 == 0:
                print(f" \n ----- Expansion Budget: {budget} -----")

        return root.best_child()
