import numpy as np
import chess
from psutil import cpu_count
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class Node:
    '''
    This class represents the nodes with which the search tree is built.

    Propeties:
        board (chess.Board):    A python chess representation of the current game state.
        parent (Node):          A reference to the parent node.
        color (Optional[chess.COLOR]): chess.WHITE or chess.BLACK, the color of the player that has to make a move.
        stats (list):           List of size 3 containing the statistics for this node: [No. wins for Black, No. of wins for White, No. of Draws]
        is_expanded (bool):     False if this node has not been expanded yet. True if it has. Expanded means a rollout was conducted from this node.
        is_f_expanded (bool):   True if all children of this node are expanded.
        children (list):        Contains references to the children.
        is_terminal (bool):     Indicates wether or not the child is a terminal position.
        is_checkmate (bool):    Indicates if this node is a checkmate.
        checkmate_idx (int):    If this node has a checkmate in its children, this is the index of the move leading to the checkmate.

    '''
    def __init__(self, board, parent = None, root = False):
        self.board = board
        self.parent = parent
        self.color = board.turn # The color to move
        self.stats = [0, 0, 0] # [Blacks wins, Whites wins, Draws]
        self.children = list()
        self.is_expanded = True if root else False
        self.is_f_expanded = False # Weather or not all children are expanded
        self.is_terminal = False
        self.is_checkmate = False
        self.checkmate_idx = None # If this node has a checkmate in its children, this is the index of that position
        if root:
            self.make_children()


    def is_root(self):
        '''
        Returns a boolean value indicating whether or not this node is the root.
        '''
        # Checks if this node is the root node
        if self.parent is not None:
            return False
        else:
            return True


    def is_fully_expanded(self):
        '''
        Checks if this node is fully expanded and if so sets its corresponding flag.
        Returns a boolean indicating if it is fully expanded.
        '''
        # Checks if this node is fully expanded
        # Which is the case if all children are expanded
        if self.is_f_expanded:
            return True
        if not self.children:
            return False
        else:
            children_status = [child.is_expanded for child in self.children]
            self.is_f_expanded = all(children_status)
            return self.is_f_expanded # True if all in list are True, False otherwise


    def best_child(self):
        '''
        Returns the child with the highest visit count.

        Return:
            (Node): The child of this node with the highest visit count.
        '''
        # Returns the best child of this node by evaluating their visit count
        child_values = [np.sum(child.stats) for child in self.children]
        return self.children[np.argmax(child_values)]


    def make_children(self):
        '''
        Generates all children of this node.
        This is done by creating a new node for every legal move in this position and appending it to the list of children.
        '''
        # Generates all legal moves in this node
        # and safes the corresponding nodes in self.children
        if not self.children:
            for m in self.board.legal_moves:
                board = self.board.copy()
                board.push_san(board.san(m))
                self.children.append(Node(board, parent = self))


    def __str__(self):
        '''
        A printing function for the Node. Prints the color of the player whos turn it is, the stats of the node and the number of children.
        '''
        # Prints the statistics of this node
        color = "White" if self.color else "Black"
        return f" \n #### {color} #### \n Stats: {self.stats} \n Number of Children: {len(self.children)}"

class Vanilla_MCTS:
    '''
    A very basic MCTS implementation.

    Properties:
        print_budget (bool): If True is given, then the remaining expansion budget is printed to the console every 500 steps.
        expansion_budget (int): The number of expansions that should be performed by the search.
        rollout_budget (int): The number of rollout games that should be played at the leafes.
        c (float): The exploration parameter for the UCT fromula.

    '''
    def __init__(self, expansion_budget = 100, rollout_budget = 1, c = np.sqrt(2), print_budget = False):
        self.print_budget = print_budget
        self.expansion_budget = expansion_budget
        self.rollout_budget = rollout_budget
        self.c = c


    def search(self, root):
        '''
        The main loop performing the search.

        Arguments:
            root (Node): The root of the tree to be searched.

        Returns:
            (Node): The most promising child according to the search. Selected by root.best_child() at the end of the search.
        '''
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
                # Expand the Leaf
                leaf.make_children()
                leaf.is_expanded = True
                # Perform a rollout for the leaf
                result = self.rollout(leaf)

            # Backpropagate the results
            self.backpropagate(leaf, result)

            # Budget decrementation and printing
            budget -= 1
            if self.print_budget and budget % 500 == 0:
                print(f" \n ----- Expansion Budget: {budget} -----")

        return root.best_child()


    def traverse(self, node):
        '''
        Traverses down the current tree by selecting nodes according to the UCT formula implemented in self.uct(node).

        Arguments:
            node (Node): The tree is always traversed starting from the root. Therefore this is always the root of the tree.

        Return:
            (Node): This is a leaf of the current tree. Particularly the one chosen by traversing the tree according to the UCT formula.
                    If the last fully expanded node is reached a random child of this node is returned as the leaf.
        '''
        #########################
        #### Selection Phase ####
        #########################
        # Traverses the currently fully expanded nodes and returns a random leaf to expand
        while node.is_fully_expanded():
            # If the node has a checkmate in its children, select that node always
            if node.checkmate_idx is not None:
                return node.children[node.checkmate_idx]

            # Use UCT rule to determin the selected child
            children_uct = [self.uct(child) for child in node.children]
            child_idx = np.argmax(children_uct)
            node = node.children[child_idx]

        outcome = node.board.outcome()

        # If the leaf is a terminal position, update its flags and return it
        if outcome is not None:
            node.is_terminal = True
            if outcome.winner is not None:
                node.is_checkmate = True
                # Give parent node the infromation that it has a checkmate position in children
                node.parent.checkmate_idx = child_idx
                node.parent.is_f_expanded = True
            return node
        else:
            # Return a random child to be expanded
            unvisited_children = [child for child in node.children if child.is_expanded == False]
            return np.random.choice(unvisited_children)


    def rollout(self, node):
        '''
        Perfroms a number of rollouts according to the rollout budget. It uses the rollout_policy to play games.

        Arguments:
            node (Node): The node from which the rollout should be started.

        Return:
            resukt (list): The result of the performed rollouts:
                            [No. of wins for Black, No. of wins for White, No. of Draws]
        '''
        ##########################################
        #### Rollout/Simulation/Playout Phase ####
        ##########################################
        # Uses the rollout policy to make moves until the end of the game and returns the results

        result = [0, 0, 0] # [Blacks wins, Whites wins, Draws]
        budget = self.rollout_budget

        while budget != 0:
            board = node.board.copy()
            winner = self.rollout_policy(board)
            if winner is not None:
                result[winner] += 1
            else:
                result[2] += 1
            budget -= 1

        return result


    def rollout_policy(self, board):
        '''
        Plays a game from a given starting position to the end. This implementation plays random moves.

        Arguments:
            board (chess.Board): The current board position from which a game should be played.

        Returns:
            winner (Optional[chess.COLOR]): The winner of the rollout.
                                            chess.WHITE if white wins, chess.BLACK if black wins, None for outcomes other than checkmate.

        '''
        # Random games

        while not board.is_game_over():
            move = board.san(np.random.choice([m for m in board.legal_moves]))
            board.push_san(move)

        return board.outcome().winner


    def backpropagate(self, node, result):
        '''
        Backpropagates the result of the rollout back through the tree along the path taken by the traverse function.
        This is done by simply adding the result to the node.stats.

        Arguments:
            node (Node): The leaf node from which the results are backpropagated.
            result (list): The result returned by the rollout.
        '''
        ###############################
        #### Backpropagation Phase ####
        ###############################
        # Backpropagates the results from the rollout through the path taken
        node.stats = np.add(result, node.stats).tolist()

        if node.is_root():
            return

        self.backpropagate(node.parent, result)


    def uct(self, node):
        '''
        Implements the UCT formula.

        Arguments:
            node (Node): The node for which the UCT value should be calculated.

        Return:
            (float): The UCT value of the node.
        '''
        # UCT Selection Rule
        win_count = node.stats[node.parent.color]
        visit_count = np.sum(node.stats, axis = 0)
        parent_visit_count = np.sum(node.parent.stats, axis = 0)
        return win_count / visit_count + self.c * np.sqrt(np.log(parent_visit_count) / visit_count)



class Parallel_Roullout_MCTS(Vanilla_MCTS):
    '''
    An implementation of MCTS that performs rollouts at the leafs in parallel.

    Arguments:
        no_of_cpus (int): The number of CPU cores that should be used to perform the rollouts in parallel.
        For the rest see above.
    '''
    def __init__(self, expansion_budget = 100, rollout_budget = 1, c = np.sqrt(2), print_budget = False, no_of_cpus = cpu_count(logical=False)):
        super().__init__(expansion_budget, rollout_budget, c, print_budget)
        self.no_of_cpus = no_of_cpus

    def search(self, root):
        '''
        A parallel implementation of the search function. Does the same as described above,
        with the only difference that during the rollout phase multiple processes are spawned to pefrom the rollouts in parallel.
        '''
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
                # Expand the Leaf
                leaf.make_children()
                leaf.is_expanded = True
                ########## Parallel Processing ############
                # Perform a rollout for the leaf and backpropagate the result
                args = [leaf for i in range(self.no_of_cpus)]
                with mp.Pool(self.no_of_cpus) as pool:
                    mp_result = pool.map(self.rollout, args)
                rollout_result = np.sum(mp_result, axis=0)

            self.backpropagate(leaf, rollout_result)

            # Budget decrementation and printing
            budget -= 1
            if self.print_budget and budget % 500 == 0:
                print(f" \n ----- Expansion Budget: {budget} -----")

        return root.best_child()
