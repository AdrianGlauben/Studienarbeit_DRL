################################################################
#### https://www.moderndescartes.com/essays/deep_dive_mcts/ ####
################################################################
import numpy as np
import chess
import time

INIT_STATS = 10e+11
INIT_VISITS = 10e+9

class DummyNode():
    '''
    This class serves as a dummy to hold the statistics of the root node. The roots parent is set to be an instance of this class.
    This is needed since in this implementation the parent node holds the statistics of its children.

    Propeties:
        parent (None): This always holds None to identify it as the end point in the backpropagation.
        child_stats (np.array): Holds the win counts for black and white starting from the root state. White has index 1 and black index 0.
        child_visits (np.array): Holds the visit count of the root node.
        legal_moves (None): This always holds None, but needs to be available for printing the root node.
    '''
    def __init__(self):
        self.parent = None
        self.child_stats = np.array([[INIT_STATS, INIT_STATS]], dtype=np.float32)
        self.child_visits = np.array([INIT_VISITS], dtype=np.float32)
        self.legal_moves = None



class Node():
    '''
    This class represents the nodes with which the search tree is built.

    Propeties:
        board (chess.Board):    A python chess representation of the current game state.
        color ([1, 0]):         One if the color to move is white, 0 if its black.
        move (int):             The index of the move that was played to get this position.
        is_expanded (bool):     False if this node has not been expanded yet. True if it has. Expanded means a rollout was conducted from this node.
        parent (Node):          A reference to the parent node.
        legal_moves (list(chess.Move)): A list of the legal moves for the current game state.
        children (dict):        Contains references to the children. Keys are the move indicies of the children.
        checkmate_idx (int):    If this node has a checkmate in its children, this is the index of the move leading to the checkmate.
        is_terminal (bool):     Indicates wether or not the child is a terminal position.
        child_stats (np.array): Array of shape (number of children, 2) where for each child the win counts for black and white are tracked. Black is index 0, white is index 1.
        child_visits (np.array): An array of shape (number of children,) where the visit counts of the children are tracked.
    '''
    def __init__(self, board, move = 0, parent = DummyNode()):
        self.board = board
        self.color = 1 if board.turn else 0
        self.move = move # Index of the move, NOTE: This is just a number, not the uci string of the move!
        self.is_expanded = False
        self.parent = parent
        self.legal_moves = list(board.legal_moves)
        self.children = dict()
        self.checkmate_idx = None
        self.is_terminal = False
        # Initializing these arrays whith arbitrary but very large numbers will make sure, that all children are expanded.
        self.child_stats = np.full([len(self.legal_moves), 2], INIT_STATS, dtype=np.float32)
        self.child_visits = np.full([len(self.legal_moves)], INIT_VISITS, dtype=np.float32)


    @property
    def stats(self):
        '''
        Gets the stats of the node. This is necessary since the nodes don't know their own stats only his parent does.
        '''
        return self.parent.child_stats[self.move]


    @stats.setter
    def stats(self, statistic):
        '''
        Updates the stats for this node, by updating the child stats of its parent.
        '''
        self.parent.child_stats[self.move] = statistic


    @property
    def visit_count(self):
        '''
        Gets the visit count for this node.
        '''
        return self.parent.child_visits[self.move]


    @visit_count.setter
    def visit_count(self, count):
        '''
        Updates the visit count of this node.
        '''
        self.parent.child_visits[self.move] = count


    def children_win_ratio(self):
        '''
        Calculates the exploitation part of the UCT formula for all children. Q = wins / visits.

        Return:
            (np.array): Array of size (number of children,) containing the win ratio (aka the exploitation part of UCT) for each child.
        '''
        return self.child_stats[:,self.color] / self.child_visits


    def children_exploration_factor(self):
        '''
        Calculates the exploration part of the UCT formula for all children. U = sqrt(parent visits / child visits).

        Return:
            (np.array): Array of size (number of children,) containing the calculation of the explortation part for each child.
        '''
        return np.sqrt(np.log(self.visit_count) / self.child_visits)


    def best_child(self, c):
        '''
        Applies the UCT formula by summing over the exploration and exploitation part.

        Arguments:
            c (float): The parameter governing the trade off between exploration and exploitation. Higher value means more exploration.

        Return:
            (int): The index of the best child according to the UCT value.
        '''
        return np.argmax(self.children_win_ratio() + c * self.children_exploration_factor())


    def most_visited_child(self):
        '''
        Returns the most visited child. This is used at the end of a search to return the node corresponding to the move that should be played.

        Return:
            (node): The child with the highest visit count.
        '''
        try:
            index = np.argmax(self.child_visits)
            child = self.children[index]
        except:
            print(self)
            print(self.board)
            print(self.child_visits)
            print(self.children)
            exit()
        return child


    def __str__(self):
        '''
        Print function for the node. This prints the color, statistics, visit count, number of children, last move played and checkmate index.
        '''
        color = 'White' if self.color == 1 else 'Black'
        move = 'Starting Position' if self.parent.legal_moves is None else self.parent.legal_moves[self.move]
        return f'\n #### {color} #### \n Statistics: {self.stats} \n Visit Count: {self.visit_count} \n Number of Children: {len(self.legal_moves)} \n Last Move: {move} \n Checkmate Index: {self.checkmate_idx} \n'



class MCTS():
    '''
    This class contains the MCTS search.

    Properties:
        expansion_budget (int): Specifies how many nodes should be expanded to conduct the search.
        rollout_budget (int): Specifies how many rollouts should be performed at the leafs.
        c (float): The exploration-exploitation parameter used in the UCT formula. Higher values mean more exploration.
        print_step (int): If not None the number of performed expansions will pe printed every print_step amount of steps.
        root (Node): Holds the root of the ongoing search.
    '''
    def __init__(self, expansion_budget = 22, rollout_budget = 1, c = np.sqrt(2), print_step = None):
        self.expansion_budget = expansion_budget
        self.rollout_budget = rollout_budget
        self.c = c
        self.print_step = print_step
        self.root = None

    def search(self, board=None, use_exisitng_node=None):
        '''
        The main loop conducting the search.

        Arguments:
            board (chess.Board): A python chess board.
            use_exisitng_node (Node): If this is given the search will continue on the given tree instead of building a new tree.
                                        This is used in self play and evaluation to not build a new tree after every move,
                                        but rather reuse a subtree from the previous search.

        Return:
            (Node): The most promising child of the root according to the conducted search.
        '''
        if use_exisitng_node is not None:
            self.root = use_exisitng_node
        else:
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
        '''
        Traverses down the current tree by selecting nodes according to the UCT formula implemented in Node.best_child().

        Arguments:
            node (Node): The tree is always traversed starting from the root. Therefore this is always the root of the tree.

        Return:
            current (Node): This is a leaf of the current tree. Particularly the one chosen by traversing the tree according to the UCT formula.
        '''
        current = node
        while current.is_expanded:
            if current.checkmate_idx is not None:
                current = current.children[current.checkmate_idx]
                break
            elif current.is_terminal:
                break
            else:
                current = self.maybe_add_child(current, current.best_child(self.c))

        # If the leaf is a terminal position, update its flags and return it
        outcome = current.board.outcome()
        if outcome is not None:
            current.is_terminal = True
            if outcome.termination == 1: # If it is a checkmate
                current.parent.checkmate_idx = current.move


        return current


    def rollout(self, node):
        '''
        Plays a number of games defined by the rollout_budget, starting from the position corresponding to the given node.
        Moves are played according to the rollout policy. Return is the result of the game played.

        Arguments:
            node (Node): The leaf from which a rollout should be performed.

        Return:
            result (np.array): An array of size 3 containing the follwing statistics of the performed rollouts:
                                [No. wins for Black, No. of wins for White, No. of Draws]
        '''
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
            move = np.random.choice([m for m in board.legal_moves])
            board.push(move)
        return board.outcome().winner


    def backpropagate(self, stats, visits, node):
        '''
        Backpropagates the result from the rollout, along the path taken by the traverse function. On the way visit counts and win counts are updated.

        Arguments:
            stats (np.array): Array of size 2 containing the wins for black and white respectively:
                                [No. of wins for Black, No. of wins for White]
            visits (int): Essentially this is just the rollout_budget. So the number of games played during rollout.
            node (Node): The leaf node from which backpropagation should be started.
        '''
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
        '''
        This function is used to add new nodes to the current tree.
        It checks if the move chosen by the UCT formula already has a corresponding node in the tree, if so it returns that node, if not it adds a node.

        Arguments:
            node (Node): The node to which a new child should be added.
            move (int): The index of the move chosen by the UCT formula.

        Returns:
            (Node): Either an existing node corresponding to the move or a newly added node.
        '''
        if move not in node.children:
            board = node.board.copy()
            board.push(node.legal_moves[move])
            node.children[move] = Node(
                board, move = move, parent = node)
        return node.children[move]
