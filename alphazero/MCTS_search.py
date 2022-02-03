import numpy as np
import chess
from input_representation import board_to_planes
from output_representation import legal_moves_to_flat_planes, planes_to_move_probabilities
from net import AlphaZeroResNet
import torch

class DummyNode():
    '''
    This class serves as a dummy to hold the statistics of the root node. The roots parent is set to be an instance of this class.
    This is needed since in this implementation the parent node holds the statistics of its children.

    Propeties:
        parent (None): This always holds None to identify it as the end point in the backpropagation.
        child_total_values (np.array):  Array with only 1 entry. Holds the total value of the root node.
        child_visits (np.array): Array with only 1 entry. Holds the visit count of the root node.
        child_priors (np.array): Array with only 1 entry. Holds the prior probabilites for the legal moves available in the root.
        legal_moves (None): This always holds None, but needs to be available for printing the root node.
        last_moves (list(None)): A list containing None 8 times.
                                    This is used to initialize the list of last 8 moves, which is used for the input representation.

    '''
    # A dummy node to serve as parent for the root, holding its statistics
    def __init__(self):
        self.parent = None
        self.child_total_values = np.array([0], dtype=np.float32)
        self.child_visits = np.array([0], dtype=np.float32)
        self.child_priors = np.array([0], dtype=np.float32)
        self.legal_moves = None
        self.last_moves = [None] * 8



class Node():
    '''
    This class represents the nodes with which the search tree is built.

    Propeties:
        board (chess.Board):    A python chess representation of the current game state.
        color (Optional[chess.COLOR]): chess.WHITE or chess.BLACK. The color whos turn it is.
        move (int):             The index of the move that was played to get this position.
        is_expanded (bool):     False if this node has not been expanded yet. True if it has. Expanded means a rollout was conducted from this node.
        is_terminal (bool):     Indicates wether or not the child is a terminal position.
        parent (Node):          A reference to the parent node.
        legal_moves (list(chess.Move)): A list of the legal moves for the current game state. Exluding underpromotions.
        children (dict):        Contains references to the children. Keys are the move indicies of the children.
        checkmate_idx (int):    If this node has a checkmate in its children, this is the index of the move leading to the checkmate.
        child_total_values (np.array): Size (len(legal_moves),). Contains the value predictions for this nodes children.
                                        While each node is only predicted once, thorugh backpropagation the values from the complete subtree will acumulate here.
        child_visits (np.array): Size (len(legal_moves),). The visit counts of this nodes children.
        child_priors (np.array): Size (len(legal_moves),). The prior porbabilities for all children predicted by the neural network.
    '''
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
        '''
        Gets the total value of the node by retrieving it from its parent node.
        '''
        return self.parent.child_total_values[self.move]


    @total_value.setter
    def total_value(self, statistic):
        '''
        Sets the total value of a node by setting the value in its parent.
        '''
        self.parent.child_total_values[self.move] = statistic


    @property
    def visit_count(self):
        '''
        Gets the visit count of a node by retrieving it from its parent.
        '''
        return self.parent.child_visits[self.move]


    @visit_count.setter
    def visit_count(self, count):
        '''
        Sets the visit coun of a node by setting the value in its parent.
        '''
        self.parent.child_visits[self.move] = count


    def child_Q(self):
        '''
        Calculates the exploitation part of the pUCT formula and returns it.
        '''
        return self.child_total_values / (1 + self.child_visits)


    def child_U(self):
        '''
        Calculates the exploration part of the pUCT fromula and returns it.
        '''
        return np.sqrt(np.sum(self.child_visits)) * (
            self.child_priors / (1 + self.child_visits))


    def best_child(self, c):
        '''
        Returns the best child according to the pUCT formula.

        Arugments:
            c (float): The exploration paramter. Higher value means mor weight on exploration. Standard is 2.5.
        '''
        index = np.argmax(self.child_Q() + c * self.child_U())
        return index


    def most_visited_child(self):
        '''
        Returns the best child according to the visit count.
        '''
        return self.children[np.argmax(self.child_visits)]


    def sample_child(self, cut_off):
        '''
        Either samples a child with probabilites with respect to their relative visit count or choses the most visited child.
        The latter happens if more then the number of full moves speciefied by the cut_off paramter have ben played.

        Arguments:
            cut_off (int): Specifies the number of full moves after which moves should not be sampled anymore.

        Return:
            (Node): The chosen child.
        '''
        probs = self.child_visits / np.sum(self.child_visits)
        if self.board.fullmove_number < cut_off:
            index = np.random.choice(np.arange(len(probs)), p=probs)
            return self.children[index]
        else:
            return self.children[np.argmax(probs)]


    def __str__(self):
        '''
        Print function for the node. This prints the color, average value, visit count, number of children, last move played and checkmate index.
        '''
        color = 'White' if self.color == 1 else 'Black'
        move = 'Starting Position' if self.parent.legal_moves is None else self.parent.legal_moves[self.move]
        return f'\n #### {color} #### \n Value: {self.total_value / self.visit_count} \n Visit Count: {self.visit_count} \n Number of Children: {len(self.legal_moves)} \n Last Move: {move} \n Checkmate Index: {self.checkmate_idx} \n'



class MCTS():
    '''
    This class contains the MCTS search.

    Properties:
        expansion_budget (int): Specifies how many nodes should be expanded to conduct the search.
        root (Node): Holds the root of the ongoing search.
        net (AlphaZeroResNet): A reference to the neural network that should be used.
        device (torch.device): Sepcifies the device on which the neural network evaluation should be performed.
        play_mode (bool): If this is True moves are always selected by root.most_visited_child().
        noise_alpha (float): The alpha parameter for the Dirichlet noise.
        noise_epsilon (float): A paramter specified the ratio between the cumpted priors and the added noise.
        c (float): The exploration-exploitation parameter used in the pUCT formula. Higher values mean more exploration.
        cut_off (int): The number of full moves after which the chosen child is not sampled anymore. See: Node.sample_child.
    '''
    def __init__(self, net, expansion_budget = 22, device=torch.device('cpu'), play_mode=False, noise_alpha=0.3, noise_epsilon=0.25, c = 2.5, cut_off=10):
        self.expansion_budget = expansion_budget
        self.root = None
        self.net = net
        self.device = device
        self.play_mode = play_mode
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon
        self.c = c
        self.cut_off = cut_off


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
            leaf.child_priors = (1-self.noise_epsilon) * leaf.child_priors + self.noise_epsilon * np.random.dirichlet([self.noise_alpha] * len(leaf.child_priors))

        if self.play_mode:
            return self.root.most_visited_child()
        else:
            return self.root.sample_child(self.cut_off)


    def traverse(self, node):
        '''
        Traverses down the current tree by selecting nodes according to the pUCT formula implemented in Node.best_child().

        Arguments:
            node (Node): The tree is always traversed starting from the root. Therefore, this is always the root of the tree.

        Return:
            (Node): This is a leaf of the current tree. Particularly the one chosen by traversing the tree according to the pUCT formula.
        '''
        current = node
        while current.is_expanded:
            if current.checkmate_idx is not None:
                return current.children[current.checkmate_idx]
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


    def backpropagate(self, value, node):
        '''
        Backpropagates the value prediction from the neural network along the path taken by the traverse function.
        On the way visit counts are updated.

        Arguments:
            value (float): The value prediction obtained from the neural network.
            node (Node): The leaf node from which backpropagation should be started.
        '''
        while node.parent is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent
