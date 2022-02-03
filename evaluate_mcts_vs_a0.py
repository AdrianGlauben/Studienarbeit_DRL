from alphazero.MCTS_search import MCTS as a0_mcts
from mcts.improved_mcts import MCTS as basic_mcts
from alphazero.net import AlphaZeroResNet
from alphazero.MCTS_search import Node as a0_Node
from mcts.improved_mcts import Node as mcts_Node
import torch
import chess
import pandas as pd

def get_corresponding_child(board, tree, player):
    '''
    Helper function to ensure that the MCTS and A0 can make use of their respective trees from past searches.

    Arguments:
        board (chess.Board): The board position for which the correspondig node should be returned.
        tree (Node): The node returned by the previous search.
        player (str): Either 'mcts' or 'a0'. Indicates which type of node should be created if no node corresponding to the board is found in the tree.
    '''
    for child in tree.children.values():
        if board == child.board:
            return child
    if player == 'mcts':
        return mcts_Node(board)
    elif player == 'a0':
        return a0_Node(board)


def evaluate_material(board):
    '''
    Function used for tracking the material balance after every move.

    Arguments:
        board (chess.Board): The board for which the material balance should be obtained.

    Return:
        material_balance (list()): List of length 5, indicating the material balance between white and black.
                                    Ordered: [Pawns, Knights, Bishops, Rooks, Queens].
                                    Positive entries indicate that white has that amount more pieces of the respective type.
                                    Negative entries indicate the same for black.
    '''
    material_balance = list()

    for piece_type in chess.PIECE_TYPES[:-1]:
        material_white = len(board.pieces(piece_type, chess.WHITE))
        material_black = len(board.pieces(piece_type, chess.BLACK))
        material_balance.append(material_white - material_black)

    return material_balance


MODEL_NAME = '44f_256c_18b_run_4_1'
EXPANSIONS_PER_MOVE = 800
device = torch.device('cuda:0')

model = AlphaZeroResNet(44, 256, num_res_blocks=18)
model.load_state_dict(torch.load(r'./alphazero/trained_models/' + MODEL_NAME))
model.to(device)

a0_search = a0_mcts(model, EXPANSIONS_PER_MOVE, device, play_mode=True, c=2)
basic_search = basic_mcts(EXPANSIONS_PER_MOVE)
material_balance = list()

# Initialize all variables by making one move each
board = chess.Board()
mcts_choice = basic_search.search(board)
print(mcts_choice.board)
print('---- Move 1 : MCTS ----')
a0_choice = a0_search.search(mcts_choice.board)
print(a0_choice.board)
print('---- Move 2 : A0 ----')
count = 2

# Play a game by letting MCTS and A0 make moves after eachother
while True:
    current_node = get_corresponding_child(a0_choice.board, mcts_choice, 'mcts')
    mcts_choice = basic_search.search(use_exisitng_node=current_node)
    count += 1
    print(mcts_choice.board)
    material = evaluate_material(mcts_choice.board)
    material_balance.append(material)
    print(material)
    print(f'---- Move {count} : MCTS ----')
    outcome = mcts_choice.board.outcome()
    if outcome is not None:
        break

    current_node = get_corresponding_child(mcts_choice.board, a0_choice, 'a0')
    a0_choice = a0_search.search(use_exisitng_node=current_node)
    count += 1
    print(a0_choice.board)
    material = evaluate_material(a0_choice.board)
    material_balance.append(material)
    print(material)
    print(f'---- Move {count} : A0 ----')
    outcome = a0_choice.board.outcome()
    if outcome is not None:
        break

print(outcome)
# Save the statistics
df = pd.read_csv('./eval_statistics_run_4_1.csv')
df = df.append({'outcome': outcome, 'material_balance': material_balance, 'move_count': int(count/2)}, ignore_index=True)
df.to_csv('./eval_statistics_run_4_1.csv', header=True, index=False)
