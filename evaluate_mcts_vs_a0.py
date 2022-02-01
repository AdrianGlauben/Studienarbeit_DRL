from alphazero.MCTS_search import MCTS as a0_mcts
from mcts.improved_mcts import MCTS as basic_mcts
from alphazero.net import AlphaZeroResNet
from alphazero.MCTS_search import Node as a0_Node
from mcts.improved_mcts import Node as mcts_Node
import torch
import chess
import pandas as pd

def get_corresponding_child(board, tree, player):
    for child in tree.children.values():
        if board == child.board:
            return child
    if player == 'mcts':
        return mcts_Node(board)
    elif player == 'a0':
        return a0_Node(board)


def evaluate_material(board):
    material_balance = list()

    for piece_type in chess.PIECE_TYPES[:-1]:
        material_white = len(board.pieces(piece_type, chess.WHITE))
        material_black = len(board.pieces(piece_type, chess.BLACK))
        material_balance.append(material_white - material_black)

    return material_balance


MODEL_NAME = '44f_256c_18b_run_3'
EXPANSIONS_PER_MOVE = 800
device = torch.device('cuda:0')

model = AlphaZeroResNet(44, 256, num_res_blocks=18)
model.load_state_dict(torch.load(r'./alphazero/trained_models/' + MODEL_NAME))
model.to(device)

a0_search = a0_mcts(model, EXPANSIONS_PER_MOVE, device, play_mode=True)
basic_search = basic_mcts(EXPANSIONS_PER_MOVE)
material_balance = list()

board = chess.Board()
mcts_choice = basic_search.search(board)
print(mcts_choice.board)
print('---- Move 1 : MCTS ----')
a0_choice = a0_search.search(mcts_choice.board)
print(a0_choice.board)
print('---- Move 2 : A0 ----')
count = 2

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
df = pd.read_csv('./eval_statistics.csv')
df = df.append({'outcome': outcome, 'material_balance': material_balance, 'move_count': int(count/2)}, ignore_index=True)
df.to_csv('./eval_statistics.csv', header=True, index=False)
