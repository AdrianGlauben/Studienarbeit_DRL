import chess
import numpy as np
import time

def make_random_move(board):
    move = np.random.choice([m for m in board.legal_moves])
    board.push_san(board.san(move))

def make_mcts_move(board):
    node = mcts.Node(board, root=True)
    t0 = time.time()
    best_child = algo.search(node)
    t1 = time.time()
    print(f'Time: {t1-t0}')
    return best_child.board


if __name__ == '__main__':
    import mcts
    result = [0, 0, 0]

    board = chess.Board()
    algo = mcts.Vanilla_MCTS(expansion_budget = 21, rollout_budget = 1, c = 1)
    count = 0

    while not board.is_game_over():
        count += 1
        print(f'Move Count: {count}')
        board = make_mcts_move(board)
        if not board.is_game_over():
            make_random_move(board)

    outcome = board.outcome()
    if outcome.winner:
        result[0] += 1
    elif outcome.winner:
        result[1] += 1
    else:
        result[2] += 1

    print(outcome)
