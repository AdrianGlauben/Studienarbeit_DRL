import time
import numpy as np
import chess

if __name__ == '__main__':
    import mcts
    fen = 'rn1q1r2/ppp2n1k/6R1/3pB2p/3P1P2/2PB3P/P1P1QP2/2KR4 w - - 0 20'
    board = chess.Board(fen=fen)

    algo = mcts.Vanilla_MCTS(expansion_budget = 1000, rollout_budget = 10, c = np.sqrt(2), print_budget = True)
    node = mcts.Node(board, root = True)

    best_child = algo.search(node)
    print(board)
    print(best_child)
    print(best_child.board)
