import time
import numpy as np
import chess

if __name__ == '__main__':
    import mcts
    fen = 'rn1q1r2/ppp4k/3n2p1/3pB2p/3P1P2/2PB3P/P1P1QP2/2KR2R1 w - - 0 20'
    board = chess.Board(fen=fen)
    print(board)

    algo = mcts.Vanilla_MCTS(expansion_budget = 10000, rollout_budget = 1, c = np.sqrt(2), print_budget = True)
    node = mcts.Node(board, root = True)

    best_child = algo.search(node)

    print(best_child)
    print(best_child.board)

    for child in node.children:
        if child.board.peek().uci() == 'e2h5':
            print(child)
