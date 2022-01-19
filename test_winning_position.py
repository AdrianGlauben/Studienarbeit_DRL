import time
import numpy as np
import chess

if __name__ == '__main__':
    import mcts
    import improved_mcts
    fen = 'rn1q1r2/ppp4k/3n2p1/3pB2p/3P1P2/2PB3P/P1P1QP2/2KR2R1 w - - 0 20'
    board = chess.Board(fen=fen)
    print(board)

    ######## Improved MCTS ########
    algo = improved_mcts.MCTS(expansion_budget = 200, rollout_budget = 1)
    t0 = time.time()
    chosen_one = algo.search(board)
    t1 = time.time()

    print(f'---- Time: {t1-t0} ----')
    print(chosen_one)
    print(chosen_one.board)

    ######## MT Parallel MCTS ########
    # algo = mcts.MT_Roullout_MCTS(expansion_budget = 100, rollout_budget = 1, print_budget = True, threads=16)
    # node = mcts.Node(board, root = True)
    #
    # t0 = time.time()
    # best_child = algo.search(node)
    # t1 = time.time()
    #
    # print(f'---- Time: {t1-t0} ----')
    # print(best_child)
    # print(best_child.board)

    ######## MP Parallel MCTS ########
    # algo = mcts.Parallel_Roullout_MCTS(expansion_budget = 100, rollout_budget = 10, print_budget = True)
    # node = mcts.Node(board, root = True)
    #
    # t0 = time.time()
    # best_child = algo.search(node)
    # t1 = time.time()
    #
    # print(f'---- Time: {t1-t0} ----')
    # print(best_child)
    # print(best_child.board)

    ######## Vanilla MCTS ########
    # algo = mcts.Vanilla_MCTS(expansion_budget = 100, rollout_budget = 16, print_budget = True)
    # node = mcts.Node(board, root = True)
    #
    # t0 = time.time()
    # best_child = algo.search(node)
    # t1 = time.time()
    #
    # print(f'---- Time: {t1-t0} ----')
    # print(best_child)
    # print(best_child.board)


    # for child in node.children:
    #     if child.board.peek().uci() == 'e2h5':
    #         print(child)
