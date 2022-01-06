import time
import numpy as np
import chess

if __name__ == '__main__':
    import mcts
    #### Parallel MCTS ####
    algo = mcts.Parallel_MCTS(expansion_budget = 60, rollout_budget = 2, c = 1, no_of_cpus = 8)
    root = mcts.Node(chess.Board(), root = True)

    t0 = time.time()
    best_child = algo.search(root)
    t1 = time.time()

    print('--- Parallel ----')
    print(f'Time: {t1 - t0}')
    print(root)
    print('')

    # #### Vanilla MCTS ####
    # algo = mcts.Parallel_MCTS(expansion_budget = 20, rollout_budget = 200, c = 1, no_of_cpus = 8)
    # root = mcts.Node(chess.Board(), root = True)
    #
    # t0 = time.time()
    # best_child = algo.search(root)
    # t1 = time.time()
    #
    # print('--- Vanilla ----')
    # print(f'Time: {t1 - t0}')
    # print(root)
