import improved_mcts
import chess
import numpy as np
import time

board = chess.Board()
algo = improved_mcts.MCTS_TT(expansion_budget = 5000)

t0 = time.time()
chosen = algo.search(board)
t1 = time.time()

print(f'----- Time: {t1-t0} -----')
print(chosen)
print(chosen.parent.child_visits)
print(algo.root)
