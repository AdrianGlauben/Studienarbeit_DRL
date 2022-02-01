import net
import torch
import torch.nn as nn
import numpy as np
import chess
from MCTS_search import MCTS
from net import AlphaZeroResNet
import time
import pandas as pd
from ast import literal_eval

# board = chess.Board()
# net = AlphaZeroResNet(44, 128, num_res_blocks=19)
# algo = MCTS(net, expansion_budget=800)
#
# t0 = time.time()
# chosen_child = algo.search(board)
# t1 = time.time()
#
# print(t1 - t0)
# print(chosen_child)
# print(algo.root.child_visits)
# print(algo.root.child_total_values)

df = pd.read_csv('./trained_models/training_statistics_run_4.csv')
df = df.drop('training_data', axis='columns')
print(df)
df.to_csv('./trained_models/training_statistics_run_4.csv', header=True, index=False)
