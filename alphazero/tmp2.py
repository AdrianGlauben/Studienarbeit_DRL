import net
import torch
import torch.nn as nn
import numpy as np
import chess
from input_representation import board_to_planes

board = chess.Board()
dummy1 = board_to_planes(board, [None] * 8)
dummy2 = board_to_planes(board, [None] * 8)
dummy = torch.empty((2, 44, 8, 8))
dummy[0] = dummy1
dummy[1] = dummy2

a0 = net.AlphaZeroResNet(44, 128)

value, policy = a0(dummy)
print(value)
np_policy = policy.detach().numpy()
print(np.max(np_policy))
print(policy)
