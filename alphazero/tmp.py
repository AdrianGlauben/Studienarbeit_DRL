import chess
import input_representation
import numpy as np


board = chess.Board()
last_moves = [None] * 8
for i in range(7):
    move = np.random.choice(list(board.legal_moves))
    board.push(move)
    last_moves[i] = move
planes = input_representation.board_to_planes(board, 2, last_moves)
print(planes.shape)
