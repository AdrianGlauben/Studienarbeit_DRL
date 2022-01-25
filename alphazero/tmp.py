import chess
import input_representation
import numpy as np
from output_representation import planes_to_move_probabilities, legal_moves_to_flat_planes


board = chess.Board()
promotion_board = chess.Board(fen='8/5P2/8/8/4N3/8/8/1k4K1 b - - 0 20')
planes = np.random.rand(64, 8, 8)
# last_moves = [None] * 8
# for i in range(0):
#     move = np.random.choice(list(board.legal_moves))
#     board.push(move)
#     last_moves[i] = move
# planes = input_representation.board_to_planes(board, 0, last_moves)
# print(planes[36:50])

legal_moves = list(promotion_board.legal_moves)
# Filter underpromotions
legal_moves = [m for m in promotion_board.legal_moves if not m.promotion or m.promotion == chess.QUEEN]
print(legal_moves)
print(promotion_board)
probs = planes_to_move_probabilities(planes, legal_moves, promotion_board.turn)
print(probs)

flat_planes = legal_moves_to_flat_planes(legal_moves, probs, promotion_board.turn)
flat_planes = flat_planes.reshape((64, 8, 8))

print(flat_planes[28][7][1])
