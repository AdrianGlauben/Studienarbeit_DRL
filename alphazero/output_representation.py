import chess
import numpy as np
from input_representation import get_row_col

BOARD_HEIGHT = BOARD_WIDTH = 8
NB_CHANNELS_TOTAL = 64
# Mapping from move to the respective channel containing the probabilities for that move.
# Contrary to AlphaZero underpromotions are not taken into account.
# See https://arxiv.org/pdf/1712.01815.pdf, p. 13 for an in depth explanation.
MOVE_TO_CHANNEL_DICT = {
    '10': 0, '20': 1, '30': 2, '40': 3, '50': 4, '60': 5, '70': 6, # North Queen Moves
    '11': 7, '22': 8, '33': 9, '44': 10, '55': 11, '66': 12, '77': 13, # North East Queen Moves
    '01': 14, '02': 15, '03': 16, '04': 17, '05': 18, '06': 19, '07': 20, # East Queen Moves
    '-11': 21, '-22': 22, '-33': 23, '-44': 24, '-55': 25, '-66': 26, '-77': 27, # South East Queen Moves
    '-10': 28, '-20': 29, '-30': 30, '-40': 31, '-50': 32, '-60': 33, '-70': 34, # South Queen Moves
    '-1-1': 35, '-2-2': 36, '-3-3': 37, '-4-4': 38, '-5-5': 39, '-6-6': 40, '-7-7': 41, # South West Queen Moves
    '0-1': 42, '0-2': 43, '0-3': 44, '0-4': 45, '0-5': 46, '0-6': 47, '0-7': 48, # West Queen Moves
    '1-1': 49, '2-2': 50, '3-3': 51, '4-4': 52, '5-5': 53, '6-6': 54, '7-7': 55, # North West Queen Moves
    '21': 56, '12': 57, '-12': 58, '-21': 59, '-2-1': 60, '-1-2': 61, '1-2': 62, '2-1': 63, # Knight Moves
}


def planes_to_move_probabilities(planes, legal_moves, color):
    '''
    Transforms the neural network policy output (plane representation)
    into a probability vector corresponding to the legal moves.

    Arguments:
        planes (np.array):  AlphaZero representation. 64 x 8 x 8 array containing the move probabilities.
                            For information on this representation see https://arxiv.org/pdf/1712.01815.pdf, p. 13.
        legal_moves (list): List of legal moves, excluding underpromotions.
        color (Optional[chess.Color]): The color which has to make a move.

    Return:
        probs (np.array):   Probaility vector corresponding to the legal moves.
                            Rescaled with Sotftmax.
    '''
    probs = np.zeros(len(legal_moves))

    mirror = color == chess.BLACK

    for i, move in enumerate(legal_moves):
        from_row, from_col = get_row_col(move.from_square, mirror=mirror)
        to_row, to_col = get_row_col(move.to_square, mirror=mirror)
        # The channel can be uniquely identified by the row and column distance of the move
        channel = MOVE_TO_CHANNEL_DICT[str(to_row - from_row) + str(to_col - from_col)]
        probs[i] = planes[channel, from_row, from_col]

    return np.exp(probs) / np.sum(np.exp(probs))


def legal_moves_to_flat_planes(legal_moves, probs, color):
    '''
    Transforms the legal moves plus their corresponding probabilities to the planes notation.
    Then the planes are flattened to a one dimensional vector to be the same shape as the NN policy output.

    Arguments:
        legal_moves (list): List of legal moves, excluding underpromotions.
        probs (list): Corresponding probabilities for the legal moves.
        color (Optional[chess.Color]): The color which has to make a move.

    Return:
        planes (np.array): List of size NB_CHANNELS_TOTAL * BOARD_HEIGHT * BOARD_WIDTH
    '''
    planes = np.zeros((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

    mirror = color == chess.BLACK

    for move, prob in zip(legal_moves, probs):
        from_row, from_col = get_row_col(move.from_square, mirror=mirror)
        to_row, to_col = get_row_col(move.to_square, mirror=mirror)
        # The channel can be uniquely identified by the row and column distance of the move
        channel = MOVE_TO_CHANNEL_DICT[str(to_row - from_row) + str(to_col - from_col)]
        planes[channel, from_row, from_col] = prob

    return np.reshape(planes, NB_CHANNELS_TOTAL * BOARD_HEIGHT * BOARD_WIDTH)
