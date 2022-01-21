import numpy as np
import chess

BOARD_HEIGHT = BOARD_WIDTH = 8
NB_CHANNELS_TOTAL = 36
NB_LAST_MOVES = 8

NORMALIZE_MOBILITY = 64
NORMALIZE_PIECE_NUMBER = 8
NORMALIZE_50_MOVE_RULE = 50

def board_to_planes(board: chess.Board, board_occurence, last_moves, normalize=True):
    planes = np.zeros((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH))

    # Set the player colors
    p1 = board.turn
    p2 = not board.turn
    colors = [p1, p2]

    # Set the starting channel. Channel will be increased after each added plane.
    channel = 0

    # Set the mirror, to mirror position if black is to move
    mirror = board.turn == chess.BLACK

    #### Pieces to Planes ####
    # 2 Players | 6 Piece types | 1 Plane per Piece type
    # 2 * 6 = 12 Planes --> Channels: 0 - 11
    for color in colors:
        for piece_type in chess.PIECE_TYPES:
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos, mirror=mirror)
                planes[channel, row, col] = 1
            channel += 1

    #### Repetitions to Planes ####
    # Sets how often the position has already occured, for 3 fold repetition Rule.
    # Two channels: Ones in the first channel if one repetition occured,
    #               ones in the second channel if two occured.
    # Channels: 12 - 13
    if board_occurence == 1:
        planes[channel, :, :] = 1
    elif board_occurence == 2:
        planes[channel + 1, :, :] = 1
    channel += 2

    #### Last 8 Moves to Planes ####
    # Two planes per move. One plane for the FROM square and one for the TO square,
    # each represented by a binary mask with.
    # 8 Moves | 2 Planes per Move
    # 8 * 2 = 16 Planes --> Channels: 14 - 29
    for move in last_moves:
        if move:
            frow_row, from_col = get_row_col(move.from_square, mirror=mirror)
            to_row, to_col = get_row_col(move.to_square, mirror=mirror)
            planes[channel, frow_row, from_col] = 1
            planes[channel + 1, to_row, to_col] = 1
        channel += 2

    #### Constants to Planes ####
    # Channels: 30-35
    # Feature               | Planes                        | Channel
    # No-Progress-count     | 1                             | 30
    planes[channel, :, :] = board.halfmove_clock / NORMALIZE_50_MOVE_RULE if normalize else board.halfmove_clock
    channel += 1
    # Colour                | 1                             | 31
    planes[channel, :, :] = 1 if p1 == chess.WHITE else 0
    channel += 1
    # P1 castling rights    | 2 (one per side of castling)  | 32 - 33
    # P2 castling rights    | 2                             | 34 - 35
    for color in colors:
        if board.has_kingside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1

        if board.has_queenside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1

    return planes

def get_row_col(position, mirror=False):
    # Maps the scalar value position representation
    # from python-chess ([0, 63]) to a row, column notation.
    row = position // 8
    col = position % 8

    if mirror:
        row = 7 - row

    return row, col
