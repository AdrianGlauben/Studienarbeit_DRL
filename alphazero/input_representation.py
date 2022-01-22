import numpy as np
import chess

BOARD_HEIGHT = BOARD_WIDTH = 8
NB_CHANNELS_TOTAL = 46
NB_LAST_MOVES = 8

NORMALIZE_PIECE_NUMBER = 8
NORMALIZE_50_MOVE_RULE = 50

def board_to_planes(board: chess.Board, board_occurence, last_moves, normalize=True):
    """
    Transofroms the chess.Board representation to a plane representation.
    See https://arxiv.org/pdf/1712.01815.pdf, p. 13 for an in depth explanation.

    Feature             | Planes
    ----
    P1 Pieces           | 6 (ordered: Pawn, Knight, Bishop, Rook, Queen, King)

    P2 Pieces           | 6

    Repetitions         | 2

    Last 8 Moves        | 16

    No-Progress-count   | 1

    Colour              | 1

    P1 castling rights  | 2 (one per side of castling)

    P2 castling rights  | 2

    P1 Material Count   | 5 (One plane per piece type)

    P2 Material Count   | 5
    ---- 6 + 6 + 2 + 16 + 1 + 1 + 2 + 2 + 5 + 5 = 46 Planes (= NB_CHANNELS_TOTAL)

    Arguments:
        board (chess.Board): Python-chess board object.
        board_occurence (int): Counts the times this exact board has been seen. For threefold repetititon rule.
        last_moves (list[chess.Move]): A list conataining the last 8 moves. Contains 'None' entries if less then 9 moves played.
        normalize (bool): If True all input planes are normalized to have values between 0 and 1.

    Return:
        planes (np.array): 8 x 8 x NB_CHANNELS_TOTAL plane representation of current board.
    """
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

    #### Material Count to Planes ####
    # Channels: 36-45
    # P1 Material Count     | 5 (One plane per piece type)
    # P2 Material Count     | 5

    for color in colors:
        for piece_type in chess.PIECE_TYPES[:-1]:
            material_count = len(board.pieces(piece_type, color))
            planes[channel, :, :] = material_count / NORMALIZE_PIECE_NUMBER if normalize else material_count
            channel += 1

    return planes


def get_row_col(position, mirror=False):
    """
    Maps the scalar position representation from python-chess to a row, column notation.
    Python-chess counts positions starting from 0 in the bottom left corner of the board,
    to 63 in the top right corner.

    Arguments:
        position (int): Scalar value [0 - 63].
        mirror (bool): Boolean to indicate if the board should be flipped since black is to move.

    Returns:
        row (int): The row of the chess piece.
        col (int): The column of the chess piece.
    """
    # Maps the scalar value position representation
    # from python-chess ([0, 63]) to a row, column notation.
    row = position // 8
    col = position % 8

    if mirror:
        row = 7 - row

    return row, col
