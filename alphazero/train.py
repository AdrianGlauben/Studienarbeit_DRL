from net import AlphaZeroResNet
import torch
import chess
from input_representation import board_to_planes
from output_representation import legal_moves_to_flat_planes
from MCTS_search import MCTS
import numpy as np
import time
import pandas as pd

def get_priors(visits, legal_moves, color):
    '''
    Helper function for getting the priors which are used for training the neural network.

    Arguments:
        visits (np.array): Contains the visit counts for the children correspondig to the legal moves.
        legal_moves (list(chess.MOVE)): A list of all possible legal moves.
        color (Optional[chess.COLOR]): Either chess.WHITE or chess.BLACK.

    Return:
        (np.array): A vetor of with 4096 entries containing the probabilities according to the visit count for all legal moves.
                    See alphazero.output_representation for further explanation on this representation.
    '''
    probs = visits / np.sum(visits)
    return legal_moves_to_flat_planes(legal_moves, probs, color)


def self_play(model, expansions_per_move, device):
    '''
    This plays a game in selfplay move, using the given model and device.

    Arguments:
        model (AlphaZeroResNet): The model with which the neural network evaluation should be peformed.
        expansions_per_move (int): The number of nodes that should be expanded before the MCTS decides on a move to play.
        device (torch.device()): The device on which the neural network evaluation should be performed.

    Returns:
        states (list()): All states that were chosen by the search.
        values (np.array): The values correspondig to the states.
                            Obtained by making a list of the outcome of the game containing the values from the perspective from the player at time step t.
        priors (np.array): The priors correspondig to the states. Entries are obtained by the function get_priors().
        outcome (chess.Outcome): The outcome of the self play game.
        move_count (int): The full move count after which the game ended.
    '''
    priors = list()
    states = list()

    board = chess.Board()
    mcts = MCTS(model, expansions_per_move, device)

    # Initialization run
    chosen_child = mcts.search(board)
    state = board_to_planes(chosen_child.parent.board, chosen_child.parent.last_moves)
    states.append(state)
    priors.append(get_priors(chosen_child.parent.child_visits, chosen_child.parent.legal_moves, chosen_child.parent.color))

    for i in range(239):
        if chosen_child.board.is_game_over():
            break
        chosen_child = mcts.search(use_exisitng_node=chosen_child)
        state = board_to_planes(chosen_child.parent.board, chosen_child.parent.last_moves)
        states.append(state)
        priors.append(get_priors(chosen_child.parent.child_visits, chosen_child.parent.legal_moves, chosen_child.parent.color))

    outcome = chosen_child.board.outcome()
    print(outcome)
    if outcome is not None:
        value = 0 if outcome.winner is None else (1 if outcome.winner else -1)
    else:
        value = 0

    print(chosen_child.board)
    print(chosen_child.board.fullmove_number)

    values = np.empty((len(states),), dtype=np.float32)
    values[::2] = value
    values[1::2] = -value

    return states, np.array(values, dtype=np.float32), np.array(priors, dtype=np.float32), outcome, chosen_child.board.fullmove_number


def update_model(states, values, priors, model, optimizer, device):
    '''
    This function updates the given model, using the given optimizer and device.
    Updates are performed using the training data obtained from one self play game.

    Arguments:
        states (list()): The list of states chosen in self play.
        values (np.array): The correspondig values.
        priors (np.array): The correspondig priors.
        model (AlphaZeroResNet): The model that should be trained.
        device (torch.device()): The device the model should be trained on.
    '''
    for state, value, prior in zip(states, values, priors):
        value_prediction, prior_prediction = model(state.to(device))
        value = torch.FloatTensor([value]).to(device)
        prior = torch.FloatTensor(prior).to(device)

        value_loss = torch.pow(value_prediction - value, 2)
        policy_loss = torch.dot(prior, torch.log(prior_prediction[0]))
        loss = 0.2 * value_loss - 0.8 * policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



for i in range(50):
    MODEL_NAME = '44f_256c_18b_trained'
    EXPANSIONS_PER_MOVE = 800
    device = torch.device('cuda:0')

    model = AlphaZeroResNet(44, 256, num_res_blocks=18)
    model.load_state_dict(torch.load('./trained_models/' + MODEL_NAME))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=0.0001, momentum=0.9)

    outcomes = list()
    times = list()
    move_counts = list()

    t0 = time.time()
    states, values, priors, outcome, move_count = self_play(model, EXPANSIONS_PER_MOVE, device)
    t1 = time.time()

    outcomes.append(outcome)
    times.append(t1-t0)
    move_counts.append(move_count)

    print(f'Time: {t1-t0}')

    update_model(states, values, priors, model, optimizer, device)

    df_old = pd.read_csv('./trained_models/training_statistics.csv')
    df_new = pd.DataFrame.from_dict({'outcome': outcomes, 'time': times, 'move_count': move_counts})
    df = pd.concat([df_old, df_new])
    df.to_csv('./trained_models/training_statistics.csv', header=True, index=False)

    torch.save(model.state_dict(), './trained_models/' + MODEL_NAME)

    print(f'No. Games played: {i + 1}')
