## Code From: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
import gym
import numpy as np
import matplotlib.pyplot as plt

NUM_ACTIONS = 2
NUM_GAMES = 30000

#####################
## Helperfunctions ##
#####################
def maxAction(Q, state):
    values = np.array([Q[state, a] for a in range(NUM_ACTIONS)])
    action = np.argmax(values)
    return action

# Discretize Space
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)

def getState(observation):
    cartX, cartXdot, cartTheta, cartThetadot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetadot)

#####################
##      SARSA      ##
#####################
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # Hyperparameters
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 1.0

    # Construct state space
    state_space = []
    for i in range(len(cartPosSpace) + 1):
        for j in range(len(cartVelSpace) + 1):
            for k in range(len(poleThetaSpace) + 1):
                for l in range(len(poleThetaVelSpace) + 1):
                    state_space.append((i, j, k, l))

    # Initialize Q function
    Q = {}
    for s in state_space:
        for a in range(NUM_ACTIONS):
            Q[s, a] = 0

    total_rewards = np.zeros(NUM_GAMES)
    for i in range(NUM_GAMES):
        if i % 5000 == 0:
            print('starting game', i)

        # Get initial observation
        observation = env.reset()
        s = getState(observation)

        # Choose action (epsilon-greedy)
        rand = np.random.random()
        a = maxAction(Q, s) if rand < (1 - EPSILON) else env.action_space.sample()

        done = False
        epRewards = 0

        # Run episodes
        while not done:
            # Get next state (S')
            observation_, reward, done, info = env.step(a)
            s_ = getState(observation_)

            # Chose next action (A')
            rand = np.random.random()
            a_ = maxAction(Q, s_) if rand < (1 - EPSILON) else env.action_space.sample()

            # Acuumulate rewards
            epRewards += reward

            # Update Q-Table (ON-POLICY)
            Q[s, a] = Q[s, a] + ALPHA * (reward + GAMMA * Q[s_, a_] - Q[s, a])
            s, a = s_, a_

        EPSILON -= 2 / NUM_GAMES if EPSILON > 0 else 0
        total_rewards[i] = epRewards

    # Plotting
    plt.scatter(np.arange(0, NUM_GAMES, 200), total_rewards[1::200])
    plt.show()
