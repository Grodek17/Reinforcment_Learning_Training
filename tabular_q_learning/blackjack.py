import gymnasium as gym
import numpy as np

#hyperparameters
alpha = 0.3         #learning rate
gamma = 0.99        #"importance of future"
epsilon = 0.9       #random move probability

Q = {}
state = (15,10, False)

if state not in Q:
    Q[state] = np.zeros(2)

Q[state] = [1,2]

print(Q, "initial_state")

state = (20,21, False)

if state not in Q:
    Q[state] = np.zeros(2)
Q[state] = [2,-3]

print(Q, "second_state")



