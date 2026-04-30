import gymnasium as gym
import numpy as np
import random

#constants
NUMBER_OF_EPISODES = 50

#hyperparameters
alpha = 0.3         #learning rate
gamma = 0.99        #"importance of future"
epsilon = 0.9       #random move probability
Q = {}



'''
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
print(Q[state])
print(np.argmax(Q[state]))
'''
def fill_Q_table(Qtable):
    '''
    ================= filling the Q table =========================
    creating dictionary of empty q values (0,0) for every possible state on the board
    e.g.  (19,7,0):(0.1,2.3) -> when our hand is 19 with no ace, there is higher reward for standing than hitting
    '''
    first_obs = 30 #player cards
    second_obs = 10 # dealer

    for x in range(first_obs):
        for y in range(second_obs):
            state = (x+1, y+1, 0)     #no ace
            Qtable[state] = np.zeros(2)  
    
            state = (x+1, y+1, 1)      #ace
            Qtable[state] = np.zeros(2) 
    
def print_Q_table(Q):
    for keys,values in Q.items():
        print(keys)
        print(values)


env = gym.make('Blackjack-v1', natural=False, sab=False)

fill_Q_table(Q)
print_Q_table(Q)

observation, info = env.reset()
print(observation, " <- observation")

#epsilon-greedy choice selection
if random.random() > epsilon:
    action = np.argmax(Q[observation])
else:
    action = random.randint(0,1)
    
print(Q[observation])

 
       

