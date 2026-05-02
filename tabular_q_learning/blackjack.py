import gymnasium as gym
import numpy as np
import random

#constants
NUMBER_OF_EPISODES = 20000

#hyperparameters
alpha = 0.3         #learning rate
gamma = 0.99        #"importance of future"
epsilon = 0.999       #random move probability
Q = {}
ActionNames = {0: "stand (koncze ture)", 1:"hit - dobieram"}



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
#print_Q_table(Q)

episode_reward_list = []

for x in range(NUMBER_OF_EPISODES):
    observation, info = env.reset()
    print(observation, " <- observation")
    print(Q[observation][0], "<- reward for 0 || reward for 1 ->", Q[observation][1])
    episode_end = False
    episode_reward = 0

    while not episode_end:
        print(observation, " <- observation")
        #epsilon-greedy choice selection
        if random.random() > epsilon:
            action = np.argmax(Q[observation])
            print("[DEBUG], argmax chosen for:", observation, " ; ", Q[observation], " ; ", action, " ", ActionNames[action] )
        else:
            action = random.randint(0,1)
            print("[DEBUG], random action chosen: ", action, " ", ActionNames[action] )
        
        epsilon = epsilon * 0.999

        new_observation, reward, terminated, truncated, info = env.step(action)
        print("[DEBUG], observation after action: ", new_observation)

        episode_reward += reward
        print("episode_reward: ", episode_reward)

        #updating Q table
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * (np.max(Q[new_observation]))
            print("[DEBUG], new reward calculated: target(", target, ") = reward(", reward, ") + gamma * np.maxQ(newobs)(", np.max(Q[new_observation]), "new obs: ", Q[new_observation])
        Q[observation][action] = Q[observation][action] + alpha * (target - Q[observation][action])
        print("[DEBUG], updated", observation, " ; ", Q[observation], " ; ", Q[observation][action], " ")

        #ending episode
        if terminated or truncated:
            episode_end = True
            episode_reward_list.append(episode_reward)
           # print("episode reward list = ", episode_reward_list, "lenght of a list: ", len(episode_reward_list), " sum of a list: ", sum(episode_reward_list))
            print("================= EPISODE ", x, " TERMINATED =====================")

        observation = new_observation
        print("average_award = ", sum(episode_reward_list), "<- sum of list  / number of items in the list ", len(episode_reward_list), "=", sum(episode_reward_list)/len(episode_reward_list))
        print("epsilon = ", epsilon)
        print("======================================================================")



       

