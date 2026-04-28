# Run `pip install "gymnasium[classic-control]"` for this example.
import random
import gymnasium as gym
import numpy as np

def training():
    numOfRows = 4
    numofCols = 4
    alpha = 0.2                 #how fast new value will update (Q = alfa*newQ), later should descend
    gamma = 0.9                 #
    epsilon = 0.7               #exploration rate - how often we take random route
    QTable = np.zeros(((numOfRows * numofCols), 4))
    print(QTable)

    # Create our training environment - map of the frozen lake
    env = gym.make("FrozenLake-v1", is_slippery=True, success_rate=7.0/8.0, render_mode="human") #human #ansi

    # Reset environment to start a new episode


    for r in range(1000):
        print("training run: ", r)
        observation, info = env.reset()
        episode_over = False
        total_reward = 0
        epsilon = epsilon - ( (epsilon - 0.1)/1000 )   #epsilon gradually lower
            
        while not episode_over:
            observationToUpdate = observation   
            x = random.random()   #random number from 0 to 1, for epsilon
            if x > epsilon:     #initially 30% chance, we go with normal Qmaxxing
                action = np.argmax(QTable[observation]) #numer wektora z największą wartością np: 0,1,2,3
            else:
                action = random.randint(0,3) #random move


            #print(observation, " before step || observationToUpdate: ", observationToUpdate) #debug purpose

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
        # print(observation, "after step || observationToUpdate: ", observationToUpdate)  #debug purpose
        # print("after step variables: observation: ", observation, " reward: ", reward, " terminated: ", terminated) #debug purpose

            #updating the Qboard
            if terminated == True:
                target = reward
            else:
                biggestRewardPos = np.argmax(QTable[observation])
                biggestRewardVal = QTable[observation][biggestRewardPos]
                target = reward + gamma * biggestRewardVal
            
            #updating the table itself
            QTable[observationToUpdate][action] = QTable[observationToUpdate][action] + alpha * (target - QTable[observationToUpdate][action])
            

            total_reward += reward
            episode_over = terminated or truncated

        print(f"Episode finished! Total reward: {total_reward}")
    env.close()

    print(QTable)

    np.save("trained_table.npy", QTable)
    print("table saved in file", " trained_table.npy")

def checkingresults():
    TrainedTable = np.load("trained_table.npy")
    env = gym.make("FrozenLake-v1", is_slippery=True, success_rate=7.0/8.0, render_mode="human") #human #ansi

    for r in range(50):
        print("testing run: ", r)
        observation, info = env.reset()
        episode_over = False
        total_reward = 0
            
        while not episode_over:
            observationToUpdate = observation   
            action = np.argmax(TrainedTable[observation]) #numer wektora z największą wartością np: 0,1,2,3

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_over = terminated or truncated

        print(f"Episode finished! Total reward: {total_reward}")
    env.close()


while(True):
    print("nothing | exit | train(todo) | test(todo)")
    a = input("choose an option:_ ")

    if a == "nothing":
        print("n o t h i n g")
    elif a == "exit":
        break;






'''
### subprogram testing how position coding works in the frozenlake

numrows = 4
numcols = 4

ColumnCode = np.zeros((numrows,numcols))
print ("   ", end="")
for x in range(numrows):
    print(x, "  ", end="")
print("\n")

for i in range(numrows):
    for j in range(numcols):
        ColumnCode[i][j] = i*numcols + j

for i in range(numrows):
    print(i, " ", end="")
    for j in range(numcols):
        print(int(ColumnCode[i][j]), " ", end=" ")
    print("\n")
    '''

'''
#testowanie q table

print(QTable[0][0])      # 4 liczby dla pola (0,0)
print(QTable[2][3])      # 4 liczby dla pola (2,3)

QTable[2][3][1] = 5.6
print(QTable[2][3])      # teraz jedna akcja ma wartość 5

print("chuj")
'''

''' initial run
[[4.38956073e-01 5.04434456e-01 3.71740204e-01 4.14581475e-01]
 [3.33829078e-01 9.49857783e-02 3.87698541e-01 3.72148290e-01]
 [4.08940897e-01 4.41013934e-01 4.16041471e-01 4.13332311e-01]
 [4.31428068e-01 1.79864368e-02 4.16941831e-01 3.92904344e-01]
 [5.08126043e-01 5.67607919e-01 2.07529679e-02 4.23984851e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [7.09655279e-04 4.48314917e-01 6.84680484e-02 3.67321941e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [5.68645369e-01 1.38383748e-01 5.71427760e-01 5.32662071e-01]
 [5.22430335e-01 7.79818567e-01 7.03796567e-01 1.51252373e-01]
 [7.28651934e-01 6.48792273e-01 3.76832043e-04 5.03292642e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.19112158e-03 7.97454439e-01 8.38459896e-01 7.34967638e-01]
 [7.88901335e-01 9.09156508e-01 9.53453325e-01 7.56963972e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
 '''