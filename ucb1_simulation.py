## Source:   https://github.com/j2kun/ucb1/blob/master/ucb1.py
import math
import random
from matplotlib import pyplot as plt
from numpy import *
import numpy as np
import sys
import pandas as pd

### UCB Parameters ###
num_bandits = 4
numRounds = 50000
######################


# upperBound: int, int -> float
# the size of the upper confidence bound for ucb1
def upperBound(step, numPlays):
   return math.sqrt(2 * math.log(step + 1) / numPlays)

# ucb1: int, (int, int -> float) -> generator
# perform the ucb1 bandit learning algorithm.  num_bandits is the number of
# actions, indexed from 0. reward is a function (or callable) accepting as
# input the action and producing as output the reward for that action

numPlays_array = []
payoffSums_array = []

def ucb1(num_bandits, reward):
    global numPlays_array
    global payoffSums_array
    payoffSums = [0] * num_bandits
    numPlays = [1] * num_bandits
    ucbs = [0] * num_bandits

    #initialize empirical sums
    for t in range(num_bandits):
        payoffSums[t] = reward(t,t)
        yield t, payoffSums[t], ucbs

    t = num_bandits

    while True:
        ucbs = [payoffSums[i] / numPlays[i] + upperBound(t, numPlays[i]) for i in range(num_bandits)]
        action = max(range(num_bandits), key=lambda i: ucbs[i])

        theReward = reward(action, t)

        numPlays[action] += 1
        numPlays_array = np.append(numPlays_array, numPlays)

        payoffSums[action] += theReward
        payoffSums_array = np.append(payoffSums_array, payoffSums)

        yield action, theReward, ucbs
        t = t + 1


biases = [1.0 / k for k in range(5,5+num_bandits)]
means = [0.5 + b for b in biases]
deltas = [means[0] - x for x in means[1:]]
deltaSum = sum(deltas)
invDeltaSum = sum(1/x for x in deltas)

#bestAction = 0

rewards = lambda choice, t: random.random() + biases[choice]

cumulativeReward = 0
bestActionCumulativeReward = 0

regret_list =[]
regretBound_list=[]
t_list=[]

t = num_bandits

for (choice, reward, ucbs) in ucb1(num_bandits, rewards):
    randomAction = random.randint(0, num_bandits)
    cumulativeReward += reward
    bestActionCumulativeReward += reward if choice == randomAction else rewards(randomAction, t)

    regret = bestActionCumulativeReward - cumulativeReward
    regret_list.append(regret)
    regretBound = 8 * math.log(t + 5) * invDeltaSum + (1 + math.pi*math.pi / 3) * deltaSum
    regretBound_list.append(regretBound)
    t += 1

    #print("regret: %d\tregretBound: %.2f" % (regret, regretBound))
    t_list.append(t)
    if t >= numRounds:
        break

# Reshape to be N by Number of Bandits
numPlays_vert = numPlays_array.reshape((-1, num_bandits))
payoffSums_vert = payoffSums_array.reshape((-1, num_bandits))
true_row_count = len(numPlays_vert)


# Plot regret
n = arange(true_row_count)+1

from pylab import *
subplot(211)
plot(t_list, regret_list, label="Regret")
plot(t_list, regretBound_list, label="UCB")
title('Overall Regret from UCB1 Algorithm')
ylabel('Cumulative Regret')

legend()
show()


# Plot bandit reward evolution
subplot(211)

for k in range(num_bandits):
    plot(n, payoffSums_vert[:, k], label="Arm %d" % k)

title('Rewards per Arm (%i Simulated Events)' % true_row_count)
xlabel("Number of Events")
ylabel("Cumulative Reward")

legend()
show()


# Plot bandit evolution
subplot(211)

for k in range(num_bandits):
    plot(n, numPlays_vert[:, k], label="Arm %d" % k)

title('Simulated Allocations per Arm (%i Simulated Events)' % true_row_count)
xlabel("Number of Events")
ylabel("Allocations")

legend()
show()