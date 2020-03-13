#!usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from aima3.agents import *
from misio.aima import * 

current_position = random.choice([0,1])
loc = [0, 0]
iter, ac = -1, -1
flag = False

def MyAgent(): 
    def program(percept): 
        global current_position, loc, iter, ac, flag 
        location, status = percept 
        loc[location[0]] = (1 if status == 'Dirty' else 0) # status
        iter += 1
        ac += 1

        if loc[current_position] == 1:
            loc[current_position] = 0
            flag = True
            return 'Suck'
        elif (ac == 0) or (ac == 1 and flag):
            if current_position == 0: return 'Right'
            else: return 'Left'
        elif sum(loc) == 0 and iter < 8: 
            return 'NoOp'
        elif loc[current_position] == 0 and current_position == 0:
            iter = 0
            current_position = 1
            return 'Right'
        elif loc[current_position] == 0 and current_position == 1:
            iter = 0
            current_position = 0 
            return 'Left'
    
    return Agent(program)

# params
no_samples = 50000
n = 1
steps = 50
confidence = .95

# default chart
def agent_factory_1():
    return MyAgent()

def env_factory():
    return TrivialVacuumEnvironmentWithChildren(random_dirt_prob=0.05)

def run_agent(EnvFactory, AgentFactory, n=10, steps=1000):
    envs = [EnvFactory() for i in range(n)]
    return test_agent(AgentFactory, steps, copy.deepcopy(envs))

data = [run_agent(env_factory, agent_factory_1, n, steps) for _ in range(no_samples)]

print(f"Exp. val. {np.mean(data)} - std. dev. {np.std(data)}")
print(st.norm.interval(confidence, loc=np.mean(data), scale=st.sem(data)))

plt.style.use("seaborn-deep")
plt.hist(data, density=True, bins=20)
plt.show()