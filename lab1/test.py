#!usr/bin/python3
from aima3.agents import *
from misio.aima import * 

number_of_no_op = 0 
no_op_limit = 9 
oper = 0 
model = {loc_A: None, loc_B: None} 

def MyAgent(test = False): 
    def program(percept): 
        location, status = percept 
        global number_of_no_op 
        global oper 
        global no_op_limit 
        oper+=1 
        model[location] = status 
        
        if model[loc_A] == model[loc_B] == 'Clean' and number_of_no_op < no_op_limit: 
            number_of_no_op += 1
            return 'NoOp' 
        
        elif status == 'Dirty':
            return 'Suck' 
        elif oper <= no_op_limit:
            number_of_no_op += 1
            return 'NoOp' 
        
        # elif oper % (n-1) and 'Clean':
            # return 'NoOp' 
        elif location == loc_A:
            number_of_no_op = 0
            return 'Right'
        elif location == loc_B:
            number_of_no_op = 0
            return 'Left' 
    
    if not test:
        return program
    else:
        return Agent(program)

no_samples = 100000
n = 1
steps = 50
confidence = .95
def agent_factory_1():
    return MyAgent(True)

def env_factory():
    return TrivialVacuumEnvironmentWithChildren(random_dirt_prob=0.05)

def run_agent(EnvFactory, AgentFactory, n=10, steps=1000):
    envs = [EnvFactory() for i in range(n)]
    return test_agent(AgentFactory, steps, copy.deepcopy(envs))

data = [run_agent(env_factory, agent_factory_1, n, steps) for _ in range(no_samples)]
import numpy as np
print("Expected value {}, standard deviation {}.".format(np.mean(data), np.std(data)))

import scipy.stats as st
print(st.norm.interval(confidence, loc=np.mean(data), scale=st.sem(data)))

import matplotlib.pyplot as plt
plt.hist(data, normed=True, bins=20)
plt.show()