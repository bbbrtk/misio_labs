state = '[ 0.21653779 -0.97627424  0.55878125]'
state2 = '[-0.39177177 -0.92006243 -3.93427678]'


observation = [[float(i) for i in state[1:-1].split(' ') if len(i) > 0 ]]
print(observation)
observation = [[float(i) for i in state2[1:-1].split(' ') if len(i) > 0 ]]
print(observation)
