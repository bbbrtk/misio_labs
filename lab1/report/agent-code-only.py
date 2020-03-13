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