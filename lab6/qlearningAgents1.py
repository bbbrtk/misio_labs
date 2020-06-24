# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from misio.pacman.game import *
from misio.pacman.learningAgents import ReinforcementAgent
# from misio.pacman.featureExtractors import IdentityExtractor
from featureExtractors import IdentityExtractor, SimpleExtractor
from misio.pacman.util import CustomCounter, lookup
import random, math, operator


class PacmanQAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self,
                 epsilon=0.1, # 0.25
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=9,
                 extractor=SimpleExtractor(),
                 **args):
        "You can initialize Q-values here..."

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.featExtractor = extractor
        self.index = 0  # This is always Pacman
        self.weights = CustomCounter()
        self.q_value = CustomCounter()

        ReinforcementAgent.__init__(self, 
                                    epsilon=epsilon,
                                    gamma=gamma,
                                    alpha=alpha,
                                    numTraining=numTraining
                                    )

        self.lastActions = []
        self.lastStates = []
        self.episodesSoFar = 0
        self.score = 0
        self.gamma = gamma


    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    def getQValue(self, state, action):
        return self.q_value[(state,action)]

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # return the maximum Q of state
    def getMaxQ(self, state):
        q_list = []
        for a in state.getLegalPacmanActions(): 
            q = self.getQValue(state,a)
            q_list.append(q) 
        if len(q_list) ==0:
            return 0
        # print(q_list)
        return max(q_list) # wez najlepsze Q sposrod dostpenychs stanow

    # update Q value
    def update(self, last_state, action, state, qmax):
        q = self.getQValue(last_state, action)
        reward = state.getScore()-self.score
        tmp = q + self.alpha*(reward + self.gamma*qmax - q)
        # print(tmp)
        self.q_value[(state, action)] = tmp

    def flipCoin(self, p ):
        r = random.random()
        return r < p

    # return the action maximises Q of state
    def doTheRightThing(self, state):
        legal = self.getLegalActions(state)
        # in the first half of trianing, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        # if self.getEpisodesSoFar()*1.0/self.getNumTraining()<0.5:
        #     if 'Stop' in legal:
        #         legal.remove('Stop')
        #     if len(self.lastActions) > 0:
        #         last_action = self.lastActions[-1]
        #         distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
        #         distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
        #         if math.sqrt(distance0**2 + distance1**2) > 2:
        #             if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
        #                 legal.remove(Directions.REVERSE[last_action])
        
        # tmp = CustomCounter()
        tmp_dist = {}
        for action in legal:
          # tmp[action] = self.getQValue(state, action)
          tmp_dist[action] = self.getQValue(state, action)

        print(tmp_dist)
        # return key with max value
        return max(tmp_dist.items(), key=operator.itemgetter(1))[0]
        # return tmp.argMax()


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal = self.getLegalActions(state)
        # if 'Stop' in legal: legal.remove('Stop')
        # action = None

        reward = state.getScore()-self.score # wez wynik obecny = ogolny - poprzedni wynik
        if len(self.lastStates) > 0: # jesli to nie pierwszy ruch
            last_state = self.lastStates[-1]
            last_action = self.lastActions[-1]
            max_q = self.getMaxQ(state) # wez maksymalne Q
            self.update(last_state, last_action, state, max_q) # update?

        # e-greedy
        # zrob czasami ruch losowy
        if self.flipCoin(self.epsilon):
            action =  random.choice(legal)
        else:
            action =  self.doTheRightThing(state)


        # update attributes
        self.score = state.getScore()
        self.lastStates.append(state)
        self.lastActions.append(action)
        # self.lastState = state
        # self.lastAction = action

        "*** end of CODE HERE ***"
        self.doAction(state, action)

        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def final(self, state):
        "Called at the end of each game."

        # update Q-values
        reward = state.getScore()-self.score
        last_state = self.lastStates[-1]
        last_action = self.lastActions[-1]
        self.update(last_state, last_action, state, 0)

        # reset attributes
        self.score = 0
        self.lastStates = []
        self.lastActions = []

        # decrease epsilon during the trianing
        ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()
        self.setEpsilon(ep*0.1)

        # self.incrementEpisodesSoFar()

        # call the super-class final method
        ReinforcementAgent.final(self, state)

        # did we finish training?
        # if self.getEpisodesSoFar() % 100 == 0:
        #   print ("Completed %s runs of training" % self.getEpisodesSoFar())

        if self.episodesSoFar == self.numTraining:
            # msg = 'Training Done (turning off epsilon and alpha)'
            # print('%s\n%s' % (msg,'-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
