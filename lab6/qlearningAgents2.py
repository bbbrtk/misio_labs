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
import random, math


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
                 epsilon=0.25,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=0,
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
        self.lastAction = None
        ReinforcementAgent.__init__(self, 
                                    epsilon=epsilon,
                                    gamma=gamma,
                                    alpha=alpha,
                                    numTraining=numTraining
                                    )

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()

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
        legalActions = self.getLegalActions(state)
        if 'Stop' in legalActions: legalActions.remove('Stop')
        action = None
                
        
        "*** YOUR CODE HERE ***"

        action = random.choice(legalActions)

    

        if self.lastAction:
          features = self.featExtractor.getFeatures(state, self.lastAction)
          print(features)
          print(state.getScore())

          foodDirX = features["food-location"][0] - features["pacman"][0]
          foodDirY = features["food-location"][1] - features["pacman"][1]
          print (foodDirX, foodDirY)
          possibleActions = []
          if foodDirX < 0 and 'West' in legalActions: possibleActions.append('West')
          if foodDirX > 0 and 'East' in legalActions: possibleActions.append('East')
          if foodDirY < 0 and 'North' in legalActions: possibleActions.append('North')
          if foodDirX > 0 and 'South' in legalActions: possibleActions.append('South')

          if len(possibleActions) > 0: action = random.choice(possibleActions)
          else: action = random.choice(legalActions)

          features = self.featExtractor.getFeatures(state, action)
          
          if features["#-of-ghosts-1-step-away"] > 0:
                print("aa!!!")
                print(action)
                legalActions.remove(action)
                action = random.choice(legalActions)
                print(legalActions)

        
        self.lastAction = action

        "*** end of CODE HERE ***"
        self.doAction(state, action)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # run after getAction
        # raise NotImplementedError()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
