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
from misio.pacman.util import CustomCounter, lookup
import random, math

TAKE_WEIGHTS = False
HARDCODED = False
OPTILIO_MODE = False

if OPTILIO_MODE:
    from misio.pacman.featureExtractors import IdentityExtractor, SimpleExtr
else:
    from featureExtractors import IdentityExtractor, SimpleExtractor


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
                 epsilon=0.05,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=900,
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
        self.q_values = CustomCounter()
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
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        q_vals = []
        for action in self.getLegalActions(state):
            q_vals.append(self.getQValue(state, action))
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        else:
            return max(q_vals)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_action = None
        max_q_val = 0
        for action in self.getLegalActions(state):
            q_val = self.getQValue(state, action)
            if q_val > max_q_val or max_action is None:
                max_q_val = q_val
                max_action = action
        return max_action

    def flipCoin(self, p ):
        r = random.random()
        return r < p

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
               
        
        legalActions = self.getLegalActions(state)
        if self.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

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
        first_part = (1 - self.alpha) * self.getQValue(state, action)
        if len(self.getLegalActions(nextState)) == 0:
            sample = reward
        else:
            sample = reward + (self.discount * max([self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
        second_part = self.alpha * sample
        self.q_values[(state, action)] = first_part + second_part


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

class ApproxAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', train=False, optilio=False, weights_values=[0,0,0,0], **args):
        self.featExtractor = lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = CustomCounter()
        self.weight = 0
        self.optilio = optilio
        self.train = train

        if self.train:
            for number, feature in zip(weights_values, self.weights):
                self.weights[feature] = float(number)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q_value = 0
        features = self.featExtractor.getFeatures(state, action)
        counter = 0
        for feature in features:
            q_value += features[feature] * self.weights[feature]
            counter += 1

        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        # features_list = features.sortedKeys()
        counter = 0
        for feature in features:
            difference = 0
            if len(self.getLegalActions(nextState)) == 0:
                difference = reward - self.getQValue(state, action)
            else:
                difference = (reward + self.discount * max([self.getQValue(nextState, nextAction) for nextAction in self.getLegalActions(nextState)])) - self.getQValue(state, action)
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]
            counter += 1

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # # did we finish training?
        if (not self.optilio) and self.train:
            if self.episodesSoFar == self.numTraining:
                with open("weights.txt", "w") as f:
                    string = ""
                    for w in self.weights:
                        string += str(self.weights[w])
                        string += " "
                    print(string, file=f)
                    print(f"weights: {string}")

