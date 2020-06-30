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
from misio.optilio.pacman import StdIOPacmanRunner
from misio.pacman.learningAgents import ReinforcementAgent
from misio.pacman.util import CustomCounter, lookup
import random, math

HARDCODED = False
OPTILIO_MODE = False

if OPTILIO_MODE:
    from misio.pacman.featureExtractors import IdentityExtractor, SimpleExtractor
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
        if not self.optilio:
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
    def __init__(self, extractor='IdentityExtractor', train=False, optilio=False, weights_values=[0,0,0,0,0], **args):
        self.featExtractor = lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = CustomCounter()
        self.weight = 0
        self.optilio = optilio
        self.train = train
        

        if (not self.train) or self.optilio or weights_values != [0,0,0,0,0]:
            for number, feature in zip(weights_values, ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food", "capsules"]):
                self.weights[feature] = float(number)
        
        if not self.optilio:
            # print(self.weights)
            pass

    def getWeights(self):
        return self.weights

    def registerInitialState(self, state):
        self.startEpisode()
        if not self.optilio:
            if self.episodesSoFar == 0:
                # print('Beginning %d episodes of Training' % (self.numTraining))
                pass

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
        # print(self.weights)

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        if not self.optilio:
            NUM_EPS_UPDATE = 100
            if self.episodesSoFar % NUM_EPS_UPDATE == 0:
                print('Reinforcement Learning Status:')
                windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
                if self.episodesSoFar <= self.numTraining:
                    trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                    print('\tCompleted %d out of %d training episodes' % (
                        self.episodesSoFar, self.numTraining))
                    print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
                else:
                    testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                    print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                    print('\tAverage Rewards over testing: %.2f' % testAvg)
                print('\tAverage Rewards for last %d episodes: %.2f' % (
                    NUM_EPS_UPDATE, windowAvg))
                print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
                self.lastWindowAccumRewards = 0.0
                self.episodeStartTime = time.time()

            if self.episodesSoFar == self.numTraining:
                # msg = 'Training Done (turning off epsilon and alpha)'
                # print('%s\n%s' % (msg, '-' * len(msg)))
                pass

        # # did we finish training?
        if (not self.optilio) and self.train:
            if self.episodesSoFar == self.numTraining:
                with open("weights.txt", "w") as f:
                    string = ""
                    for w in self.weights:
                        string += str(self.weights[w])
                        string += " "
                    print(string, file=f)
                    # print(f"FINAL weights: {string}")

if __name__ == "__main__":
    if HARDCODED:
        # -172.99452015340418 -4546.398547755926 316.4952221706852 393.3623709571098 53.103908112869306 
        weights = [45.66863508222667, -4484.887811912104, 237.21305014217407, -746.7090726269469, 93.30690509724428]
    else:
        with open("weights.txt") as f:
            weights = [float(x) for x in f.readline().split()]

    runner = StdIOPacmanRunner()
    games_num = int(input())

    agent = ApproxAgent(train=False, optilio=True, numTraining=0, weights_values=weights)

    for _ in range(games_num):
        runner.run_game(agent)


