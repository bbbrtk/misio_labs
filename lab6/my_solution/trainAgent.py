#!/usr/bin/env python3

from qlearningAgents import ApproxAgent
from misio.pacman.pacman import LocalPacmanGameRunner
import glob
import numpy as np

def playGame(layouts, numberOfGames, training, optilio, numTraining):
    for layout in layouts:
        print(">>> ------------------------------ >>>")
        print(layout)

        with open("weights.txt") as f:
            weights = [float(x) for x in f.readline().split()]

        # weights = [0,0,0,0]
        # weights = [-27.9914705800362, -3394.38754114026, 334.87185911475, -60.5944964389825]

        # random
        # 237.0996722631916 -4588.853575536811 245.5868697927386 -84.61209079023303 
        # zeros
        # 131.6082538875898 -3662.5998496219695 293.8740546541827 -29.798118394469327 
        # 157.61426275835402 -3713.705833105122 -35.71944232589656 296.6912552309395 
        # again from above
        # -14.760990254051723 -4567.035436781129 372.17484562084405 274.64338077262755 


        agent = ApproxAgent(train=training, optilio=optilio, numTraining=numTraining, weights_values=weights)

        runner = LocalPacmanGameRunner(layout, random_ghosts=False)
        games = []
        for i in range(numberOfGames):
            game = runner.run_game(agent)
            games.append(game)
    
        scores = [game.state.getScore() for game in games]
        results = np.array([game.state.isWin() for game in games])
        print("Avg score >>:     {:0.2f}".format(np.mean(scores)))
        print("Median score:  {:0.2f}".format(np.median(scores)))
        print("Win Rate:      {}/{} {:0.2f}".format(results.sum(), len(results), results.mean()))

if __name__ == "__main__":

    layouts = []
    for layout in  glob.glob("./pacman_layouts/*Classic*"):
        layouts.append(layout[2:-4])
    print(layouts)
        
    numberOfGames = 1
    # for i in range(1000):
    #     playGame(layouts, numberOfGames, training=True, optilio=False, numTraining=numberOfGames)

    print("=== === === === TEST === === === ===")
    playGame(layouts, int(100), training=False, optilio=False, numTraining=0)

# python pacman_run.py --agent qlearningAgents.ApproxAgent -l pacman_layouts/originalClassic -sg -ng -n 1000

