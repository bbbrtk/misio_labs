#!/usr/bin/env python3

from qlearningAgents import ApproxAgent
from misio.pacman.pacman import LocalPacmanGameRunner
import glob
import numpy as np

def playGame(layouts, numberOfGames, training, numTraining):
    for layout in layouts:
        print(">>> ------------------------------ >>>")
        print(layout)
        with open("weights.txt") as f:
            weights = [float(x) for x in f.readline().split()]

        agent = ApproxAgent(train=training, optilio=False, numTraining=numTraining, weights_values=weights)

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
    for layout in  glob.glob("./pacman_layouts/*mediumClassic*"):
        layouts.append(layout[2:-4])
        
    numberOfGames = 500
    # playGame(layouts, numberOfGames, True, numberOfGames)
    print("=== === === === TEST === === === ===")
    playGame(layouts, int(numberOfGames/5), False, 0)

# python pacman_run.py --agent qlearningAgents.ApproxAgent -l pacman_layouts/originalClassic -sg -ng -n 1000