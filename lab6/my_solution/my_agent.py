#!/usr/bin/env python3

from qlearningAgents import ApproxAgent
from misio.optilio.pacman import StdIOPacmanRunner

if __name__ == "__main__":
    with open("weights.txt") as f:
        weights = [float(x) for x in f.readline().split()]

    runner = StdIOPacmanRunner()
    games_num = int(input())

    agent = ApproxAgent(train=False, optilio=True, weights_values=weights)

    for _ in range(games_num):
            runner.run_game(agent)
