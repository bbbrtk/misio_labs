#!/usr/bin/env python3

from qlearningAgents import ApproxAgent
from misio.optilio.pacman import StdIOPacmanRunner


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args, _ = parser.parse_known_args()

    if args.train:
        agent = ApproxAgent(train=True, optilio=False)

    else:
        with open("weights.txt") as f:
            weights = [float(x) for x in f.readline().split()]

        runner = StdIOPacmanRunner()
        games_num = int(input())

        agent = ApproxAgent(train=False, optilio=True, weights=weights)

        for _ in range(games_num):
            runner.run_game(agent)
