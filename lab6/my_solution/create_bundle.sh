#!/usr/bin/env bash

main_file="my_agent.py"
archive="${main_file}".tgz
train="train.sh"

chmod +x ${main_file}
chmod +x ${train}

tar -cvzf ${archive} ${main_file} weights.txt instruction.md qlearningAgents.py featureExtractors.py trainAgent.py pacman_run.py train.sh
