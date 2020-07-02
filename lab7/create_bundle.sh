#!/usr/bin/env bash

main_file="run_trained_model.py"
archive="${main_file}".tgz
train="train.sh"

chmod +x ${main_file}
chmod +x ${train}

tar -cvzf ${archive} ${main_file} ddpg.py random_solution.py run_locally.py tmp train.sh
