#!/usr/bin/env bash

main_file="optilio.py"
archive="${main_file}".tgz
train="train.sh"

chmod +x ${main_file}
chmod +x ${train}

tar -cvzf ${archive} ${main_file} ddpg.py run_trained_model.py tmp tmpBipedalDrop train.sh
