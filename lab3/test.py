#!/usr/bin/env python3

from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import RandomAgent, SnakeAgent
from lost_wumpus import MyWumpus
from opt_wmp import OptWmp
import numpy as np
import time

np.set_printoptions(precision=3, suppress=True)
n = 10

# test_locally("tests/2015.in", RandomAgent, n=n)
# test_locally("tests/2016.in", RandomAgent, n=n)

t1 = time.time()
test_locally("tests/2015.in", SnakeAgent, n=n)
# test_locally("tests/2015.in", OptWmp, n=n)
print(time.time() - t1)

t1 = time.time()
# test_locally("tests/2015.in", MyWumpus, n=n)
test_locally("tests/2015.in", MyWumpus, n=n)

print(time.time() - t1)
