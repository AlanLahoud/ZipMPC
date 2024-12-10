import torch
import torch.nn as nn
import numpy as np
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor

import utils_new, utils
import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from torch.optim.lr_scheduler import StepLR

import time

import argparse

import sys
from sys import exit

from matplotlib import pyplot as plt




k_curve = 25.
dt = 0.03


track_density = 300
track_width = 0.5

t_track = 0.3
init_track = [0,0,0]

max_p = 100

gen = simple_track_generator.trackGenerator(track_density,track_width)
track_name = 'TEST_TRACK2'

track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'TEST_TRACK'    : track_functions.test_track,
    'TEST_TRACK2'    : track_functions.test_track2,
    'SNAIL_TRACK'   : track_functions.snail_track
}.get(track_name, track_functions.demo_track)

track_function(gen, t_track, init_track)
gen.populatePointsAndArcLength()
gen.centerTrack()
track_coord = torch.from_numpy(np.vstack(
    [gen.xCoords,
     gen.yCoords,
     gen.arcLength,
     gen.tangentAngle,
     gen.curvature]))

fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=150)
gen.plotPoints(ax)

plt.show()
