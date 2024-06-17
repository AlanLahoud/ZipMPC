import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import numpy.random as npr

import matplotlib
from matplotlib import pyplot as plt

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx
from mpc import casadi_control
#from mpc.dynamics import NNDynamics
#import mpc.util as eutil
from mpc.env_dx import frenet_dyn_bicycle, frenet_kin_bicycle  #changed
from mpc.track.src import simple_track_generator, track_functions

import time
import os
import shutil
import pickle as pkl
import collections

import argparse

from time import time

import utils




# Parameters
device = 'cpu' #to do
k_curve = 100.

mpc_T = 35
H_curve = 70

n_batch = 16

l_r = 0.2
l_f = 0.2

v_max = 2.5
a_max = 3.
delta_max = 0.6

dt = 0.04

# not using
ac_max = (0.7*v_max)**2 * delta_max / (l_r+l_f)

track_density = 300
track_width = 0.5



params = torch.tensor([l_r, l_f, track_width, v_max, ac_max, dt, a_max, delta_max, k_curve])





# Let's try to create a track 
gen = simple_track_generator.trackGenerator(track_density,track_width)
track_name = 'LONG_TRACK'

t = 0.3
init = [0,0,0]

track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'SNAIL_TRACK'   : track_functions.snail_track
}.get(track_name, track_functions.demo_track)
    
track_function(gen, t, init)
    
gen.populatePointsAndArcLength()
gen.centerTrack()

track_coord = torch.from_numpy(np.vstack([gen.xCoords, gen.yCoords, gen.arcLength, gen.tangentAngle, gen.curvature]))





true_dx = frenet_kin_bicycle.FrenetKinBicycleDx(track_coord, params, device)
true_sim_dx = frenet_kin_bicycle.FrenetKinBicycleDx(track_coord, params, device)
#true_sim_dx = frenet_dyn_bicycle.FrenetDynBicycleDx(track_coord, params)

u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1).to(device)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1).to(device)
u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1).to(device)

n_state = true_dx.n_state
n_ctrl = true_dx.n_ctrl

eps = .1
lqr_iter = 30
grad_method = GradMethods.AUTO_DIFF




env_params = true_dx.params
env_params_sim = true_sim_dx.params

track_coord = track_coord.to(device)

dx = true_dx.__class__(track_coord,env_params,device)
dx_sim = true_sim_dx.__class__(track_coord,env_params_sim,device)

q_penalty = .0001*torch.ones(2).to(device)
p_penalty = torch.ones(2).to(device)

model = utils.NN(H_curve, 3, 8).to(device)
#model.load_state_dict(torch.load('model.pkl'))
opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
q_penalty_batch = q_penalty.unsqueeze(0).repeat(n_batch,1)
p_penalty_batch = p_penalty.unsqueeze(0).repeat(n_batch,1)




for i in range(800):
    
    start_time = time()
    
    x_init_sim = utils.sample_xinit(n_batch, track_width, v_max).to(device)
    
    #x_init_train = x_init_sim[:,[0,1,2,4,6,7,8,9]]
    x_init_train = x_init_sim
    
    Q_batch, p_batch = utils.inference_params(
        x_init_train, track_coord, H_curve, model, 
        q_penalty_batch, p_penalty_batch, 
        n_batch, mpc_T)
           
    progress_loss, d_loss = utils.get_loss_progress_new(x_init_train, x_init_sim, 
                                      dx, dx_sim, 
                                      Q_batch, p_batch, 
                                      mpc_T, H_curve)
    
    total_loss = progress_loss
    
    opt.zero_grad()
    total_loss.backward()
    opt.step()
    
    end_time = time()
    
    print(f'Batch: {i} , Prog. with (mpc_T, H_curve) = ({mpc_T} , {H_curve}): ', 
          -round(progress_loss.item(), 4),
          '\t Time: ', round(end_time-start_time, 4)
         )
    
    if i%20 == 0:
        torch.save(model.state_dict(), f'model_{i}.pkl')
    
    # It would be nice to add a validation step here 