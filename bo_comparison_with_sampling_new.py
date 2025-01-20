import torch
import torch.nn as nn
import numpy as np
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor

import utils_new
import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from torch.optim.lr_scheduler import StepLR

import time

import utils

import argparse

import sys
from sys import exit


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Set parameters for the program.')

#     parser.add_argument('--mpc_T', type=int, default=10)
#     parser.add_argument('--mpc_H', type=int, default=20)
#     parser.add_argument('--n_Q', type=int, default=5)
#     parser.add_argument('--l_r', type=float, default=0.10)
#     parser.add_argument('--v_max', type=float, default=1.8)
#     parser.add_argument('--delta_max', type=float, default=0.40)
#     parser.add_argument('--p_sigma_manual', type=float, default=3.0)

#     return parser.parse_args()


# # Parsing arguments
# args = parse_arguments()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--dyn', type=str, default='kin')
    parser.add_argument('--seed_n', type=int, default=0)
    parser.add_argument('--NS', type=int, default=5)
    parser.add_argument('--NL', type=int, default=18)
    parser.add_argument('--n_Q', type=int, default=1)
    parser.add_argument('--p_sigma_manual', type=float, default=8.0)

    return parser.parse_args()


args = parse_arguments()

##########################################################################################
################### P A R A M E T E R S ##################################################
##########################################################################################

dyn_model = args.dyn

assert dyn_model in ['kin','pac']

if dyn_model=='kin':
    import utils_kin as utils_car
else:
    import utils_pac as utils_car

NS = args.NS # Short horizon Length 
NL = args.NL # Long Horizon Length
n_Q = args.n_Q # Number of learnable parameters through the short horizon

assert n_Q<=NS
assert NS%n_Q==0

# Manual progress cost parameter (initial guess)
p_sigma_manual = args.p_sigma_manual

# Seed for reproducibility
seed_n= args.seed_n
torch.manual_seed(seed_n)
np.random.seed(seed_n)

# Car axis length
l_r = 0.05
l_f = l_r

if dyn_model=='kin':
    delta_max = 0.40
    lr = 1e-4
    BS = 80
    epochs = 20

else:
    delta_max = 0.50
    lr = 5e-4
    BS = 120
    epochs = 60
    

# Curve smoothness
k_curve = 25.

#discretization
dt = 0.03

# Maximum v and a
v_max=1.8
a_max = 1.0

# Clip learnable parameters (TanH, check NN)
max_p = 10

# Batch size
BS_val = 80
BS_test = 1

# Model path to save
str_model = f'{dyn_model}_{NS}_{NL}_{n_Q}_{p_sigma_manual}'

# Track parameters
track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68
bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width
t_track = 0.3
init_track = [0,0,0]

# Parameters as tensors for both Short and Long horizons
params = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NS])
params_H = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NL])


# Generating track
gen = simple_track_generator.trackGenerator(track_density,track_width)
track_name = 'TEST_TRACK'
track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'TEST_TRACK'    : track_functions.test_track,
    'TEST_TRACK_2'    : track_functions.test_track2,
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

# Setting the learnable model dynamics (short horizon)
# Here, we also set manual cost parameters / initial guess
if dyn_model=='kin':
    print('KINEMATICS')
    dx=4
    du=2
    true_dx = utils_car.FrenetKinBicycleDx(track_coord, params, 'cpu')
    control = utils_car.CasadiControl(track_coord, params)
    Q_manual = np.repeat(np.expand_dims(
        np.array([0.0, 3.0, 1.0, 0.01, 0.01, 0.01, 1, 1, 0.01, 1.0]), 0), NS, 0)
    p_manual = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NS, 0)
    
    control_H = utils_car.CasadiControl(track_coord, params_H)
    Q_manual_H = np.repeat(np.expand_dims(
        np.array([0.0, 3.0, 1.0, 0.01, 0.01, 0.01, 1, 1, 0.01, 1.0]), 0), NL, 0)
    p_manual_H = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)

    idx_to_casadi = [5,1,2,3,8,9]
    idx_to_NN = [1,2,3]
    
    
    
else:
    print('PACEJKA')
    dx=6
    du=2
    true_dx = utils_car.FrenetDynBicycleDx(track_coord, params, 'cpu')
    control = utils_car.CasadiControl(track_coord, params)
    Q_manual = np.repeat(np.expand_dims(
        np.array([0, 50.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NS, 0)
    p_manual = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NS, 0)
    
    control_H = utils_car.CasadiControl(track_coord, params_H)
    Q_manual_H = np.repeat(np.expand_dims(
        np.array([0, 50.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NL, 0)
    p_manual_H = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)
    
    idx_to_casadi = [7,1,2,3,4,5,10,11]
    idx_to_NN = [1,2,4]

# mpc_T = args.mpc_T
# mpc_H = args.mpc_H
# n_Q = args.n_Q

# mpc_L = mpc_T
# #n_Q = mpc_T

# l_r = args.l_r
# v_max = args.v_max
# delta_max = args.delta_max

# p_sigma_manual = args.p_sigma_manual

# load_model = False


# seed_n = 0
# torch.manual_seed(seed_n)
# np.random.seed(seed_n)

# k_curve = 25.
# dt = 0.03

# l_f = l_r

# assert mpc_T%n_Q==0

# a_max = 1.5

# track_density = 300
# track_width = 0.5
# max_track_width_perc_casadi = 0.68

# bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width

# t_track = 0.3
# init_track = [0,0,0]

# max_p = 10

# str_model = f'im_{mpc_T}_{mpc_H}_{n_Q}_{l_r}_{delta_max}_{v_max}_{p_sigma_manual}'

# params = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_T])
# params_H = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_H])

# gen = simple_track_generator.trackGenerator(track_density,track_width)
# track_name = 'DEMO_TRACK'

# track_function = {
#     'DEMO_TRACK'    : track_functions.demo_track,
#     'HARD_TRACK'    : track_functions.hard_track,
#     'LONG_TRACK'    : track_functions.long_track,
#     'LUCERNE_TRACK' : track_functions.lucerne_track,
#     'BERN_TRACK'    : track_functions.bern_track,
#     'INFINITY_TRACK': track_functions.infinity_track,
#     'TEST_TRACK'    : track_functions.test_track,
#     'TEST_TRACK2'    : track_functions.test_track2,
#     'SNAIL_TRACK'   : track_functions.snail_track
# }.get(track_name, track_functions.demo_track)

# track_function(gen, t_track, init_track)
# gen.populatePointsAndArcLength()
# gen.centerTrack()
# track_coord = torch.from_numpy(np.vstack(
#     [gen.xCoords,
#      gen.yCoords,
#      gen.arcLength,
#      gen.tangentAngle,
#      gen.curvature]))

# true_dx = utils_new.FrenetKinBicycleDx(track_coord, params, 'cpu')


# x0 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
# u0 = torch.tensor([0.0, 0.0])


# dx=4
# du=2

# BS = 40

gpr = GaussianProcessRegressor(random_state=0)

# control = utils_new.CasadiControl(track_coord, params)
# Q_manual = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_T, 0)
# p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_T, 0)

# control_H = utils_new.CasadiControl(track_coord, params_H)
# Q_manual_H = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_H, 0)
# p_manual_H = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_H, 0)

# idx_to_casadi = [5,1,2,3,8,9]

# here we should decide how many parameters we would like to learn
learned_param = 2
idx_to_learned_param = [5,1,2,8,9]
bo_bound = 1
bo_step = 5
array1 = np.linspace(-0.4, 0.5, bo_step).tolist()
array2 = np.linspace(-0.8, 0.8, bo_step).tolist()
array3 = np.linspace(-1.2, 1.2, bo_step).tolist()
array4 = np.linspace(-0.4, -0.1, bo_step).tolist()
array5 = np.linspace(-0.4, 0.5, bo_step).tolist()
bo_grid = np.array(list(itertools.product(array1,array2,array3,array4,array5))) #,array3,array4,array5
#bo_grid = list(itertools.product([(np.linspace(-bo_bound, bo_bound, bo_step)).tolist() for _ in range(learned_param)]))
grid_shape = np.shape(bo_grid)
print('grid:', bo_grid)
print('grid shape:', grid_shape)

# now we sample from this grid without sampling any location twice
samples = np.zeros(0)
warm_start = 5

for i in range(warm_start):
    sample = np.random.randint(0,grid_shape[0],1)
    while np.isin(sample,samples):
        sample = np.random.randint(0,grid_shape[0],1)

    samples = np.append(samples,sample,axis=0)

samples = samples.astype(int)

#run lap with new samples
beta = 1.0
bo_iter = 100
p_bo_base = np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0])
p_bo_add = np.zeros(10)
Q_bo = Q_manual


# epochs = 35
# num_patches = 20
# BS_init = 40
# BS_val = 10

# Get initial lap_time

# BS_test = 1

# This sampling should bring always the same set of initial states
# x0_lap = utils_new.sample_init_test(BS_test, true_dx, sn=0).numpy()

# x0_lap_manual = x0_lap[:,:6]

# This sampling should bring always the same set of initial states
x0_lap = utils_car.sample_init_test(1, true_dx, sn=0).numpy()

x0_lap_manual = x0_lap[:,:dx+4]

finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

for b in range(BS_test):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=1500

    x0_b_manual = x0_lap_manual[b].copy()
    x_manual_full_H = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_manual_H[:,idx_to_casadi].T
        p_lap_manual_casadi = p_manual_H[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_car.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control_H)

        #x0_b_manual = x_b_manual[1]
        x0_b_manual = true_dx.forward((torch.tensor(x0_b_manual)).unsqueeze(0), 
                                                  torch.tensor(u_b_manual)[0:1]).squeeze()[:dx+4].detach().numpy()
    
        x_manual_full_H = np.append(x_manual_full_H, x0_b_manual.reshape(-1,1), axis=1)
        #print("x_manual:", x_b_manual[1])

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.04 or x0_b_manual[1]<-bound_d_casadi-0.04 or steps>max_steps:
            crashed=1

        steps = steps+1
        #print("long horizon step:", steps)

    lap_time = dt*steps

    print(f'Manual extended NL = {NL}, lap time: {lap_time}, finished: {finished}')


finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

for b in range(BS_test):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=1500

    x0_b_manual = x0_lap_manual[b].copy()
    x_manual_full = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_manual[:,idx_to_casadi].T
        p_lap_manual_casadi = p_manual[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_car.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control)

        #x0_b_manual = x_b_manual[1]

        x0_b_manual = true_dx.forward((torch.tensor(x0_b_manual)).unsqueeze(0), 
                                                    torch.tensor(u_b_manual)[0:1]).squeeze()[:dx+4].detach().numpy()
    
        x_manual_full = np.append(x_manual_full, x0_b_manual.reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.04 or x0_b_manual[1]<-bound_d_casadi-0.04 or steps>max_steps:
            crashed=1

        steps = steps+1
        #print("short horizon step:", steps)

    lap_time = dt*steps

    print(f'Manual NS = {NS}, lap time: {lap_time}, finished: {finished}')



q_manual_casadi = Q_manual[:,idx_to_casadi].T
p_manual_casadi = p_manual[:,idx_to_casadi].T


# set fastest lap_time and corresponding params
current_time = lap_time
q_manual_casadi = q_manual_casadi
p_current_casadi = p_manual_casadi
x_current_full = x_manual_full

if finished == 0:
    current_time = np.inf

# start here with the warm start 
for b in range(warm_start):

        #x0_1 = utils_car.sample_init(BS//2, true_dx).float()
        #x0_2 = utils_car.sample_init_traj_dist(BS//2, true_dx, x_star, 20).float()

        #x0 = torch.vstack((x0_1, x0_2))
        
        x0= utils_car.sample_init(BS, true_dx).float()

        #if dyn_model == 'kin':
        #    x0= utils_car.sample_init(BS, true_dx).float()

        #else:
        #    x0 = utils_car.sample_init_traj_dist(BS, true_dx, x_star, 20).float()

        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_car.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1),
            np.repeat(p_manual_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control_H)

        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)

        # here I should sample from the BO process
        p_bo_add[idx_to_learned_param] = bo_grid[samples[b]]

        p_bo_app = p_bo_base + p_bo_add

        p_bo = np.repeat(np.expand_dims(p_bo_app, 0), NS, 0)
        
        q_bo_casadi = np.expand_dims((Q_bo[:,idx_to_casadi].T), 1)
        p_bo_casadi = np.expand_dims((p_bo[:,idx_to_casadi].T), 1)
        x_bo, u_bo = utils_car.solve_casadi_parallel(
            np.repeat(q_bo_casadi, BS, 1),
            np.repeat(p_bo_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control)
        
        x_bo_torch = torch.tensor(x_bo, dtype=torch.float32)
        u_bo_torch = torch.tensor(u_bo, dtype=torch.float32)

        loss_dsigma = ((x_true_torch[:5, :, idx_to_casadi[0]] - x_bo_torch[:5, :, idx_to_casadi[0]])**2).sum(0).mean()
        loss_d = ((x_true_torch[:5, :, 1] - x_bo_torch[:5, :, 1])**2).sum(0).mean()
        loss_phi = ((x_true_torch[:5, :, 2] - x_bo_torch[:5, :, 2])**2).sum(0).mean()
        loss_v = ((x_true_torch[:5, :, 3] - x_bo_torch[:5, :, 3])**2).sum(0).mean()

        loss_a = ((u_true_torch[:5, :, 0] - u_bo_torch[:5, :, 0])**2).sum(0).mean()
        loss_delta = ((u_true_torch[:5, :, 1] - u_bo_torch[:5, :, 1])**2).sum(0).mean()

        # The constants below is for normalization purpose, 
        # to avoid giving more emphasis in a specific term
        loss = 100*loss_dsigma + 10*loss_d + 0.1*loss_phi + 0.01*loss_a + 0.1*loss_delta

        loss = loss.reshape(1)

        if b == 0:
            losses = loss
        else:
            losses = np.append(losses,loss,axis=0)


# Fit GP from initial loss objective value combinations

gpr_fit = gpr.fit(bo_grid[samples],losses)
mean_fit, std_fit = gpr.predict(bo_grid, return_std=True)

print("mean fit:", mean_fit)
print("std fit:", std_fit)

# Construct accquisition funtion 
# (beta as tuning parameter -> the higher beta the more we are exploring)
accq_fun = mean_fit - beta*std_fit

############################## MISTAKE - SAMPLING FROM WRONG DIMENSIONS - ONCE GRID ONCE INDEX #############

# find minimum of accquisition function to find next sampling point
sampling_idx = np.argmin(accq_fun).reshape(1)
samples = np.append(samples,sampling_idx,axis=0)

print("sampling_idx:", sampling_idx)
print("ssamples:", samples)

its_per_epoch = 60

for ep in range(epochs):

    # mpc_L = 5 + ep//3
    # mpc_L = int(np.minimum(mpc_L, mpc_T))

    print(f'Epoch {ep}')
    #x_star = np.transpose(x_current_full)

    loss_train_avg = 0.

    loss_sig_avg = 0.
    loss_d_avg = 0.
    loss_phi_avg = 0.
    loss_a_avg = 0.
    loss_delta_avg = 0.

    x_star = np.transpose(x_current_full)

    for it in range(its_per_epoch):

        #x0_1 = utils_car.sample_init(BS//2, true_dx).float()
        #x0_2 = utils_car.sample_init_traj_dist(BS//2, true_dx, x_star, 20).float()

        #x0 = torch.vstack((x0_1, x0_2))

        x0= utils_car.sample_init(BS, true_dx).float()
        
        #if dyn_model == 'kin':
        #    x0= utils_car.sample_init(BS, true_dx).float()

        #else:
        #    x0 = utils_car.sample_init_traj_dist(BS, true_dx, x_star, 20).float()

        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_car.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1),
            np.repeat(p_manual_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control_H)

        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)

        # here I should sample from the BO process
        p_bo_add[idx_to_learned_param] = bo_grid[sampling_idx]

        print(f'BO sampling_idx: {sampling_idx}')
        print(f'BO values: {bo_grid[sampling_idx]}')

        p_bo_app = p_bo_base + p_bo_add

        p_bo = np.repeat(np.expand_dims(p_bo_app, 0), NS, 0)
        
        q_bo_casadi = np.expand_dims((Q_bo[:,idx_to_casadi].T), 1)
        p_bo_casadi = np.expand_dims((p_bo[:,idx_to_casadi].T), 1)
        x_bo, u_bo = utils_car.solve_casadi_parallel(
            np.repeat(q_bo_casadi, BS, 1),
            np.repeat(p_bo_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control)
        
        x_bo_torch = torch.tensor(x_bo, dtype=torch.float32)
        u_bo_torch = torch.tensor(u_bo, dtype=torch.float32)

        loss_dsigma = ((x_true_torch[:5, :, idx_to_casadi[0]] - x_bo_torch[:5, :, idx_to_casadi[0]])**2).sum(0).mean()
        loss_d = ((x_true_torch[:5, :, 1] - x_bo_torch[:5, :, 1])**2).sum(0).mean()
        loss_phi = ((x_true_torch[:5, :, 2] - x_bo_torch[:5, :, 2])**2).sum(0).mean()
        loss_v = ((x_true_torch[:5, :, 3] - x_bo_torch[:5, :, 3])**2).sum(0).mean()

        loss_a = ((u_true_torch[:5, :, 0] - u_bo_torch[:5, :, 0])**2).sum(0).mean()
        loss_delta = ((u_true_torch[:5, :, 1] - u_bo_torch[:5, :, 1])**2).sum(0).mean()

        # The constants below is for normalization purpose, 
        # to avoid giving more emphasis in a specific term
        loss = 100*loss_dsigma + 10*loss_d + 0.1*loss_phi + 0.01*loss_a + 0.1*loss_delta

        loss = loss.reshape(1)
        print(f'BO loss: {loss}')

        losses = np.append(losses,loss,axis=0)

        loss_sig_avg = loss_sig_avg + 100*loss_dsigma.detach().item()/its_per_epoch
        loss_d_avg = loss_d_avg + 10*loss_d.detach().item()/its_per_epoch
        loss_phi_avg = loss_phi_avg + 0.1*loss_phi.detach().item()/its_per_epoch
        loss_a_avg = loss_a_avg + 0.01*loss_a.detach().item()/its_per_epoch
        loss_delta_avg = loss_delta_avg + 0.1*loss_delta.detach().item()/its_per_epoch

        loss_train_avg = loss_train_avg + loss.detach().item()/its_per_epoch

        # Update GP fit with new loss objective value combination
        gpr_fit = gpr.fit(bo_grid[samples],losses)
        mean_fit, std_fit = gpr.predict(bo_grid, return_std=True)

        #print("mean fit:", mean_fit)
        #print("std fit:", std_fit)

        # Accordingly update accquisition function
        accq_fun = mean_fit - beta*std_fit

        #print("accq fun:", accq_fun)

        # find minimum of accquisition function to find next sampling point
        sampling_idx = np.argmin(accq_fun).reshape(1)
        samples = np.append(samples,sampling_idx,axis=0)

        #print("sampling idx:", sampling_idx)

        if it%its_per_epoch==its_per_epoch-1:
            # d_pen = true_dx.penalty_d(x_bo_torch[:, :, 1].detach())
            # v_pen = true_dx.penalty_v(x_bo_torch[:, :, 3].detach())
            if dyn_model == 'kin':
                print('V max: ', x_bo_torch[:, :, 3].detach().max().item())
            else:
                print('V max: ', x_bo_torch[:, :, 4].detach().max().item())

            # L O S S   V A LI D A T I O N
            with torch.no_grad():

                BS_val = 100

                # This sampling should bring always the same set of initial states (sn fixed)
                x0_val = utils_car.sample_init(BS_val, true_dx, sn=0).float()

                q_bo_casadi = np.expand_dims((Q_bo[:,idx_to_casadi].T), 1)
                p_bo_casadi = np.expand_dims((p_bo[:,idx_to_casadi].T), 1)
                x_bo_val, u_bo_val = utils_car.solve_casadi_parallel(
                    np.repeat(q_bo_casadi, BS_val, 1),
                    np.repeat(p_bo_casadi, BS_val, 1),
                    x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control)

                q_manual_casadi_val = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
                p_manual_casadi_val = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
                x_true_val, u_true_val = utils_car.solve_casadi_parallel(
                    np.repeat(q_manual_casadi_val, BS_val, 1),
                    np.repeat(p_manual_casadi_val, BS_val, 1),
                    x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control_H)


                loss_dsigma_val = ((x_true_val[:5, :, idx_to_casadi[0]] - x_bo_val[:5, :, idx_to_casadi[0]])**2).sum(0)
                loss_d_val = ((x_true_val[:5, :, 1] - x_bo_val[:5, :, 1])**2).sum(0)
                loss_phi_val = ((x_true_val[:5, :, 2] - x_bo_val[:5, :, 2])**2).sum(0)
                loss_v_val = ((x_true_val[:5, :, 3] - x_bo_val[:5, :, 3])**2).sum(0)

                loss_a_val = ((u_true_val[:5, :, 0] - u_bo_val[:5, :, 0])**2).sum(0)
                loss_delta_val = ((u_true_val[:5, :, 1] - u_bo_val[:5, :, 1])**2).sum(0)

                # Ideal here would be to scale, but this is fine just to be in the same range
                loss_val = 100*loss_dsigma_val + 10*loss_d_val + 0.1*loss_phi_val + 0.01*loss_a_val + 0.1*loss_delta_val

                loss_val_mean = loss_val.mean()
                loss_val_std = loss_val.std()

                print('mean validation loss', loss_val_mean)
                print('standard deviation validation loss', loss_val_std)

                print('Train loss:',
                      round(loss_sig_avg, 5),
                      round(loss_d_avg, 5),
                      round(loss_phi_avg, 5),
                      round(loss_a_avg, 5),
                      round(loss_delta_avg, 5),
                      round(loss_train_avg, 5))

                # print('Validation loss:',
                #       round(100*loss_dsigma_val.item(), 5),
                #       round(10*loss_d_val.item(), 5),
                #       round(0.1*loss_phi_val.item(), 5),
                #       round(0.01*loss_a_val.item(), 5),
                #       round(0.1*loss_delta_val.item(), 5),
                #       round(loss_val.item(), 5))

            # L A P   P E R F O R M A N C E    (E V A L U A T I O N)
            with torch.no_grad():

                # This sampling should bring always the same set of initial states
                x0_lap = utils_car.sample_init_test(BS_test, true_dx, sn=0).numpy()

                x0_lap_pred = x0_lap[:,:dx+4]
                x0_lap_manual = x0_lap[:,:dx+4]

                finish_list = np.zeros((BS_test,))
                lap_time_list = np.zeros((BS_test,))

                for b in range(BS_test):
                    finished = 0
                    crashed = 0
                    steps = 0
                    max_steps=500

                    x0_b_pred = x0_lap_pred[b].copy()

                    x_pred_full = x0_b_pred.reshape(-1,1)

                    while finished==0 and crashed==0:

                        x0_lap_pred_torch = torch.tensor(x0_b_pred, dtype=torch.float32).unsqueeze(0)
                        
                        q_bo_casadi = np.expand_dims((Q_bo[:,idx_to_casadi].T), 1)
                        p_bo_casadi = np.expand_dims((p_bo[:,idx_to_casadi].T), 1)

                        x_b_pred, u_b_pred = utils_car.solve_casadi(
                            q_bo_casadi[:,0,:], p_bo_casadi[:,0,:],
                            x0_b_pred, dx, du, control)

                        x0_b_pred = true_dx.forward((torch.tensor(x0_b_pred)).unsqueeze(0), 
                                                  torch.tensor(u_b_pred)[0:1]).squeeze()[:dx+4].detach().numpy()
                    
                        x_pred_full = np.append(x_pred_full, x0_b_pred.reshape(-1,1), axis=1)

                        if x0_b_pred[0]>track_coord[2].max().numpy()/2:
                            finished=1

                        if x0_b_pred[1]>bound_d_casadi+0.04 or x0_b_pred[1]<-bound_d_casadi-0.04 or steps>max_steps:
                            crashed=1

                        steps = steps+1

                    lap_time = dt*steps

                    x_current_full = x_pred_full
                    if finished == 1 and lap_time <= current_time:
                        current_time = lap_time
                        q_current = q_bo_casadi
                        p_current = p_bo_casadi


                print(f'current lap time: {current_time} \t Pred lap time: {lap_time} \t Finished: {finished}')

                try:
                    print(x_pred_full[0,60], x_pred_full[0,90], x_pred_full[0,120], x_pred_full[0,150], x_pred_full[0,180])
                    print(x_manual_full_H[0,60], x_manual_full_H[0,90], x_manual_full_H[0,120], x_manual_full_H[0,150], x_manual_full_H[0,180])
                except:
                    print('crash')
