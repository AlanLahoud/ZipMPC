import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from mpc.env_dx import frenet_dyn_bicycle, frenet_kin_bicycle  #changed
from mpc.track.src import simple_track_generator, track_functions

import utils

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from tqdm import tqdm

from time import time

import argparse


dyn_model = 'kin'
empc = True

if dyn_model=='kin':
    import utils_kin as utils_car
else:
    import utils_pac as utils_car
    
NS = 10
NL = 25
n_Q = 5

p_sigma_manual = 8.0

track_name = 'TEST_TRACK'

seed_n = 0
torch.manual_seed(seed_n)
np.random.seed(seed_n)

l_r = 0.05
l_f = l_r


if dyn_model=='kin':
    delta_max = 0.40

else:
    delta_max = 0.50

# Curve smoothness
k_curve = 25.

#discretization
dt = 0.03

# Maximum v and a
v_max=1.8
a_max = 1.0

# Clip learnable parameters (TanH, check NN)
max_p = 10

out=5

if empc:
    max_p = 2.0
    out=2


# Model path to save
str_model = f'{dyn_model}_{NS}_{NL}_{n_Q}_{p_sigma_manual}'

if empc:
    str_model = f'empc_{dyn_model}_{NS}_{NL}_{p_sigma_manual}'

# Track parameters
track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68
max_track_width_perc = 0.68
bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width
bound_d = 0.5*max_track_width_perc*track_width
t_track = 0.3
init_track = [0,0,0]

# Parameters as tensors for both Short and Long horizons
params = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NS])
params_H = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NL])


# Generating track
gen = simple_track_generator.trackGenerator(track_density,track_width)
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
    lqr_iter = 20
    eps=0.00001
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
    lqr_iter = 35
    eps=0.00001
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


model = utils.TCN(NL, n_Q, out, max_p)
model.load_state_dict(torch.load(f'./models/model_{str_model}.pkl'))
model.eval()

BS_val = 100


def eval_mse(Q_manual, p_manual, control, model=None, sn=0):

    x0_val = utils_car.sample_init(BS_val, true_dx, sn=sn).float()

    time_nn = 0
    
    if model==None:
        q_S_np_casadi = np.repeat(np.expand_dims((Q_manual[:,idx_to_casadi].T), 1), BS_val, 1)
        p_S_np_casadi = np.repeat(np.expand_dims((p_manual[:,idx_to_casadi].T), 1), BS_val, 1)
    
    else:
        start_time_nn = time()
        curv_val = utils.get_curve_hor_from_x(x0_val, track_coord, NL)
        inp_val = torch.hstack((x0_val[:,idx_to_NN], curv_val))

        if empc:
            control_val = model(inp_val)

        else:    
            q_p_pred_val = model(inp_val)
        
            q_val, p_val = utils_car.q_and_p(NS, q_p_pred_val, Q_manual, p_manual)
            Q_val = torch.diag_embed(q_val, offset=0, dim1=-2, dim2=-1)  
   
            q_S_np_casadi = torch.permute(q_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
            p_S_np_casadi = torch.permute(p_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()

        end_time_nn = time()
        time_nn = end_time_nn - start_time_nn

    time_start_short = time()
    if not empc or model is None:
        x_pred_val, u_pred_val = utils_car.solve_casadi_parallel(
            q_S_np_casadi, p_S_np_casadi,
            x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control)
    
    else:
        u_pred_val = control_val.detach().numpy()


    time_end_short = time()
    

    time_start_long = time()
    q_L_np_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
    p_L_np_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
    x_true_val, u_true_val = utils_car.solve_casadi_parallel(
        np.repeat(q_L_np_casadi, BS_val, 1),
        np.repeat(p_L_np_casadi, BS_val, 1),
        x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control_H)
    time_end_long = time()

    time_short = time_end_short - time_start_short
    time_long = time_end_long - time_start_long
    
    loss_a_val = ((u_true_val[:5, :, 0] - u_pred_val[:5, :, 0])**2).sum(0).mean()
    loss_delta_val = ((u_true_val[:5, :, 1] - u_pred_val[:5, :, 1])**2).sum(0).mean()
    
    loss_val = 0.01*loss_a_val + 0.1*loss_delta_val

    return loss_val, time_nn, time_short, time_long


N_sim = 10
results = np.zeros((N_sim,4))
for i in tqdm(range(N_sim)):
    results[i] = eval_mse(Q_manual, p_manual, control, model=model, sn=i)

N_sim = 10
results_MCSH = np.zeros((N_sim,4))
for i in tqdm(range(N_sim)):
    results_MCSH[i] = eval_mse(Q_manual, p_manual, control, model=None, sn=i)


print(np.sqrt(results).mean(0), np.sqrt(results).std(0))
print(np.sqrt(results_MCSH).mean(0), np.sqrt(results_MCSH).std(0))