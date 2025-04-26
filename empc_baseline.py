import torch
import torch.nn as nn
import numpy as np

import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

import time

import utils

import argparse

import sys
from sys import exit


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--dyn', type=str, default='kin')
    parser.add_argument('--seed_n', type=int, default=0)
    parser.add_argument('--NS', type=int, default=10)
    parser.add_argument('--NL', type=int, default=20)
    parser.add_argument('--RNN', type=bool, default=False)
    parser.add_argument('--p_sigma_manual', type=float, default=8.0)

    return parser.parse_args()


args = parse_arguments()


dyn_model = args.dyn

NS = args.NS
NL = args.NL

RNN = args.RNN

# Manual progress cost parameter (initial guess)
p_sigma_manual = args.p_sigma_manual

if dyn_model=='kin':
    import utils_kin as utils_car
else:
    import utils_pac as utils_car


# Seed for reproducibility
seed_n = args.seed_n
torch.manual_seed(seed_n)
np.random.seed(seed_n)

# Car axis length
l_r = 0.05
l_f = l_r


if dyn_model=='kin':
    delta_max = 0.40
    lr = 1e-4
    BS = 120
    epochs = 30

else:
    delta_max = 0.50
    lr = 8e-5
    BS = 160
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

rnn_str=''
if RNN:
    rnn_str = 'rnn'


# Model path to save
str_model = f'empc{rnn_str}_{dyn_model}_{NS}_{NL}_{p_sigma_manual}'


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


##########################################################################################
################### M O D E L  &  T R A I N ##############################################
##########################################################################################

model = utils.TCN(NL, 5, 2, 2.0)

if RNN:
    model = utils.RNNModel(NL, 5, 2, 2.0)

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

its_per_epoch = 20

loss_val_best = np.inf

current_time = np.inf

for ep in range(epochs):

    loss_train_avg = 0.

    loss_a_avg = 0.
    loss_delta_avg = 0.

    for it in range(its_per_epoch):
        model.train()
        x0= utils_car.sample_init(BS, true_dx).float()
        
        curv = utils.get_curve_hor_from_x(x0, track_coord, NL, v_max, dt)
        inp = torch.hstack((x0[:,idx_to_NN], curv))
        
        control = model(inp)
        
        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_car.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1),
            np.repeat(p_manual_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control_H)
        
        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)
        
        loss_a = ((u_true_torch[:NS,:, 0] - control[:NS,:, 0])**2).sum(0).mean()
        loss_delta = ((u_true_torch[:NS,:, 1] - control[:NS,:, 1])**2).sum(0).mean()
        
        loss = 0.1*loss_a + loss_delta
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_a_avg = loss_a_avg + 0.01*loss_a.detach().item()/its_per_epoch
        loss_delta_avg = loss_delta_avg + 0.1*loss_delta.detach().item()/its_per_epoch

        loss_train_avg = loss_train_avg + loss.detach().item()/its_per_epoch
        
    
        if it%its_per_epoch==its_per_epoch-1:
            # L O S S   V A LI D A T I O N
            model.eval()
            with torch.no_grad():
        
                # This sampling should bring always the same set of initial states (sn fixed)
                x0_val = utils_car.sample_init(BS_val, true_dx, sn=0).float()
        
                curv_val = utils.get_curve_hor_from_x(x0_val, track_coord, NL, v_max, dt)
                inp_val = torch.hstack((x0_val[:,idx_to_NN], curv_val))
                control_val = model(inp_val).numpy()

                q_manual_casadi_val = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
                p_manual_casadi_val = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
                x_true_val, u_true_val = utils_car.solve_casadi_parallel(
                    np.repeat(q_manual_casadi_val, BS_val, 1),
                    np.repeat(p_manual_casadi_val, BS_val, 1),
                    x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control_H)

                loss_a_val = ((u_true_val[:5, :, 0] - control_val[:5, :, 0])**2).sum(0).mean()
                loss_delta_val = ((u_true_val[:5, :, 1] - control_val[:5, :, 1])**2).sum(0).mean()

                loss_val = 0.1*loss_a_val + loss_delta_val

                print('Train loss:',
                      round(loss_a_avg, 5),
                      round(loss_delta_avg, 5),
                      round(loss_train_avg, 5))

                print('Validation loss:',
                      round(0.1*loss_a_val.item(), 5),
                      round(loss_delta_val.item(), 5),
                      round(loss_val.item(), 5))


            # L A P   P E R F O R M A N C E    (E V A L U A T I O N)
            model.eval()
            with torch.no_grad():

                # This sampling should bring always the same set of initial states
                x0_lap = utils_car.sample_init_test(BS_test, true_dx, sn=0).numpy()

                x0_lap_pred = x0_lap[:,:dx+4]
                x0_lap_manual = x0_lap[:,:dx+4]

                finish_list = np.zeros((BS_test,))
                lap_time_list = np.zeros((BS_test,))

                finished = 0
                crashed = 0
                steps = 0
                max_steps=500

                x0_b_pred = x0_lap_pred[0].copy()

                x_pred_full = x0_b_pred.reshape(-1,1)

                while finished==0 and crashed==0:

                    x0_lap_pred_torch = torch.tensor(x0_b_pred, dtype=torch.float32).unsqueeze(0)
                    curv_lap = utils.get_curve_hor_from_x(x0_lap_pred_torch, track_coord, NL, v_max, dt)
                    inp_lap = torch.hstack((x0_lap_pred_torch[:,idx_to_NN], curv_lap))
                    control_lap = model(inp_lap).numpy()

                    x0_b_pred =  true_dx.forward((torch.tensor(x0_b_pred)).unsqueeze(0), 
                                                 torch.tensor(control_lap[0])).squeeze()[:dx+4].detach().numpy()
                    
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
                    q_current = q_lap_np_casadi
                    p_current = p_lap
                    torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')

                print(f'current lap time: {current_time} \t Pred lap time: {lap_time} \t Finished: {finished}')

                try:
                    print(x_pred_full[0,60], x_pred_full[0,90], x_pred_full[0,120], x_pred_full[0,150], x_pred_full[0,180])
                    print(x_manual_full_H[0,60], x_manual_full_H[0,90], x_manual_full_H[0,120], x_manual_full_H[0,150], x_manual_full_H[0,180])
                except:
                    print('crash')

if current_time > 9999.:
    torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')