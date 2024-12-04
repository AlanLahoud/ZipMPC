import torch
import torch.nn as nn
import numpy as np

import utils_new
import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from torch.optim.lr_scheduler import StepLR

import time

import argparse

import sys
from sys import exit


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--mpc_T', type=int, default=9)
    parser.add_argument('--mpc_H', type=int, default=20)
    parser.add_argument('--n_Q', type=int, default=3)
    parser.add_argument('--l_r', type=float, default=0.10)
    parser.add_argument('--v_max', type=float, default=1.8)
    parser.add_argument('--delta_max', type=float, default=0.40)
    parser.add_argument('--p_sigma_manual', type=float, default=3.0)

    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

mpc_T = args.mpc_T
mpc_H = args.mpc_H
n_Q = args.n_Q

mpc_L = 5
#n_Q = mpc_T

l_r = args.l_r
v_max = args.v_max
delta_max = args.delta_max

p_sigma_manual = args.p_sigma_manual

load_model = False


seed_n = 0
torch.manual_seed(seed_n)
np.random.seed(seed_n)

k_curve = 25.
dt = 0.03

l_f = l_r

assert mpc_T%n_Q==0

a_max = 1.5

track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68

bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width

t_track = 0.3
init_track = [0,0,0]

max_p = 10

str_model = f'im_{mpc_T}_{mpc_H}_{n_Q}_{l_r}_{delta_max}_{v_max}_{p_sigma_manual}'

params = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_T])
params_H = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_H])

gen = simple_track_generator.trackGenerator(track_density,track_width)
track_name = 'DEMO_TRACK'

track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'TEST_TRACK'    : track_functions.test_track,
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

true_dx = utils_new.FrenetKinBicycleDx(track_coord, params, 'cpu')


x0 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = torch.tensor([0.0, 0.0])


dx=4
du=2

BS = 40
u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)
eps=0.00001
lqr_iter = 18

grad_method = GradMethods.AUTO_DIFF

model = utils_new.FullLearningNN(max_p)

if load_model==True:
    try:
        model.load_state_dict(torch.load(f'./saved_models/model_{str_model}_0.pkl'))
        print('Model loaded')
    except:
        print('No model found to load')

#opt = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
#opt = torch.optim.RMSprop(model.parameters(), lr=0.0001)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

control = utils_new.CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_T, 0)

control_H = utils_new.CasadiControl(track_coord, params_H)
Q_manual_H = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_H, 0)
p_manual_H = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_H, 0)

idx_to_casadi = [5,1,2,3,8,9]


epochs = 35
num_patches = 20
BS_init = 40
BS_val = 10

# Get initial lap_time

BS_test = 1

# This sampling should bring always the same set of initial states
x0_lap = utils_new.sample_init_test(BS_test, true_dx, sn=0).numpy()

x0_lap_manual = x0_lap[:,:6]

finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

for b in range(BS_test):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=500

    x0_b_manual = x0_lap_manual[b].copy()
    x_manual_full_H = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_manual_H[:,idx_to_casadi].T
        p_lap_manual_casadi = p_manual_H[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_new.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control_H)

        x0_b_manual = x_b_manual[1]
        x_manual_full_H = np.append(x_manual_full_H, x0_b_manual.reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.001 or x0_b_manual[1]<-bound_d_casadi-0.001 or steps>max_steps:
            crashed=1

        steps = steps+1

    lap_time_H = dt*steps

    finish_list[b] = finished
    lap_time_list[b] = lap_time_H

    print(f'Manual extended mpc_H = {mpc_H}, lap time: {lap_time_H}')


finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

for b in range(BS_test):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=500

    x0_b_manual = x0_lap_manual[b].copy()
    x_manual_full = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_manual[:,idx_to_casadi].T
        p_lap_manual_casadi = p_manual[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_new.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control)

        x0_b_manual = x_b_manual[1]
        x_manual_full = np.append(x_manual_full, x0_b_manual.reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.001 or x0_b_manual[1]<-bound_d_casadi-0.001 or steps>max_steps:
            crashed=1

        steps = steps+1

    lap_time = dt*steps

    finish_list[b] = finished
    lap_time_list[b] = lap_time

    print(f'Manual mpc_T = {mpc_T}, lap time: {lap_time}')



q_manual_casadi = Q_manual[:,idx_to_casadi].T
p_manual_casadi = p_manual[:,idx_to_casadi].T


# set fastest lap_time and corresponding params
#if finished == 1:
current_time = lap_time
q_manual_casadi = q_manual_casadi
p_current_casadi = p_manual_casadi
x_current_full = x_manual_full
#else:
#    sys.exit("Manual parameter choice not feasible")


its_per_epoch = 60

for ep in range(epochs):

    mpc_L = 5 + ep//3
    mpc_L = int(np.minimum(mpc_L, mpc_T))

    print(f'Epoch {ep}, Update reference path, mpcL={mpc_L}')
    x_star = np.transpose(x_current_full)

    loss_train_avg = 0.

    loss_sig_avg = 0.
    loss_d_avg = 0.
    loss_phi_avg = 0.
    loss_a_avg = 0.
    loss_delta_avg = 0.

    for it in range(its_per_epoch):

        model.train()

        # u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
        # u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
        # u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)

        #import pdb
        #pdb.set_trace()

        npat = num_patches
        #if ep+2 < npat:
        #    npat = ep + 2

        #x0 = utils_new.sample_init_traj_dist(BS, true_dx, x_star, npat).float()
        #x0 = utils_new.sample_init_traj_dist(BS, true_dx, np.transpose(x_manual_full_H), npat).float()
        x0 = utils_new.sample_init(BS, true_dx).float()

        #x0 = torch.vstack((x0_1, x0_2, x0_3))

        #x0 = torch.vstack((x0_1, x0_3)).float()

        #x0 = torch.vstack((x0_1, x0_2, x0_3)).float()

        #x0 = utils_new.sample_init(BS, true_dx)


        #curv = utils_new.get_curve_hor_from_x(x0, track_coord, mpc_H)
        inp = x0[:,0:4]
        #inp_norm = inp/torch.tensor([0.05,0.05,0.05,1.8])

        pred_u = model(inp)

        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_new.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1),
            np.repeat(p_manual_casadi, BS, 1),
            x0.detach().numpy()[:,:6], BS, dx, du, control_H)

        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)

        # --------------------------------------------------------------------------------------------------
        # this part is changed with respect to our proposed method and the MPC is replaced with an NN
        # ---------------------------------------------------------------------------------------------------

        # and the loss is done with respect to only the first step

        loss_a = ((u_true_torch[0,:, 0] - pred_u[0, :])**2).sum(0).mean()
        print(u_true_torch[0,:, 0])
        print(pred_u[0, :])
        print(loss_a)
        loss_delta = ((u_true_torch[0,:, 1] - pred_u[1, :])**2).sum(0).mean()

        loss = 0.01*loss_a + loss_delta

        print(loss)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_a_avg = loss_a_avg + 0.01*loss_a.detach().item()/its_per_epoch
        loss_delta_avg = loss_delta_avg + 0.1*loss_delta.detach().item()/its_per_epoch

        loss_train_avg = loss_train_avg + loss.detach().item()/its_per_epoch


        if it%its_per_epoch==its_per_epoch-1:

            # L O S S   V A LI D A T I O N
            model.eval()
            with torch.no_grad():

                BS_val = 100

                # This sampling should bring always the same set of initial states
                x0_val = utils_new.sample_init_traj_dist(BS_val, true_dx, np.transpose(x_manual_full_H), npat, sn=0)
                x0_val = x0_val.float()
                #x0_val = utils_new.sample_init(BS_val, true_dx, sn=0)

                inp_val = x0_val[:,0:4]
                #inp_val_norm = inp_val/torch.tensor([0.05,0.05,0.05,1.8])
                u_pred_val = model(inp_val)
                #print(u_pred_val.size())

                q_manual_casadi_val = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
                p_manual_casadi_val = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
                x_true_val, u_true_val = utils_new.solve_casadi_parallel(
                    np.repeat(q_manual_casadi_val, BS_val, 1),
                    np.repeat(p_manual_casadi_val, BS_val, 1),
                    x0_val.detach().numpy()[:,:6], BS_val, dx, du, control_H)


                loss_a_val = ((u_true_val[0, :, 0] - u_pred_val[0, :].numpy())**2).sum(0).mean()
                loss_delta_val = ((u_true_val[0, :, 1] - u_pred_val[1, :].numpy())**2).sum(0).mean()

                # Ideal here would be to scale, but this is fine just to be in the same range
                loss_val =  0.01*loss_a_val + 0.1*loss_delta_val

                print('Train loss:',
                      round(loss_a_avg, 5),
                      round(loss_delta_avg, 5),
                      round(loss_train_avg, 5))

                print('Validation loss:',
                      round(0.01*loss_a_val.item(), 5),
                      round(0.1*loss_delta_val.item(), 5),
                      round(loss_val.item(), 5))

            # L A P   P E R F O R M A N C E    (E V A L U A T I O N)
            model.eval()
            with torch.no_grad():

                #print('LAP PERFORMANCE:')
                BS_test = 1

                # This sampling should bring always the same set of initial states
                x0_lap = utils_new.sample_init_test(BS_test, true_dx, sn=0).numpy()

                x0_lap_pred = x0_lap[:,:6]
                x0_lap_manual = x0_lap[:,:6]

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
                        inp_lap = x0_lap_pred_torch[:,0:4]
                        #inp_lap_norm = inp_lap/torch.tensor([0.05,0.05,0.05,1.8])
                        u_pred_lap = model(inp_lap)

                        # until here: now I need to call the forward model.
                        #print(x0_b_pred)
                        #print(x_pred_full)
                        x0_b_pred = torch.transpose(true_dx.forward(torch.from_numpy(x0_b_pred.reshape(1,-1)),torch.transpose(u_pred_lap,0,1)),0,1)
                        x0_b_pred = x0_b_pred.numpy().reshape(-1,1)
                        x_pred_full = np.append(x_pred_full, x0_b_pred[:6,:], axis=1)
                        x0_b_pred = x0_b_pred[:,0]

                        if x0_b_pred[0]>track_coord[2].max().numpy()/2:
                            finished=1

                        if x0_b_pred[1]>bound_d_casadi+0.001 or x0_b_pred[1]<-bound_d_casadi-0.001 or steps>max_steps:
                            crashed=1

                        steps = steps+1

                    lap_time = dt*steps

                    x_current_full = x_pred_full
                    if finished == 1 and lap_time <= current_time:
                        current_time = lap_time
                        q_current = q_lap_np_casadi
                        p_current = p_lap
                        torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')

                    finish_list[b] = finished
                    lap_time_list[b] = lap_time

                print(f'current lap time: {current_time} \t Pred lap time: {lap_time} \t Finished: {finished}')

                try:
                    print(x_pred_full[0,60], x_pred_full[0,90], x_pred_full[0,120], x_pred_full[0,150], x_pred_full[0,180])
                    print(x_manual_full_H[0,60], x_manual_full_H[0,90], x_manual_full_H[0,120], x_manual_full_H[0,150], x_manual_full_H[0,180])
                except:
                    print('crash')
