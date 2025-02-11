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
    parser.add_argument('--n_Q', type=int, default=1)
    parser.add_argument('--p_sigma_manual', type=float, default=8.0)
    parser.add_argument('--track_name', type=str, default='TEST_TRACK')

    return parser.parse_args()


args = parse_arguments()




##########################################################################################
################### P A R A M E T E R S ##################################################
##########################################################################################

dyn_model = args.dyn

assert dyn_model in ['kin','pac','hard']

if dyn_model=='kin':
    import utils_kin as utils_car
elif dyn_model=='pac':
    import utils_pac as utils_car
else:
    import utils_pac_hardware as utils_car

NS = args.NS # Short horizon Length 
NL = args.NL # Long Horizon Length
n_Q = args.n_Q # Number of learnable parameters through the short horizon

assert n_Q<=NS
assert NS%n_Q==0


# Manual progress cost parameter (initial guess)
p_sigma_manual = args.p_sigma_manual

track_name = args.track_name

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

elif dyn_model=='pac':
    delta_max = 0.50 #0.4 (HW)
    lr = 5e-4
    BS = 120
    epochs = 60

elif dyn_model=='hard':
    l_r = 0.038 
    l_f = 0.052  
    delta_max = 0.40
    lr = 5e-4
    BS = 120
    epochs = 60
    
else:
    print('Not implemented')
    exit()
    

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
    
    
    
elif dyn_model=='pac':
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


elif dyn_model=='hard':
    print('PACEJKA HARDWARE')
    dx=6
    du=2
    lqr_iter = 35
    eps=0.00001
    true_dx = utils_car.FrenetDynBicycleDx(track_coord, params, 'cpu')
    control = utils_car.CasadiControl(track_coord, params)
    Q_manual = np.repeat(np.expand_dims(
        np.array([0, 20.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NS, 0)
    p_manual = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NS, 0)
    
    control_H = utils_car.CasadiControl(track_coord, params_H)
    Q_manual_H = np.repeat(np.expand_dims(
        np.array([0, 20.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NL, 0)
    p_manual_H = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)
    
    idx_to_casadi = [7,1,2,3,4,5,10,11]
    idx_to_NN = [1,2,4]

else:
    print('Not implemented')
    exit()



grad_method = GradMethods.AUTO_DIFF

# Bounds and init for control variables 
u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(NS, BS, 1)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(NS, BS, 1)
u_init= torch.tensor([a_max, 0.0]).unsqueeze(0).unsqueeze(0).repeat(NS, BS, 1)

u_lower_val = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(NS, BS_val, 1)#.to(dev)
u_upper_val = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(NS, BS_val, 1)#.to(dev)
u_init_val = torch.tensor([a_max, 0.0]).unsqueeze(0).unsqueeze(0).repeat(NS, BS_val, 1)#.to(device)




# Get initial lap_time

# This sampling should bring always the same set of initial states
x0_lap = utils_car.sample_init_test(1, true_dx, sn=0).numpy()

x0_lap_manual = x0_lap[:,:dx+4]

finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

finished = 0
crashed = 0
steps = 0
max_steps=600

x0_b_manual = x0_lap_manual[0].copy()
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

finished = 0
crashed = 0
steps = 0
max_steps=1500

x0_b_manual = x0_lap_manual[0].copy()
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

x_current_full = x_manual_full
current_time = lap_time

if finished == 0:
    current_time = np.inf


##########################################################################################
################### M O D E L  &  T R A I N ##############################################
##########################################################################################

model = utils.TCN(NL, n_Q, 5, max_p)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

its_per_epoch = 20

loss_val_best = np.inf

for ep in range(epochs):

    print(f'Epoch {ep}')

    loss_train_avg = 0.

    loss_sig_avg = 0.
    loss_d_avg = 0.
    loss_phi_avg = 0.
    loss_a_avg = 0.
    loss_delta_avg = 0.

    x_star = np.transpose(x_current_full)

    for it in range(its_per_epoch):

        model.train()

        #x0_1 = utils_car.sample_init(BS//2, true_dx).float()
        #x0_2 = utils_car.sample_init_traj_dist(BS//2, true_dx, x_star, 20).float()

        #x0 = torch.vstack((x0_1, x0_2))

        x0= utils_car.sample_init(BS, true_dx).float()
        
        #if dyn_model == 'kin':
        #    x0= utils_car.sample_init(BS, true_dx).float()

        #else:
        #    x0 = utils_car.sample_init_traj_dist(BS, true_dx, x_star, 20).float()
            
        curv = utils.get_curve_hor_from_x(x0, track_coord, NL)
        inp = torch.hstack((x0[:,idx_to_NN], curv))

        q_p_pred = model(inp)

        q, p = utils_car.q_and_p(NS, q_p_pred, Q_manual, p_manual)
        Q = torch.diag_embed(q, offset=0, dim1=-2, dim2=-1)

        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_car.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1),
            np.repeat(p_manual_casadi, BS, 1),
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control_H)

        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)

        # Check samples convergence
        q_manual_casadi_S = torch.permute(q[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
        p_manual_casadi_S = torch.permute(p[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
        x_true_S, u_true_S = utils_car.solve_casadi_parallel(
            q_manual_casadi_S, p_manual_casadi_S,
            x0.detach().numpy()[:,:dx+2], BS, dx, du, control)

        x_true_torch_S = torch.tensor(x_true_S, dtype=torch.float32)
        u_true_torch_S = torch.tensor(u_true_S, dtype=torch.float32)

        pred_x, pred_u, pred_objs = mpc.MPC(
                    true_dx.n_state, true_dx.n_ctrl, NS,
                    u_lower=u_lower, u_upper=u_upper, u_init=u_init,
                    lqr_iter=lqr_iter,
                    verbose=0,
                    exit_unconverged=False,
                    detach_unconverged=False,
                    linesearch_decay=.1,
                    max_linesearch_iter=4,
                    grad_method=grad_method,
                    eps=eps,
                    n_batch=None,
                )(x0, QuadCost(Q, p), true_dx)

        # To use only samples such that the differentiable MPC converged 
        # to approx the same as casadi
        diff_shorts = (
            (u_true_torch_S[:5, :, 0] - pred_u[:5, :, 0])**2 
            + (u_true_torch_S[:5, :, 1] - pred_u[:5, :, 1])**2).mean(0)
        args_conv = torch.argwhere(diff_shorts<0.005)

        loss_dsigma = ((x_true_torch[:5, args_conv, idx_to_casadi[0]] - pred_x[:5, args_conv, idx_to_casadi[0]])**2).sum(0).mean()
        loss_d = ((x_true_torch[:5, args_conv, 1] - pred_x[:5, args_conv, 1])**2).sum(0).mean()
        loss_phi = ((x_true_torch[:5, args_conv, 2] - pred_x[:5, args_conv, 2])**2).sum(0).mean()
        loss_v = ((x_true_torch[:5, args_conv, 3] - pred_x[:5, args_conv, 3])**2).sum(0).mean()

        loss_a = ((u_true_torch[:5, args_conv, 0] - pred_u[:5, args_conv, 0])**2).sum(0).mean()
        loss_delta = ((u_true_torch[:5, args_conv, 1] - pred_u[:5, args_conv, 1])**2).sum(0).mean()

        # The constants below is for normalization purpose, 
        # to avoid giving more emphasis in a specific term
        loss = 100*loss_dsigma + 10*loss_d + 0.1*loss_phi + 0.01*loss_a + 0.1*loss_delta

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()


        loss_sig_avg = loss_sig_avg + 100*loss_dsigma.detach().item()/its_per_epoch
        loss_d_avg = loss_d_avg + 10*loss_d.detach().item()/its_per_epoch
        loss_phi_avg = loss_phi_avg + 0.1*loss_phi.detach().item()/its_per_epoch
        loss_a_avg = loss_a_avg + 0.01*loss_a.detach().item()/its_per_epoch
        loss_delta_avg = loss_delta_avg + 0.1*loss_delta.detach().item()/its_per_epoch

        loss_train_avg = loss_train_avg + loss.detach().item()/its_per_epoch


        if it%its_per_epoch==its_per_epoch-1:
            #d_pen = true_dx.penalty_d(pred_x[:, :, 1].detach())
            #v_pen = true_dx.penalty_v(pred_x[:, :, 3].detach())
            if dyn_model == 'kin':
                print('V max: ', pred_x[:, :, 3].detach().max().item())
            else:
                print('V max: ', pred_x[:, :, 4].detach().max().item())
            print('N useful samples: ', pred_x.detach()[:, args_conv, 5].shape)

            # L O S S   V A LI D A T I O N
            model.eval()
            with torch.no_grad():

                # This sampling should bring always the same set of initial states (sn fixed)
                x0_val = utils_car.sample_init(BS_val, true_dx, sn=0).float()

                curv_val = utils.get_curve_hor_from_x(x0_val, track_coord, NL)
                inp_val = torch.hstack((x0_val[:,idx_to_NN], curv_val))
                q_p_pred_val = model(inp_val)

                q_val, p_val = utils_car.q_and_p(NS, q_p_pred_val, Q_manual, p_manual)
                Q_val = torch.diag_embed(q_val, offset=0, dim1=-2, dim2=-1)

                q_val_np_casadi = torch.permute(q_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                p_val_np_casadi = torch.permute(p_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                x_pred_val, u_pred_val = utils_car.solve_casadi_parallel(
                    q_val_np_casadi, p_val_np_casadi,
                    x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control)

                q_manual_casadi_val = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
                p_manual_casadi_val = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
                x_true_val, u_true_val = utils_car.solve_casadi_parallel(
                    np.repeat(q_manual_casadi_val, BS_val, 1),
                    np.repeat(p_manual_casadi_val, BS_val, 1),
                    x0_val.detach().numpy()[:,:dx+2], BS_val, dx, du, control_H)


                loss_dsigma_val = ((x_true_val[:5, :, idx_to_casadi[0]] - x_pred_val[:5, :, idx_to_casadi[0]])**2).sum(0).mean()
                loss_d_val = ((x_true_val[:5, :, 1] - x_pred_val[:5, :, 1])**2).sum(0).mean()
                loss_phi_val = ((x_true_val[:5, :, 2] - x_pred_val[:5, :, 2])**2).sum(0).mean()
                loss_v_val = ((x_true_val[:5, :, 3] - x_pred_val[:5, :, 3])**2).sum(0).mean()

                loss_a_val = ((u_true_val[:5, :, 0] - u_pred_val[:5, :, 0])**2).sum(0).mean()
                loss_delta_val = ((u_true_val[:5, :, 1] - u_pred_val[:5, :, 1])**2).sum(0).mean()

                loss_val = 100*loss_dsigma_val + 10*loss_d_val + 0.1*loss_phi_val + 0.01*loss_a_val + 0.1*loss_delta_val

                print('Train loss:',
                      round(loss_sig_avg, 5),
                      round(loss_d_avg, 5),
                      round(loss_phi_avg, 5),
                      round(loss_a_avg, 5),
                      round(loss_delta_avg, 5),
                      round(loss_train_avg, 5))

                print('Validation loss:',
                      round(100*loss_dsigma_val.item(), 5),
                      round(10*loss_d_val.item(), 5),
                      round(0.1*loss_phi_val.item(), 5),
                      round(0.01*loss_a_val.item(), 5),
                      round(0.1*loss_delta_val.item(), 5),
                      round(loss_val.item(), 5))
                
                #if loss_val <= loss_val_best:
                #    counter_term = 0
                #    loss_val_best = loss_val
                #    torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')

                #else:
                #    counter_term = counter_term + 1
                #    if counter_term>=4:
                #        sys.exit()



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
                    curv_lap = utils.get_curve_hor_from_x(x0_lap_pred_torch, track_coord, NL)
                    inp_lap = torch.hstack((x0_lap_pred_torch[:,idx_to_NN], curv_lap))
                    q_p_pred_lap = model(inp_lap)
                    q_lap, p_lap = utils_car.q_and_p(NS, q_p_pred_lap, Q_manual, p_manual)

                    q_lap_np_casadi = torch.permute(q_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                    p_lap_np_casadi = torch.permute(p_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()


                    x_b_pred, u_b_pred = utils_car.solve_casadi(
                        q_lap_np_casadi[:,0,:], p_lap_np_casadi[:,0,:],
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
                    q_current = q_lap_np_casadi
                    p_current = p_lap
                    torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')


                print(f'current lap time: {current_time} \t Pred lap time: {lap_time} \t Finished: {finished}')

                try:
                    print(x_pred_full[0,60], x_pred_full[0,90], x_pred_full[0,120], x_pred_full[0,150], x_pred_full[0,180])
                    print(x_manual_full_H[0,60], x_manual_full_H[0,90], x_manual_full_H[0,120], x_manual_full_H[0,150], x_manual_full_H[0,180])
                except:
                    print('crash')


