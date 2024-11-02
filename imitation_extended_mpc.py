import torch
import torch.nn as nn
import numpy as np

import utils_new
import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

import time

import argparse

from sys import exit

def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--mpc_T', type=int, default=10)
    parser.add_argument('--mpc_H', type=int, default=30)
    parser.add_argument('--n_Q', type=int, default=5)
    parser.add_argument('--l_r', type=float, default=0.10)
    parser.add_argument('--v_max', type=float, default=1.8)
    parser.add_argument('--delta_max', type=float, default=0.43)
    parser.add_argument('--p_sigma_manual', type=float, default=0.2)
    
    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

mpc_T = args.mpc_T
mpc_H = args.mpc_H
n_Q = args.n_Q

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

a_max = 2.0

track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68

bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width

t_track = 0.3
init_track = [0,0,0]

max_p = 100

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

BS = 64
u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)
eps=0.001
lqr_iter = 40

grad_method = GradMethods.AUTO_DIFF

model = utils_new.ImprovedNN(mpc_H, n_Q, 9, max_p)

if load_model==True:
    try:
        model.load_state_dict(torch.load(f'./saved_models/model_{str_model}.pkl'))
        print('Model loaded')
    except:
        print('No model found to load')
        
opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#opt = torch.optim.RMSprop(model.parameters(), lr=0.0001)

control = utils_new.CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_T, 0)

control_H = utils_new.CasadiControl(track_coord, params_H)
Q_manual_H = np.repeat(np.expand_dims(np.array([0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]), 0), mpc_H, 0)
p_manual_H = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_H, 0)

idx_to_casadi = [5,1,2,3,8,9]


epochs = 25
num_patches = 10
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
if finished == 1:
    current_time = lap_time
    q_manual_casadi = q_manual_casadi
    p_current_casadi = p_manual_casadi
    x_current_full = x_manual_full
else:
    sys.exit("Manual parameter choice not feasible")


flag_finish_training = 0
flag_finish_training_iter = 0

for ep in range(epochs):

    print(f'Epoch {ep}, Update reference path')
    x_star = np.transpose(x_current_full)

    # We are saving three models
    if flag_finish_training_iter >= 2:
        break

    loss_train_avg = 0.
    for it in range(60):

        model.train()
        
        u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
        u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
        u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)

        #import pdb
        #pdb.set_trace()

        npat = num_patches
        #if ep+2 < npat:
        #    npat = ep + 2
        
        x0_1 = utils_new.sample_init_traj_dist(BS//3, true_dx, x_star, npat)
        x0_2 = utils_new.sample_init_traj_dist(BS//3, true_dx, np.transpose(x_manual_full_H), npat)
        x0_3 = utils_new.sample_init(BS - BS//3 - BS//3, true_dx)

        x0 = torch.vstack((x0_1, x0_2, x0_3))
        
        #x0 = utils_new.sample_init(BS, true_dx)  
        
        x0_diff = x0.clone().float()

        curv = utils_new.get_curve_hor_from_x(x0_diff, track_coord, mpc_H)
        inp = torch.hstack((x0_diff[:,1:4], curv))
        q_p_pred = model(inp)

        q, p = utils_new.q_and_p(mpc_T, q_p_pred, Q_manual, p_manual)
        Q = torch.diag_embed(q, offset=0, dim1=-2, dim2=-1)
        
        q_manual_casadi = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
        p_manual_casadi = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
        x_true, u_true = utils_new.solve_casadi_parallel(
            np.repeat(q_manual_casadi, BS, 1), 
            np.repeat(p_manual_casadi, BS, 1), 
            x0_diff.detach().numpy()[:,:6], BS, dx, du, control_H) 

        x_true_torch = torch.tensor(x_true, dtype=torch.float32)
        u_true_torch = torch.tensor(u_true, dtype=torch.float32)


        # T E S T  MPC SHORT
        q_manual_casadi_S = torch.permute(q[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
        p_manual_casadi_S = torch.permute(p[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
        x_true_S, u_true_S = utils_new.solve_casadi_parallel(
            q_manual_casadi_S, p_manual_casadi_S, 
            x0_diff.detach().numpy()[:,:6], BS, dx, du, control) 

        x_true_torch_S = torch.tensor(x_true_S, dtype=torch.float32)
        u_true_torch_S = torch.tensor(u_true_S, dtype=torch.float32)      
            
        pred_x, pred_u, pred_objs = mpc.MPC(
                    true_dx.n_state, true_dx.n_ctrl, mpc_T,
                    u_lower=u_lower, u_upper=u_upper, u_init=u_init,
                    lqr_iter=lqr_iter,
                    verbose=0,
                    exit_unconverged=False,
                    detach_unconverged=False,
                    linesearch_decay=.3,
                    max_linesearch_iter=4,
                    grad_method=grad_method,
                    eps=eps,
                    n_batch=None,
                )(x0_diff, QuadCost(Q, p), true_dx)
        
        loss_dsigma = (x_true_torch[:mpc_T, :, 5] - pred_x[:, :, 5])**2
        loss_d = (x_true_torch[:mpc_T, :, 1] - pred_x[:, :, 1])**2
        loss_phi = (x_true_torch[:mpc_T, :, 2] - pred_x[:, :, 2])**2
        loss_v = (x_true_torch[:mpc_T, :, 3] - pred_x[:, :, 3])**2
        
        loss_a = (u_true_torch[:mpc_T, :, 0] - pred_u[:, :, 0])**2
        loss_delta = (u_true_torch[:mpc_T, :, 1] - pred_u[:, :, 1])**2

        diff_shorts = ((x_true_torch_S[:, :, 2] - pred_x[:, :, 2])**2).sum(0)
        args_conv = torch.argwhere(diff_shorts<0.00001)
        #print(diff_sigs)

        # Ideal here would be to scale
        loss = 10*loss_dsigma[:,args_conv].sum(0).mean() + 10*loss_d[:,args_conv].sum(0).mean() \
        + loss_v[:,args_conv].sum(0).mean() + 0.01*loss_a[:,args_conv].sum(0).mean() + 0.1*loss_delta[:,args_conv].sum(0).mean()

        loss_train_avg = loss_train_avg + loss.detach().item()/60.
        
        #loss = 0.1*loss_a[:,args_conv].mean() + 0.1*loss_delta[:,args_conv].mean()
        

        #if diff_sigs> 0.001:
        #    import pdb
        #    pdb.set_trace()
        
        if it%20==0:
            #import pdb
            #pdb.set_trace()
            d_pen = true_dx.penalty_d(pred_x[:, :, 1])
            v_pen = true_dx.penalty_v(pred_x[:, :, 3])
            #print(f'd_pen: {d_pen.sum(0).mean().item()} \t v_pen: {v_pen.sum(0).mean().item()}')
            #print(pred_x[:, :, 3].max().item())
            #print(pred_x[:, :, 1].max().item())
            #print(p.mean(0).mean(0))

        
        opt.zero_grad()
        loss.backward()
        opt.step()

        
        if it%20==0:
            # L O S S   V A LI D A T I O N
            model.eval()
            with torch.no_grad():
               
                BS_val = 32

                u_lower_val = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS_val, 1)#.to(dev)
                u_upper_val = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS_val, 1)#.to(dev)
                u_init_val = torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS_val, 1)#.to(device)
    
                # This sampling should bring always the same set of initial states
                x0_val = utils_new.sample_init_traj_dist(BS_val, true_dx, np.transpose(x_manual_full_H), npat, sn=0)
                x0_val = x0_val.float()
                #x0_val = utils_new.sample_init(BS_val, true_dx, sn=0)

                curv_val = utils_new.get_curve_hor_from_x(x0_val, track_coord, mpc_H)
                inp_val = torch.hstack((x0_val[:,1:4], curv_val))
                q_p_pred_val = model(inp_val)
        
                q_val, p_val = utils_new.q_and_p(mpc_T, q_p_pred_val, Q_manual, p_manual)
                Q_val = torch.diag_embed(q_val, offset=0, dim1=-2, dim2=-1)
                
                q_val_np_casadi = torch.permute(q_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                p_val_np_casadi = torch.permute(p_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                x_pred_val, u_pred_val = utils_new.solve_casadi_parallel(
                    q_val_np_casadi, p_val_np_casadi, 
                    x0_val.detach().numpy()[:,:6], BS_val, dx, du, control) 

                q_manual_casadi_val = np.expand_dims((Q_manual_H[:,idx_to_casadi].T), 1)
                p_manual_casadi_val = np.expand_dims((p_manual_H[:,idx_to_casadi].T), 1)
                x_true_val, u_true_val = utils_new.solve_casadi_parallel(
                    np.repeat(q_manual_casadi_val, BS_val, 1), 
                    np.repeat(p_manual_casadi_val, BS_val, 1), 
                    x0_val.detach().numpy()[:,:6], BS_val, dx, du, control_H) 

                
                loss_dsigma_val = (x_true_val[:mpc_T, :, 5] - x_pred_val[:, :, 5])**2
                loss_d_val = (x_true_val[:mpc_T, :, 1] - x_pred_val[:, :, 1])**2
                loss_phi_val = (x_true_val[:mpc_T, :, 2] - x_pred_val[:, :, 2])**2
                loss_v_val = (x_true_val[:mpc_T, :, 3] - x_pred_val[:, :, 3])**2
                
                loss_a_val = (u_true_val[:mpc_T, :, 0] - u_pred_val[:, :, 0])**2
                loss_delta_val = (u_true_val[:mpc_T, :, 1] - u_pred_val[:, :, 1])**2
        
                # Ideal here would be to scale, but this is fine just to be in the same range
                loss_val = 10*loss_dsigma_val.sum(0).mean() + 10*loss_d_val.sum(0).mean() \
                + loss_v_val.sum(0).mean() + 0.01*loss_a_val.sum(0).mean() + 0.1*loss_delta_val.sum(0).mean()

                print('Train loss:', loss_train_avg)
                
                print('Validation loss:', 
                      round(10*loss_dsigma_val.sum(0).mean().item(), 5),
                      round(10*loss_d_val.sum(0).mean().item(), 5), 
                      #round(loss_phi_val.sum(0).mean().item(), 5), 
                      round(loss_v_val.sum(0).mean().item(), 5), 
                      round(0.01*loss_a_val.sum(0).mean().item(), 5), 
                      round(0.1*loss_delta_val.sum(0).mean().item(), 5), 
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
                        curv_lap = utils_new.get_curve_hor_from_x(x0_lap_pred_torch, track_coord, mpc_H)
                        inp_lap = torch.hstack((x0_lap_pred_torch[:,1:4], curv_lap))
                        q_p_pred_lap = model(inp_lap)
                        q_lap, p_lap = utils_new.q_and_p(mpc_T, q_p_pred_lap, Q_manual, p_manual)

                        q_lap_np_casadi = torch.permute(q_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                        p_lap_np_casadi = torch.permute(p_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()


                        x_b_pred, u_b_pred = utils_new.solve_casadi(
                            q_lap_np_casadi[:,0,:], p_lap_np_casadi[:,0,:],
                            x0_b_pred, dx, du, control)

                        x0_b_pred = x_b_pred[1]
                        x_pred_full = np.append(x_pred_full, x0_b_pred.reshape(-1,1), axis=1)

                        if x0_b_pred[0]>track_coord[2].max().numpy()/2:
                            finished=1

                        if x0_b_pred[1]>bound_d_casadi+0.001 or x0_b_pred[1]<-bound_d_casadi-0.001 or steps>max_steps:
                            crashed=1

                        steps = steps+1
                        
                    lap_time = dt*steps
                    

                    if finished == 1 and lap_time < current_time:
                        current_time = lap_time
                        q_current = q_lap_np_casadi
                        p_current = p_lap
                        x_current_full = x_pred_full

                    # Early stop based on the target trajectory. If it is never satisfied, it goes until the last epoch.
                    if finished == 1 and lap_time < lap_time_H + 0.00001:
                        flag_finish_training = 1
                        print('Achieved a good enough lap performance!!!')

                    if flag_finish_training == 1 or ep==epochs-1:                        
                        torch.save(model.state_dict(), f'./saved_models/model_{str_model}_{flag_finish_training_iter}.pkl')
                        flag_finish_training_iter = flag_finish_training_iter + 1

                    finish_list[b] = finished
                    lap_time_list[b] = lap_time

                print(f'current lap time: {current_time} \t Pred lap time: {lap_time} \t Finished: {finished}')

                if ep>10:
                    import pdb
                    pdb.set_trace()
                
                try:
                    print(x_pred_full[0,60], x_pred_full[0,90], x_pred_full[0,120], x_pred_full[0,150], x_pred_full[0,180])
                    print(x_manual_full_H[0,60], x_manual_full_H[0,90], x_manual_full_H[0,120], x_manual_full_H[0,150], x_manual_full_H[0,180])
                except:
                    print('crash')