import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from concurrent.futures import ProcessPoolExecutor

import utils_new

import torch.autograd.functional as F

from mpc import casadi_control

import scipy.linalg

from tqdm import tqdm

from casadi import *

import time


# PARAMETERS

k_curve = 20.

dt = 0.04

mpc_T = 15
mpc_H = 45

n_Q = 5

assert mpc_T%n_Q==0

l_r = 0.08
l_f = 0.08

v_max = 1.3

delta_max = 0.4

a_max = 1.0

track_density = 300
track_width = 0.5
t_track = 0.3
init_track = [0,0,0]

max_p = 100 

params = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_T])



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



x0 = torch.tensor([0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = torch.tensor([0.0, 0.0])

dx=4
du=2

BS = 128
u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)
eps=0.01
lqr_iter = 50

grad_method = GradMethods.AUTO_DIFF

model = utils_new.SimpleNN(mpc_H, n_Q, 2, max_p)
opt = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)

control = utils_new.CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0, 20, 5, 0, 0, 0, 0, 0, 0, 0]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -2.0, 0, 0, 0, 0]), 0), mpc_T, 0)

idx_to_casadi = [5,1,2,3,8,9] # This is only to match the indices of Q from model to casadi

x_star, u_star = utils_new.solve_casadi(
            Q_manual[:,idx_to_casadi].T, p_manual[:,idx_to_casadi].T,
            x0.detach().numpy(),dx,du,control)

ind = np.array([0,1,3,4])

print(x_star[ind,:],u_star)

x_clamp = torch.clamp(torch.from_numpy(x_star[ind,:]),0.0,1.0)
print(x_clamp)
print(np.shape(x_star))

best_prog = -999999.
for it in range(500):

    x0 = utils_new.sample_init(BS, true_dx)  
    
    x0_diff = x0.clone()
    
    progress_pred = torch.tensor(0.)
    penalty_pred_d = torch.tensor(0.)
    penalty_pred_v = torch.tensor(0.)
    for sim in range(0, mpc_H//mpc_T):
        
        curv = utils_new.get_curve_hor_from_x(x0_diff, track_coord, mpc_H)
        inp = torch.hstack((x0_diff[:,1:4], curv))
        q_p_pred = model(inp)

        q, p = utils_new.q_and_p(mpc_T, q_p_pred, Q_manual, p_manual)
        Q = torch.diag_embed(q, offset=0, dim1=-2, dim2=-1)
        
        pred_x, pred_u, pred_objs = mpc.MPC(
                    true_dx.n_state, true_dx.n_ctrl, mpc_T,
                    u_lower=u_lower, u_upper=u_upper, u_init=u_init,
                    lqr_iter=lqr_iter,
                    verbose=0,
                    exit_unconverged=False,
                    detach_unconverged=False,
                    linesearch_decay=.8,
                    max_linesearch_iter=4,
                    grad_method=grad_method,
                    eps=eps,
                    n_batch=None,
                )(x0_diff, QuadCost(Q, p), true_dx)
        
        x0_diff = pred_x[-1].clone()
        x0_diff[:,4] = x0_diff[:,0]
        x0_diff[:,5] = 0.
        
        progress_pred = progress_pred + pred_x[-1,:,5]
        penalty_pred_d = penalty_pred_d + pred_x[:,:,1]
        penalty_pred_v = penalty_pred_v + pred_x[:,:,3]

    loss = -progress_pred.mean() \
    + 0.0001*true_dx.penalty_d(penalty_pred_d).sum(0).mean() \
    + 0.0001*true_dx.penalty_v(penalty_pred_v).sum(0).mean()
    
    #print(0.001*true_dx.penalty_d(penalty_pred_d).sum(0).mean().detach())
      
    opt.zero_grad()
    loss.backward()
    opt.step()  
    
    
    if it%10==0:
    # V A L I D A T I O N   (only casadi) 
        with torch.no_grad():

            BS_val = 64

            # This sampling should bring always the same set of initial states
            x0_val = utils_new.sample_init(BS_val, true_dx, sn=0).numpy()

            x0_val_pred = x0_val[:,:6]
            x0_val_manual = x0_val[:,:6]

            progress_val_pred = 0.
            progress_val_manual = 0.

            for sim in range(mpc_H//mpc_T):

                x0_val_pred_torch = torch.tensor(x0_val_pred, dtype=torch.float32)
                curv_val = utils_new.get_curve_hor_from_x(x0_val_pred_torch, track_coord, mpc_H)
                inp_val = torch.hstack((x0_val_pred_torch[:,1:4], curv_val))
                q_p_pred_val = model(inp_val)
                q_val, p_val = utils_new.q_and_p(mpc_T, q_p_pred_val, Q_manual, p_manual)
                                
                q_val_np_casadi = torch.permute(q_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                p_val_np_casadi = torch.permute(p_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                x_pred_val = utils_new.solve_casadi_parallel(
                    q_val_np_casadi, p_val_np_casadi, 
                    x0_val_pred, BS_val, dx, du, control)                

                q_manual_casadi = np.expand_dims((Q_manual[:,idx_to_casadi].T), 1)
                p_manual_casadi = np.expand_dims((p_manual[:,idx_to_casadi].T), 1)
                x_manual = utils_new.solve_casadi_parallel(
                    np.repeat(q_manual_casadi, BS_val, 1), 
                    np.repeat(p_manual_casadi, BS_val, 1), 
                    x0_val_manual, BS_val, dx, du, control) 
                
                progress_val_pred = progress_val_pred + x_pred_val[-1,:,5]
                progress_val_manual = progress_val_manual + x_manual[-1,:,5]
                
                x0_val_pred = x_pred_val[-1]
                x0_val_manual = x_manual[-1]
                
                x0_val_pred[:,4] = x0_val_pred[:,0]
                x0_val_manual[:,4] = x0_val_manual[:,0]
                
                x0_val_pred[:,5] = 0.
                x0_val_manual[:,5] = 0.            

            progress_val = progress_val_pred - progress_val_manual
            
            if best_prog<progress_val.mean():
                torch.save(model.state_dict(), f'./saved_models/model_{n_Q}_{mpc_T}_{mpc_H}.pkl')
                           
            print(f'{it}: Progress Diff: ', round(progress_val.mean(), 3), 
                  '\tProgress Pred: ', round(progress_val_pred.mean(), 3),
                  '\tProgress Manual: ', round(progress_val_manual.mean(), 3)
                 )
            
            #print(f'{it}: progress_val_pred: ', progress_val_pred[:4])
            #print(f'{it}: progress_val_manual: ', progress_val_manual[:4])