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

import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--mpc_T', type=int, default=15)
    parser.add_argument('--mpc_H', type=int, default=45)
    parser.add_argument('--n_Q', type=int, default=5)
    parser.add_argument('--l_r', type=float, default=0.10)
    parser.add_argument('--v_max', type=float, default=1.5)
    parser.add_argument('--delta_max', type=float, default=0.4)
    parser.add_argument('--eps_dyn', type=float, default=0.05)
    
    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

mpc_T = args.mpc_T
mpc_H = args.mpc_H
n_Q = args.n_Q

l_r = args.l_r
v_max = args.v_max
delta_max = args.delta_max

eps_dyn = args.eps_dyn

seed_n = 0
torch.manual_seed(seed_n)
np.random.seed(seed_n)


# PARAMETERS

k_curve = 30.
dt = 0.04

#mpc_T = 15
#mpc_H = 45

#n_Q = 5

assert mpc_T%n_Q==0

#l_r = 0.12
l_f = l_r

#v_max = 1.8

#delta_max = 0.4

a_max = 2.0

track_density = 300
track_width = 0.5
t_track = 0.3
init_track = [0,0,0]

max_p = 100 

str_model = f'{mpc_T}_{mpc_H}_{n_Q}_{l_r}_{delta_max}_{v_max}_{eps_dyn}'

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
lqr_iter = 70

grad_method = GradMethods.AUTO_DIFF

model = utils_new.SimpleNN(mpc_H, n_Q, 3, max_p)
opt = torch.optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-5)
#opt = torch.optim.RMSprop(model.parameters(), lr=0.0005)

control = utils_new.CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0, 0.5, 0.5, 0, 0, 0.001, 0, 0, 0, 0]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -0.1, 0, 0, 0, 0]), 0), mpc_T, 0)

idx_to_casadi = [5,1,2,3,8,9] # This is only to match the indices of Q from model to casadi

#x_star, u_star = utils_new.solve_casadi(
#            Q_manual[:,idx_to_casadi].T, p_manual[:,idx_to_casadi].T,
#            x0.detach().numpy(),dx,du,control)

ind = np.array([0,1,3,4])

#print(x_star[ind,:],u_star)

#x_clamp = torch.clamp(torch.from_numpy(x_star[ind,:]),0.0,1.0)
#print(x_clamp)
#print(np.shape(x_star))


buffer_x0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

def add_x0_to_buffer(x0, buffer_x0):
    x0_new = x0.clone()
    x0_new[:,0] = x0[:,0] + 0.1*torch.randn_like(x0[:,0])
    x0_new[:,1] = x0[:,1] + 0.02*torch.randn_like(x0[:,1])
    x0_new[:,2] = x0[:,2] + 0.02*torch.randn_like(x0[:,2])
    x0_new[:,3] = x0[:,3] + torch.randn_like(x0[:,3])
    
    mask = (x0_new[:, 1] < 0.15) \
    & (x0_new[:, 1] > -0.15) \
    & (x0_new[:, 2] < 1.00) \
    & (x0_new[:, 2] > -1.00) \
    & (x0_new[:, 3] > 0) \
    & (x0_new[:, 3] < v_max) \
    & (x0_new[:, 0] < 6.) \
    & (torch.randn(1,).squeeze()<-0.3)
    selected_rows = x0_new[mask]

    buffer_x0_new = buffer_x0.clone()
    
    buffer_x0_new = torch.vstack((buffer_x0,selected_rows))
    return buffer_x0_new

def sample_x0_from_buffer(BS, buffer_x0):
    idxs = torch.randint(0, len(buffer_x0), (BS,))
    x0_sample = buffer_x0[idxs]
    return x0_sample


def model_mismatch_apply(true_dx):
    true_dx.l_r = 0.10
    true_dx.l_f = 0.10
    return true_dx
    
def model_mismatch_reverse(true_dx):
    true_dx.l_r = l_r
    true_dx.l_f = l_r
    return true_dx


best_prog = -999999.


for it in range(361):

    #x0 = utils_new.sample_init(BS, true_dx)  
    
    #x0 = utils_new.sample_init_traj(BS, true_dx, x_star, num_patches, patch+1)
    
    #print(buffer_x0.shape, print(buffer_x0.mean(0)), print(buffer_x0.std(0)))
    
    x0 = sample_x0_from_buffer(BS, buffer_x0).detach()
    
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
        
        #import pdb
        #pdb.set_trace()
        
        pred_u_noise = pred_u #+ eps_dyn*torch.randn_like(pred_u)
              
        
        for iu in range(pred_u.shape[0]):
            x0_diff_previous = x0_diff.clone()
            
            true_dx = model_mismatch_apply(true_dx)
            x0_diff = true_dx.forward(x0_diff, pred_u_noise[iu])
            true_dx = model_mismatch_reverse(true_dx)
            
            x0_diff = torch.where((x0_diff[:,0].abs()>10.00).unsqueeze(-1), x0_diff_previous, x0_diff)
            x0_diff = torch.where((x0_diff[:,1].abs()>0.25).unsqueeze(-1), x0_diff_previous, x0_diff)
            x0_diff = torch.where((x0_diff[:,2].abs()>2.00).unsqueeze(-1), x0_diff_previous, x0_diff)
            x0_diff = torch.where((x0_diff[:,3].abs()>1.70).unsqueeze(-1), x0_diff_previous, x0_diff)
            
            
            #if x0_diff[:,0].max()>15:
            #    import pdb
            #    pdb.set_trace()

        if sim==0:
            for xx in pred_x[:3]: #only few steps
                buffer_x0_old = buffer_x0.clone()
                buffer_x0 = add_x0_to_buffer(xx, buffer_x0_old)
        #print(buffer_x0.max(0)[0])
        #x0_diff = pred_x[-1].clone()
        x0_diff[:,4] = x0_diff[:,0]     
        
        progress_pred = progress_pred + x0_diff[:,5]
        penalty_pred_d = penalty_pred_d + x0_diff[:,6]
        penalty_pred_v = penalty_pred_v + x0_diff[:,7]
        
        x0_diff[:,5] = 0.

        #import pdb
        #pdb.set_trace()
    
    loss = -progress_pred.mean() \
    + 0.001*penalty_pred_d.mean() \
    + 0.001*penalty_pred_v.mean()
    
    #print('Progress train and penalties:', 
    #      -progress_pred.mean().detach().item(), 
    #      0.001*penalty_pred_d.mean().detach().item(), 
    #      0.001*penalty_pred_v.mean().detach().item()
    #        )
    
    #print(0.001*true_dx.penalty_d(penalty_pred_d).sum(0).mean().detach())
      
    opt.zero_grad()
    loss.backward()
    opt.step()  
    
    
    if it%5==0:
    # V A L I D A T I O N   (only casadi) 
        with torch.no_grad():

            BS_val = 32

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
                
                #print('Qd, Qs:', q_val[:,:,[1,5]].mean(0).mean(0).detach().numpy())
                #print('Pd, Ps:', p_val[:,:,[1,5]].mean(0).mean(0).detach().numpy())
                                
                q_val_np_casadi = torch.permute(q_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                p_val_np_casadi = torch.permute(p_val[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
                x_pred_val, u_pred_val = utils_new.solve_casadi_parallel(
                    q_val_np_casadi, p_val_np_casadi, 
                    x0_val_pred, BS_val, dx, du, control) 
                
                np.random.seed(0)
                u_pred_noise = u_pred_val #+ eps_dyn*np.random.randn(u_pred_val.shape[0], u_pred_val.shape[1], u_pred_val.shape[2])
                
                for iu in range(u_pred_val.shape[0]):
                    x0_val_pred_previous = torch.tensor(x0_val_pred).clone()
                    true_dx = model_mismatch_apply(true_dx)
                    x0_val_pred = true_dx.forward(torch.tensor(x0_val_pred), torch.tensor(u_pred_noise[iu]))[:,:6]
                    true_dx = model_mismatch_reverse(true_dx)
                    x0_val_pred = torch.where((x0_val_pred[:,1].abs()>0.17).unsqueeze(-1), x0_val_pred_previous, x0_val_pred)
                    x0_val_pred = torch.where((x0_val_pred[:,2].abs()>2.00).unsqueeze(-1), x0_val_pred_previous, x0_val_pred)
                    x0_val_pred = torch.where((x0_val_pred[:,0].abs()>10.00).unsqueeze(-1), x0_val_pred_previous, x0_val_pred)  
                    x0_val_pred = torch.where((x0_val_pred[:,3].abs()>1.50).unsqueeze(-1), x0_val_pred_previous, x0_val_pred).numpy()   
                
                q_manual_casadi = np.expand_dims((Q_manual[:,idx_to_casadi].T), 1)
                p_manual_casadi = np.expand_dims((p_manual[:,idx_to_casadi].T), 1)
                x_manual, u_manual = utils_new.solve_casadi_parallel(
                    np.repeat(q_manual_casadi, BS_val, 1), 
                    np.repeat(p_manual_casadi, BS_val, 1), 
                    x0_val_manual, BS_val, dx, du, control) 
                
                np.random.seed(0)
                u_manual_noise = u_manual #+ eps_dyn*np.random.randn(u_manual.shape[0], u_manual.shape[1], u_manual.shape[2])
                
                for iu in range(u_manual.shape[0]):
                    x0_val_manual_previous = torch.tensor(x0_val_manual).clone()
                    true_dx = model_mismatch_apply(true_dx)
                    x0_val_manual = true_dx.forward(torch.tensor(x0_val_manual),torch.tensor(u_manual_noise[iu]))[:,:6]
                    true_dx = model_mismatch_reverse(true_dx)
                    x0_val_manual = torch.where((x0_val_manual[:,1].abs()>0.17).unsqueeze(-1), 
                                                x0_val_manual_previous, x0_val_manual)
                    x0_val_manual = torch.where((x0_val_manual[:,2].abs()>2.00).unsqueeze(-1), 
                                                x0_val_manual_previous, x0_val_manual)
                    x0_val_manual = torch.where((x0_val_manual[:,0].abs()>10.00).unsqueeze(-1), 
                                                x0_val_manual_previous, x0_val_manual)
                    x0_val_manual = torch.where((x0_val_manual[:,3].abs()>1.50).unsqueeze(-1), 
                                                x0_val_manual_previous, x0_val_manual).numpy()
                
               
                
                #x0_val_pred = x_pred_val[-1]
                #x0_val_manual = x_manual[-1]
                     
                x0_val_pred[:,4] = x0_val_pred[:,0]
                x0_val_manual[:,4] = x0_val_manual[:,0]
                
                progress_val_pred = progress_val_pred + x0_val_pred[:,5]
                progress_val_manual = progress_val_manual + x0_val_manual[:,5]
                
                x0_val_pred[:,5] = 0.
                x0_val_manual[:,5] = 0. 
                
                

            progress_val = progress_val_pred - progress_val_manual
            
            if best_prog<progress_val.mean():
                torch.save(model.state_dict(), f'./saved_models/model_{str_model}.pkl')
                           
            print(f'{it}: Progress Diff: ', round(progress_val.mean(), 3), 
                  '\tProgress Pred: ', round(progress_val_pred.mean(), 3),
                  '\tProgress Manual: ', round(progress_val_manual.mean(), 3)
                 )
            
            #print(f'{it}: progress_val_pred: ', progress_val_pred[:4])
            #print(f'{it}: progress_val_manual: ', progress_val_manual[:4])

    if it%50==0:
        # L A P   P E R F O R M A N C E    (E V A L U A T I O N)
        with torch.no_grad():

            print('LAP PERFORMANCE:')
            BS_test = 2

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
                    
                    if x0_b_pred[0]>track_coord[2].max().numpy():
                        finished=1
                        
                    if x0_b_pred[1]>0.17 or x0_b_pred[1]<-0.17 or steps>max_steps:
                        crashed=1

                    steps = steps+1

                lap_time = dt*steps
                
                finish_list[b] = finished
                lap_time_list[b] = lap_time

            import pdb
            pdb.set_trace()
            
            print('Pred finish: ', finish_list)
            print('Pred lap time: ', lap_time_list)

            if it==0:
                for b in range(BS_test):
                    finished = 0
                    crashed = 0
                    steps = 0
                    max_steps=500
    
                    x0_b_manual = x0_lap_manual[b].copy()

                    #import pdb
                    #pdb.set_trace()
                    
                    while finished==0 and crashed==0:
                        q_lap_manual_casadi = Q_manual[:,idx_to_casadi].T
                        p_lap_manual_casadi = p_manual[:,idx_to_casadi].T
                        
                        x_b_manual, u_b_manual = utils_new.solve_casadi(
                            q_lap_manual_casadi, p_lap_manual_casadi, 
                            x0_b_manual, dx, du, control) 
    
                        x0_b_manual = x_b_manual[1]
                        
                        if x0_b_manual[0]>track_coord[2].max().numpy():
                            finished=1
                            
                        if x0_b_manual[1]>0.17 or x0_b_manual[1]<-0.17 or steps>max_steps:
                            crashed=1
    
                        steps = steps+1
    
                    lap_time = dt*steps
                    
                    finish_list[b] = finished
                    lap_time_list[b] = lap_time
                    
                print('Manual finish: ', finish_list)
                print('Manual lap time: ', lap_time_list)