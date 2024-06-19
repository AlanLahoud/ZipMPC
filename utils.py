import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx



class NN(nn.Module):
    def __init__(self, H, S, O):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(H + S, 512)  
        self.fc2 = nn.Linear(512, 256)  
        self.output1 = nn.Linear(256, O) 
        self.output2 = nn.Linear(256, O) 

    def forward(self, c, x0):
        combined = torch.cat((c, x0), dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q = F.relu(self.output1(x)) + 0.0001
        p = self.output2(x)
        return q, p
        

def sample_xinit(n_batch, track_width, v_max, true_dx):
    def uniform(shape, low, high):
        r = high-low
        return torch.rand(shape)*r+low
    sigma = uniform(n_batch, 6.01, 8.05)
    d = uniform(n_batch, -track_width*0.35, track_width*0.35)
    phi = uniform(n_batch, -0.4*np.pi, 0.4*np.pi)
    v = uniform(n_batch, .01, 0.95*v_max)
    
    sigma = uniform(n_batch, 6.01, 8.05)
    d = uniform(n_batch, -track_width*0.35, track_width*0.35)
    phi = uniform(n_batch, -0.4*np.pi, 0.4*np.pi)
    v = uniform(n_batch, .01, 0.01*v_max)

    sigma_0 = sigma
    sigma_diff = sigma-sigma_0
    
    d_pen = penalty_d(d, 0.40*track_width)
    v_ub = penalty_v(v, v_max)
    
    k = true_dx.curv(sigma)

    xinit = torch.stack((sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub), dim=1)
    return xinit

    
def penalty_d(d, th, factor=100):  
    overshoot_pos = (d - th).clamp(min=0)
    overshoot_neg = (-d - th).clamp(min=0)
    penalty_pos = torch.exp(overshoot_pos) - 1
    penalty_neg = torch.exp(overshoot_neg) - 1 
    return factor*(penalty_pos + penalty_neg)


def penalty_v(v, th, factor=100): 
    overshoot_pos = (v - th).clamp(min=0)
    overshoot_neg = (-v + 0.001).clamp(min=0)
    penalty_pos = torch.exp(overshoot_pos) - 1
    penalty_neg = torch.exp(overshoot_neg) - 1 
    return factor*(penalty_pos + penalty_neg)


def get_nearest_index(point_f, ref_path):
    return ((point_f[0] - ref_path[2,:])**2).argmin()
  
    
def compute_x_coord(point_f, ref_path, nearest_index):
    return ref_path[0,nearest_index] - point_f[1]*torch.sin(ref_path[3,nearest_index])


def compute_y_coord(point_f, ref_path, nearest_index):
    return ref_path[1,nearest_index] + point_f[1]*torch.cos(ref_path[3,nearest_index])


def frenet_to_cartesian(point_f, ref_path):     
    nearest_index = get_nearest_index(point_f, ref_path)
    x = compute_x_coord(point_f, ref_path, nearest_index)
    y = compute_y_coord(point_f, ref_path, nearest_index)    
    return torch.tensor([x, y])


def get_loss_progress(x_init, dx, _Q, _p, mpc_T):     
        pred_x, pred_u, pred_objs = mpc.MPC(
            dx.n_state, dx.n_ctrl, mpc_T,
            u_lower=u_lower, u_upper=u_upper, u_init=u_init,
            lqr_iter=lqr_iter,
            verbose=1,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=.8,
            max_linesearch_iter=10,
            grad_method=grad_method,
            eps=eps,
            n_batch=n_batch,
        )(x_init, QuadCost(_Q, _p), dx)        
        progress_loss = torch.mean(-pred_x[mpc_T-1,:,0] + pred_x[0,:,0])                    
        return progress_loss
    
    
    
def get_loss_progress_new(x_init_train, x_init_sim, 
                          dx, dx_sim, true_sim_dx,
                          _Q, _p, u_lower, u_upper, u_init,
                          lqr_iter, eps, n_batch, grad_method,
                          mpc_T, H_curve):    
               
        assert H_curve%mpc_T == 0
        
        x_curr_sim = x_init_sim
        x_curr_train = x_init_train
        
        #if np.random.random()<0.02:
        #    import pdb
        #    pdb.set_trace()
        
        for s in range(H_curve//mpc_T):
                    
            pred_x, pred_u, pred_objs = mpc.MPC(
                dx.n_state, dx.n_ctrl, mpc_T,
                u_lower=u_lower, u_upper=u_upper, u_init=u_init,
                lqr_iter=lqr_iter,
                verbose=0,
                exit_unconverged=False,
                detach_unconverged=False,
                linesearch_decay=.4,
                max_linesearch_iter=4,
                grad_method=grad_method,
                eps=eps,
                n_batch=n_batch,
            )(x_curr_train, QuadCost(_Q, _p), dx)
            
            for ss in range(mpc_T):
                x_curr_sim_ = x_curr_sim.clone()
                x_curr_sim = true_sim_dx.forward(x_curr_sim_, pred_u[ss])

            x_curr_train = x_curr_sim
            x_curr_train[:,4] = x_curr_train[:,0]
            x_curr_train[:,5] = 0.
        
        progress_loss = torch.mean(-x_curr_train[:,0] + x_init_train[:,0])
         
        # Below is to check if negative sigma isbeing outputted    
        #mask_weird = x_curr_train[:,0]<x_init_train[:,0]
        #print(x_init_train[mask_weird].shape)
            
            
        d_loss = torch.mean(x_curr_train[:,1]**2)
            
        return progress_loss, d_loss
       
    
def get_curve_hor_from_x(x, track_coord, H_curve):
    idx_track_batch = ((x[:,0]-track_coord[[2],:].T)**2).argmin(0)
    idcs_track_batch = idx_track_batch[:, None] + torch.arange(H_curve)
    curvs = track_coord[4,idcs_track_batch].float()
    return curvs


def cost_to_batch_NN(q, p, n_batch, mpc_T):
    Q_batch = torch.zeros(n_batch, q.shape[1], q.shape[1])
    rows, cols = torch.arange(q.shape[1]), torch.arange(q.shape[1])  
    Q_batch[:, rows, cols] = q 
    Q_batch = Q_batch.unsqueeze(0).repeat(
                mpc_T, 1, 1, 1)    
    p_batch = p.unsqueeze(0).repeat(mpc_T, 1, 1)
    return Q_batch, p_batch


def cost_to_batch(q, p, n_batch, mpc_T):
    Q_batch = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
                mpc_T, n_batch, 1, 1
            )
    p_batch = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)   
    return Q_batch, p_batch


def bound_params(q, p):
    q[:,1] = 1.
    q[:,2] = 1.
    q[:,3] = 0.00001

    #q = q + 1.
    q[:,0] = 0.00001
    q[:,4] = 0.00001
    q[:,5] = 0.00001
    q = q.clip(0.00001, 40.)
    p[:,0] = 0.0
    p[:,1] = 0.0
    p[:,2] = 0.0
    p[:,3] = 0.0
    p[:,4] = 0.0
    p[:,5] = 0.0
    p2 = p.clone()
    p2 = p.clip(-200.,200.)
    q2 = q.clone()
    return q2, p2


def bound_params_paj(q, p):
    q[:,1] = q[:,1] + 10.0
    #q = q + 1.
    q[:,0] = 0.00001
    q[:,6] = 0.00001
    q = q.clip(0.00001, 40.)
    p[:,0] = 0.0
    p[:,1] = 0.0
    p[:,2] = 0.0
    p[:,6] = 0.0
    p2 = p.clone()
    p2 = p.clip(-200.,200.)
    q2 = q.clone()
    return q2, p2


def inference_params(x_in, track_coord, H_curve, model, q_pen, p_pen, N, mpc_T):
    curvs = get_curve_hor_from_x(x_in, track_coord, H_curve)
    q, p = model(curvs, x_in[:,1:4])
    q = torch.cat((q[:,:6], q_pen, q[:,6:]), dim=1)
    p = torch.cat((p[:,:6], p_pen, p[:,6:]), dim=1)
    q2, p2 = bound_params(q, p) 
    Q_batch, p_batch = cost_to_batch_NN(q2, p2, N, mpc_T)
    return Q_batch, p_batch


def inference_params_paj(x_in, track_coord, H_curve, model, q_pen, p_pen, N, mpc_T):
    curvs = get_curve_hor_from_x(x_in, track_coord, H_curve)
    q, p = model(curvs, x_in[:,1:6])
    q = torch.cat((q[:,:8], q_pen, q[:,8:]), dim=1)
    p = torch.cat((p[:,:8], p_pen, p[:,8:]), dim=1)
    q2, p2 = bound_params(q, p) 
    Q_batch, p_batch = cost_to_batch_NN(q2, p2, N, mpc_T)
    return Q_batch, p_batch