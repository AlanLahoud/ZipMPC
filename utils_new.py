import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from concurrent.futures import ProcessPoolExecutor

import torch.autograd.functional as F

from mpc import casadi_control

import scipy.linalg

from tqdm import tqdm

from casadi import *

import time




class CasadiControl():
    def __init__(self, track_coordinates, params):
        super().__init__()

        params = params.numpy()
        
        # states: sigma, d, phi, v (4) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1) + ac_ub (1)
        self.n_state = 4+2+1+1+1
        print(self.n_state)          # here add amount of states plus amount of exact penalty terms
        # control: a, delta
        self.n_ctrl = 2

        self.track_coordinates = track_coordinates

        # everything to calculate curvature
        self.track_sigma = self.track_coordinates[2,:]
        self.track_curv = self.track_coordinates[4,:]

        self.track_curv_shift = torch.empty(self.track_curv.size())
        self.track_curv_shift[1:] = self.track_curv[0:-1]
        self.track_curv_shift[0] = self.track_curv[-1]
        self.track_curv_diff = self.track_curv - self.track_curv_shift

        self.mask = torch.where(torch.absolute(self.track_curv_diff) < 0.1, False, True)
        self.sigma_f = self.track_sigma[self.mask]
        self.curv_f = self.track_curv_diff[self.mask]

        self.params = params

        self.l_r = params[0]
        self.l_f = params[1]
        
        self.track_width = params[2]
        
        self.delta_threshold_rad = np.pi
        self.dt = params[3]

        self.smooth_curve = params[4]
        
        self.v_max = params[5]
        
        self.delta_max = params[6]
        
        self.a_max = params[7]
        
        self.mpc_T = int(params[8])
        
    def sigmoid(self, x):
        return (tanh(x/2)+1.)/2

    def curv_casadi(self, sigma):
        
        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[1],1)
   
        sigma_f_mat_np = sigma_f_mat.numpy()
        sigma_f_np = self.sigma_f.numpy()
        curv_f_np = self.curv_f.numpy()

        sigma_shifted = reshape(sigma,num_s[1],1)- sigma_f_mat_np
        curv_unscaled = self.sigmoid(self.smooth_curve*sigma_shifted)
        curv = reshape((curv_unscaled@(curv_f_np.reshape(-1,1))),1,num_s[1])

        return curv
    
    
    def mpc_casadi(self,q,p,x0_np,dx,du):
        mpc_T = self.mpc_T

        x_sym = SX.sym('x_sym',dx,mpc_T+1)
        u_sym = SX.sym('u_sym',du,mpc_T)
        
        dt = self.dt

        #beta = np.arctan(l_r/(l_r+l_f)*np.tan(u_sym[1,0:mpc_T]))

        #dyn1 = horzcat(
        #    (x_sym[0,0] - x0_np[0]), 
        #    (x_sym[0,1:mpc_T+1] - x_sym[0,0:mpc_T] - dt*(x_sym[3,0:mpc_T]*(np.cos(x_sym[2,0:mpc_T]+beta)/(1.-self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[1,0:mpc_T])))))

        #dyn2 = horzcat(
        #    (x_sym[1,0] - x0_np[1]), 
        #    (x_sym[1,1:mpc_T+1] - x_sym[1,0:mpc_T] - dt*(x_sym[3,0:mpc_T]*np.sin(x_sym[2,0:mpc_T]+beta))))

        #dyn3 = horzcat(
        #    (x_sym[2,0] - x0_np[2]), 
        #    (x_sym[2,1:mpc_T+1] - x_sym[2,0:mpc_T] - dt*(x_sym[3,0:mpc_T]*(1/l_f)*np.sin(beta)-self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[3,0:mpc_T]*(np.cos(x_sym[2,0:mpc_T]+beta)/(1-self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[1,0:mpc_T])))))

        #dyn4 = horzcat(
        #    (x_sym[3,0] - x0_np[3]), 
        #    (x_sym[3,1:mpc_T+1] - x_sym[3,0:mpc_T] - dt*(u_sym[0,0:mpc_T])))
        
        

        #dphi = v/self.l_f*torch.sin(beta)-k*v*(torch.cos(phi+beta)/(1-k*d))               

        #dphi = (v * torch.tan(delta)) / (self.l_r+self.l_f) - k * dsigma
        
        dyn1 = horzcat(
            (x_sym[0,0] - x0_np[0]), 
            (x_sym[0,1:mpc_T+1] - x_sym[0,0:mpc_T] - dt*(x_sym[3,0:mpc_T]*(np.cos(x_sym[2,0:mpc_T])/(1.-self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[1,0:mpc_T])))))

        dyn2 = horzcat(
            (x_sym[1,0] - x0_np[1]), 
            (x_sym[1,1:mpc_T+1] - x_sym[1,0:mpc_T] - dt*(x_sym[3,0:mpc_T]*np.sin(x_sym[2,0:mpc_T]))))

        dyn3 = horzcat(
            (x_sym[2,0] - x0_np[2]), 
            (x_sym[2,1:mpc_T+1] - x_sym[2,0:mpc_T] - \
             dt*(x_sym[3,0:mpc_T]*(1/(self.l_f+self.l_r))*np.tan(u_sym[1,0:mpc_T])\
                 -self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[3,0:mpc_T]*(np.cos(x_sym[2,0:mpc_T])/(1-self.curv_casadi(x_sym[0,0:mpc_T])*x_sym[1,0:mpc_T])))))

        dyn4 = horzcat(
            (x_sym[3,0] - x0_np[3]), 
            (x_sym[3,1:mpc_T+1] - x_sym[3,0:mpc_T] - dt*(u_sym[0,0:mpc_T])))

        feat = vertcat(x_sym[0,0:mpc_T]-x0_np[0],x_sym[1:,0:mpc_T],u_sym[:,0:mpc_T])

        q_sym = SX.sym('q_sym',dx+du,mpc_T)
        p_sym = SX.sym('q_sym',dx+du,mpc_T)
        Q_sym = diag(q_sym)

        l = sum2(sum1(0.5*q_sym*feat*feat + p_sym*feat))
        dl = substitute(substitute(l,q_sym,q),p_sym,p)

        const = vertcat(
                transpose(dyn1),
                transpose(dyn2),
                transpose(dyn3),
                transpose(dyn4),
                transpose(u_sym[0,0:mpc_T]),
                transpose(u_sym[1,0:mpc_T]),
                transpose(x_sym[1,0:mpc_T+1]),
                transpose(x_sym[3,0:mpc_T+1]))

        lbg = np.r_[np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    -self.a_max*np.ones(mpc_T),
                    -self.delta_max*np.ones(mpc_T),
                    -0.3*self.track_width*np.ones(mpc_T+1),
                    -0.1*np.ones(mpc_T+1)]

        ubg = np.r_[np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    self.a_max*np.ones(mpc_T),
                    self.delta_max*np.ones(mpc_T),
                    0.3*self.track_width*np.ones(mpc_T+1),
                    self.v_max*np.ones(mpc_T+1)]


        lbx = -np.inf * np.ones(dx*(mpc_T+1)+du*mpc_T)
        ubx = np.inf * np.ones(dx*(mpc_T+1)+du*mpc_T)

        x = vertcat(reshape(x_sym[:,0:mpc_T+1],(dx*(mpc_T+1),1)),
                    reshape(u_sym[:,0:mpc_T],(du*mpc_T,1)))

        options = {
                    'verbose': False,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.tol': 1e-4,
                    'ipopt.max_iter': 4000,
                    'ipopt.hessian_approximation': 'limited-memory'
                }

        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        solver_output = solver(**solver_input)

        sol = solver_output['x']

        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*mpc_T:].reshape(-1,du)
        x = sol_evalf[:-du*mpc_T].reshape(-1,dx)

        return x, u

class SimpleNN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K):
        super(SimpleNN, self).__init__()
        input_size = 3 + mpc_H
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, mpc_T*O)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.K = K
        self.O = O
        self.mpc_T = mpc_T

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = self.output_activation(x) * self.K
        x = x.reshape(self.mpc_T, -1, self.O)
        return x/3

def sample_init(BS, dyn, sn=None):
    
    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not
    
    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)
    
    di = 1000
    sigma_sample = torch.randint(int(0.0*di), int(4.0*di), (BS,1), generator=gen)/di
    d_sample = torch.randint(int(-.10*di), int(.10*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.08*di), int(0.08*di), (BS,1), generator=gen)/di
    v_sample = torch.randint(0, int(1.0*di), (BS,1), generator=gen)/di
    
    sigma_diff_sample = torch.zeros((BS,1))
    
    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)
    
    x_init_sample = torch.hstack((
        sigma_sample, d_sample, phi_sample, v_sample, 
        sigma_sample, sigma_diff_sample, d_pen, v_pen))   
    
    return x_init_sample


def sample_init_traj(BS, dyn, traj, num_patches, patch, sn=None):
    
    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not
    
    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)
    
    di = 1000
    
    # The idea is to take the trajectory and divide it into num_patches patches.
    # The variable patch indicates in which patch we currently are. 
    
    # we also need to adapt the batch size such that no 
    #print('traj:',traj)
    
    traj_steps = np.shape(traj)
    #print('traj_steps:',traj_steps)
    patch_steps = np.floor(traj_steps[0]/num_patches)
    
    # in this step we randomly 
    traj_ind_sample = torch.randint(0,int(patch_steps*patch),(BS,1), generator=gen)
    #print('traj_ind_sample:',traj_ind_sample)
    traj_sample = traj[traj_ind_sample.detach().numpy().flatten(),:]
    #print('traj_sample:',traj_sample)
    
    # now we keep sigma as it is and sample d, phi, and v in a small "tube" around the traj points
    
    # Note that we clamp the sampled d and v values to stay in their constraints. Sampling constraint violating 
    # states would not make sense. 
    
    d_sample = torch.clamp(torch.from_numpy(traj_sample[:,1].reshape(-1,1))+torch.randint(int(-.08*di), int(.08*di), (BS,1), generator=gen)/di,-0.24,0.24)
    phi_sample = torch.from_numpy(traj_sample[:,2].reshape(-1,1))+torch.randint(int(-0.02*di), int(0.02*di), (BS,1), generator=gen)/di
    v_sample = torch.clamp(torch.from_numpy(traj_sample[:,3].reshape(-1,1))+torch.randint(int(-0.05*di), int(.05*di), (BS,1), generator=gen)/di,0.0,2.5)
    
    # and this part we can actually keep
    
    sigma_diff_sample = torch.zeros((BS,1))
    
    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)
    
    x_init_sample = torch.hstack((
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), d_sample, phi_sample, v_sample, 
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), sigma_diff_sample, d_pen, v_pen))   
    
    return x_init_sample

def get_curve_hor_from_x(x, track_coord, H_curve):
    idx_track_batch = ((x[:,0]-track_coord[[2],:].T)**2).argmin(0)
    idcs_track_batch = idx_track_batch[:, None] + torch.arange(H_curve)
    curvs = track_coord[4,idcs_track_batch].float()
    return curvs

class FrenetKinBicycleDx(nn.Module):
    def __init__(self, track_coordinates, params, dev):
        super().__init__()
        
        self.params = params

        # states: sigma, d, phi, v (4) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1)
        self.n_state = 4+2+1+1
        print('Number of states:', self.n_state)
        
        self.n_ctrl = 2 # control: a, delta

        self.track_coordinates = track_coordinates.to(dev)

        # everything to calculate curvature
        self.track_sigma = self.track_coordinates[2,:]
        self.track_curv = self.track_coordinates[4,:]

        self.track_curv_shift = torch.empty(self.track_curv.size()).to(dev)
        self.track_curv_shift[1:] = self.track_curv[0:-1]
        self.track_curv_shift[0] = self.track_curv[-1]
        self.track_curv_diff = self.track_curv - self.track_curv_shift

        self.mask = torch.where(torch.absolute(self.track_curv_diff) < 0.1, False, True)
        self.sigma_f = self.track_sigma[self.mask]
        self.curv_f = self.track_curv_diff[self.mask]
     
        self.l_r = params[0]
        self.l_f = params[1]
        
        self.track_width = params[2]
        
        self.delta_threshold_rad = np.pi
        self.dt = params[3]

        self.smooth_curve = params[4]
        
        self.v_max = params[5]
        
        self.delta_max = params[6]
        
        self.factor_pen = 1000.
                
        
        
    def curv(self, sigma):

        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[0],1)

        sigma_shifted = sigma.reshape(-1,1) - sigma_f_mat
        curv_unscaled = torch.sigmoid(self.smooth_curve*sigma_shifted)
        curv = (curv_unscaled@(self.curv_f.reshape(-1,1))).type(torch.float)

        return curv.reshape(-1)
    
    
    def penalty_d(self, d):  
        overshoot_pos = (d - 0.3*self.track_width).clamp(min=0)
        overshoot_neg = (-d - 0.3*self.track_width).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1 
        return self.factor_pen*(penalty_pos + penalty_neg)
    
    def penalty_v(self, v):          
        overshoot_pos = (v - self.v_max).clamp(min=0)
        overshoot_neg = (-v + 0.001).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1 
        return self.factor_pen*(penalty_pos + penalty_neg)
    
    def penalty_delta(self, delta):          
        overshoot_pos = (delta - self.delta_max).clamp(min=0)
        overshoot_neg = (-delta - self.delta_max).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1 
        return self.factor_pen*(penalty_pos + penalty_neg)
    
    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()


        a, delta = torch.unbind(u, dim=1)

        sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        
        beta = torch.atan(self.l_r/(self.l_r+self.l_f)*torch.tan(delta))       
        k = self.curv(sigma)

        #dsigma = v*(torch.cos(phi+beta)/(1.-k*d))
        #dd = v*torch.sin(phi+beta)
        #dphi = v/self.l_f*torch.sin(beta)-k*v*(torch.cos(phi+beta)/(1-k*d))               
        #dv = a   
        
        dsigma = v * torch.cos(phi) / (1 - d * k)
        dd = v * torch.sin(phi)
        dphi = (v * torch.tan(delta)) / (self.l_r+self.l_f) - k * dsigma
        dv = a

        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        v = v + self.dt * dv 
        
        sigma_diff = sigma - sigma_0 
                
        d_pen = self.penalty_d(d)        
        v_ub = self.penalty_v(v)

        state = torch.stack((sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub), 1)
        
        return state

def solve_casadi(q_np,p_np,x0_np,dx,du,control):
    
    mpc_T = q_np.shape[1]
    
    x_curr_opt, u_curr_opt = control.mpc_casadi(q_np,p_np,x0_np,dx,du)

    sigzero_curr_opt = np.expand_dims(x_curr_opt[[0],0].repeat(mpc_T+1), 1)
    sigsiff_curr_opt = x_curr_opt[:,[0]]-x_curr_opt[0,0]

    x_curr_opt_plus = np.concatenate((
        x_curr_opt,sigzero_curr_opt,sigsiff_curr_opt), axis = 1)

    x_star = x_curr_opt_plus[:-1]
    u_star = u_curr_opt
    
    return x_star, u_star

def process_single_casadi(sample, q, p, x0, dx, du, control):
    x, u = solve_casadi(
        q[:,sample], p[:,sample], 
        x0[sample], dx, du, control)
    return sample, x

def solve_casadi_parallel(q, p, x0, BS, dx, du, control):
    x = np.zeros((q.shape[2],q.shape[1],x0.shape[-1]))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(
            process_single_casadi, 
            sample, q, p, x0, dx, du, control) for sample in range(BS)]

        for future in futures:
            sample, x_sample = future.result()
            x[:, sample] = x_sample

    return x


def q_and_p(mpc_T, q_p_pred, Q_manual, p_manual):
    # Cost order: 
    # [for casadi] sigma_diff, d, phi, v, a, delta
    # [for model]  sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_pen, a, delta
    
    n_Q, BS, _ = q_p_pred.shape 
    
    q_p_pred = q_p_pred.repeat(mpc_T//n_Q, 1, 1)
    
    e = 1e-9
    
    q = e*torch.ones((mpc_T,BS,10)) + torch.tensor(Q_manual).unsqueeze(1).float()
    p = torch.zeros((mpc_T,BS,10)) + torch.tensor(p_manual).unsqueeze(1).float()

    #sigma_diff
    #q[:,:,5] = q[:,:,5] + q_p_pred[:,:,0].clamp(e)
    p[:,:,5] = p[:,:,5] + q_p_pred[:,:,0]
    
    #d
    q[:,:,1] = (q[:,:,1] + q_p_pred[:,:,1]).clamp(e + 0.5)
    p[:,:,1] = p[:,:,1] + q_p_pred[:,:,2]    
    
    return q, p
