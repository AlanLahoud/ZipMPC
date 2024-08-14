import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from torch.func import jacfwd, vmap

import utils

import cvxpy as cp
#from cvxpylayers.torch import CvxpyLayer

import torch.autograd.functional as F

from mpc import casadi_control

import scipy.linalg

from tqdm import tqdm

from casadi import *

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

        beta = np.arctan(l_r/(l_r+l_f)*np.tan(u_sym[1,0:mpc_T]))

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
             dt*(x_sym[3,0:mpc_T]*(1/(l_f+l_r))*np.tan(u_sym[1,0:mpc_T])\
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
                    -0.35*self.track_width*np.ones(mpc_T+1),
                    -0.1*np.ones(mpc_T+1)]

        ubg = np.r_[np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    self.a_max*np.ones(mpc_T),
                    self.delta_max*np.ones(mpc_T),
                    0.35*self.track_width*np.ones(mpc_T+1),
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
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, mpc_T*O)
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
        return x

def sample_init(BS, dyn, sn=None):
    
    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not
    
    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)
    
    di = 1000
    sigma_sample = torch.randint(int(0.5*di), int(2.0*di), (BS,1), generator=gen)/di
    d_sample = torch.randint(int(-.15*di), int(.15*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.08*di), int(0.08*di), (BS,1), generator=gen)/di
    v_sample = torch.randint(0, int(.22*di), (BS,1), generator=gen)/di
    
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
        
        self.factor_pen = 10.
                
        
        
    def curv(self, sigma):

        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[0],1)

        sigma_shifted = sigma.reshape(-1,1) - sigma_f_mat
        curv_unscaled = torch.sigmoid(self.smooth_curve*sigma_shifted)
        curv = (curv_unscaled@(self.curv_f.reshape(-1,1))).type(torch.float)

        return curv.reshape(-1)
    
    
    def penalty_d(self, d):  
        overshoot_pos = (d - 0.35*self.track_width).clamp(min=0)
        overshoot_neg = (-d - 0.35*self.track_width).clamp(min=0)
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
                
        d_pen = 1000.*self.penalty_d(d)        
        v_ub = 1000.*self.penalty_v(v)

        state = torch.stack((sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub), 1)
        
        return state

def solve_casadi(q_np,p_np,x0_np,dx,du,control):
    
    x_curr_opt, u_curr_opt = control.mpc_casadi(q_np,p_np,x0_np,dx,du)

    sigzero_curr_opt = np.expand_dims(x_curr_opt[[0],0].repeat(mpc_T+1), 1)
    sigsiff_curr_opt = x_curr_opt[:,[0]]-x_curr_opt[0,0]

    x_curr_opt_plus = np.concatenate((
        x_curr_opt,sigzero_curr_opt,sigsiff_curr_opt), axis = 1)

    x_star = x_curr_opt_plus[:-1]
    u_star = u_curr_opt
    
    return x_star, u_star



def q_and_p(mpc_T, q_p_pred, Q_manual, p_manual):
    # Cost order: 
    # [for casadi] sigma_diff, d, phi, v, a, delta
    # [for model]  sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_pen, a, delta
    
    n_Q, BS, _ = q_p_pred.shape 
    
    q_p_pred = q_p_pred.repeat(mpc_T//n_Q, 1, 1)
    
    e = 1e-8
    
    q = e*torch.ones((mpc_T,BS,10)) + torch.tensor(Q_manual).unsqueeze(1).float()
    p = torch.zeros((mpc_T,BS,10)) + torch.tensor(p_manual).unsqueeze(1).float()

    #sigma_diff
    #q[:,:,5] = q_p_pred[:,:,0].clamp(e)
    p[:,:,5] = p[:,:,5] + q_p_pred[:,:,0]
    
    #d
    #q[:,:,1] = q_p_pred[:,:,1].clamp(e)
    p[:,:,1] = p[:,:,1] + q_p_pred[:,:,1]    
    
    return q, p





# I changed from 100 to 5 to try.
k_curve = 50.

dt = 0.04

mpc_T = 15
mpc_H = 30

n_Q = 5

assert mpc_T%n_Q==0

l_r = 0.2
l_f = 0.2

v_max = 2.0

delta_max = 0.6

a_max = 1.5

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

true_dx = FrenetKinBicycleDx(track_coord, params, 'cpu')



x0 = torch.tensor([0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = torch.tensor([0.0, 0.0])

dx=4
du=2

BS = 64
u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)
eps=0.01
lqr_iter = 50

grad_method = GradMethods.AUTO_DIFF

model = SimpleNN(mpc_H, n_Q, 2, max_p)
opt = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

control = CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0, 20, 5, 0, 0, 0, 0, 0, 0, 0]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -5, 0, 0, 0, 0]), 0), mpc_T, 0)

idx_to_casadi = [5,1,2,3,8,9] # This is only to match the indices of Q from model to casadi

x_star, u_star = solve_casadi(
            Q_manual[:,idx_to_casadi].T, p_manual[:,idx_to_casadi].T,
            x0.detach().numpy(),dx,du,control)

ind = np.array([0,1,3,4])

print(x_star[ind,:],u_star)

x_clamp = torch.clamp(torch.from_numpy(x_star[ind,:]),0.0,1.0)
print(x_clamp)
print(np.shape(x_star))

for it in range(200):

    x0 = sample_init(BS, true_dx)
    
    
    x0_diff = x0.clone()
    
    progress_pred = torch.tensor(0.)
    penalty_pred_d = torch.tensor(0.)
    penalty_pred_v = torch.tensor(0.)
    for sim in range(0, mpc_H//mpc_T):
        
        curv = get_curve_hor_from_x(x0_diff, track_coord, mpc_H)
        inp = torch.hstack((x0_diff[:,1:4], curv))
        q_p_pred = model(inp)

        q, p = q_and_p(mpc_T, q_p_pred, Q_manual, p_manual)
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
        #import pdb
        #pdb.set_trace()
        
    
    # It would be good if we could solve with casadi in batches
    #x_manual = np.zeros((mpc_T, BS, 6))
    #for bb in range(BS):
    #    x_star, u_star = solve_casadi(
    #        Q_manual[:,idx_to_casadi].T, p_manual[:,idx_to_casadi].T,
    #        x0[bb].detach().numpy(),dx,du,control)
    #    x_manual[:, bb] = x_star
    
    #progress = (progress_pred - torch.tensor(x_manual[-1,:,5]))
    loss = -progress_pred.mean() \
    + true_dx.penalty_d(penalty_pred_d).sum(0).mean() \
    + true_dx.penalty_v(penalty_pred_v).sum(0).mean()
    
    print(true_dx.penalty_d(penalty_pred_d).sum(0).mean().detach())
      
    opt.zero_grad()
    loss.backward()
    opt.step()  
    
    
    if it%5==0:
    # V A L I D A T I O N   (only casadi) 
        with torch.no_grad():

            BS_val = 32

            # This sampling should bring always the same set of initial states
            x0_val = sample_init(BS_val, true_dx, sn=0).numpy()

            x0_val_pred = x0_val[:,:6]
            x0_val_manual = x0_val[:,:6]

            progress_val_pred = 0.
            progress_val_manual = 0.

            for sim in range(mpc_H//mpc_T):

                x0_val_pred_torch = torch.tensor(x0_val_pred, dtype=torch.float32)
                curv_val = get_curve_hor_from_x(x0_val_pred_torch, track_coord, mpc_H)
                inp_val = torch.hstack((x0_val_pred_torch[:,1:4], curv_val))
                q_p_pred_val = model(inp_val)
                q_val, p_val = q_and_p(mpc_T, q_p_pred_val, Q_manual, p_manual)

                #print("COMPARE Q")
                #print(Q_manual.mean(0))
                #print(q_val.mean(1).mean(0))
                
                #print("COMPARE P")
                #print(p_manual.mean(0))
                #print(p_val.mean(1).mean(0))
                
                # It would be good if we could solve with casadi in batches
                # instead of going through the for loop
                x_pred_val = np.zeros((mpc_T, BS_val, 6))
                for bb in range(BS_val):

                    q_val_ = q_val[:,bb,idx_to_casadi].detach().numpy().T
                    p_val_ = p_val[:,bb,idx_to_casadi].detach().numpy().T
                    x_val, u_val = solve_casadi(q_val_, p_val_,
                        x0_val_pred[bb],dx,du,control)
                    x_pred_val[:,bb] = x_val

                x_manual = np.zeros((mpc_T, BS_val, 6))
                for bb in range(BS_val):
                    x_star, u_star = solve_casadi(
                        Q_manual[:,idx_to_casadi].T, p_manual[:,idx_to_casadi].T,
                        x0_val_manual[bb],dx,du,control)
                    x_manual[:, bb] = x_star

                progress_val_pred = progress_val_pred + x_pred_val[-1,:,5]
                progress_val_manual = progress_val_manual + x_manual[-1,:,5]
                
                x0_val_pred = x_pred_val[-1]
                x0_val_manual = x_manual[-1]
                
                x0_val_pred[:,4] = x0_val_pred[:,0]
                x0_val_manual[:,4] = x0_val_manual[:,0]
                
                x0_val_pred[:,5] = 0.
                x0_val_manual[:,5] = 0.            

            progress_val = progress_val_pred - progress_val_manual

        #print(round(progress.mean().item(), 3))    
        #print(q_p_pred.mean(0).mean(0))

        print(f'{it}: Progress: ', round(progress_val.mean(), 3))
        print(f'{it}: progress_val_pred: ', progress_val_pred[:4])
        print(f'{it}: progress_val_manual: ', progress_val_manual[:4])