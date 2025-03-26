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
from scipy import interpolate

from tqdm import tqdm

from casadi import *

import time



class FrenetDynBicycleDx(nn.Module):
    def __init__(self, track_coordinates, params, dev):
        super().__init__()

        # states: sigma, d, phi, r, v_x, v_y (6) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1)
        self.n_state = 6+2+1+1
        print(self.n_state)          # here add amount of states plus amount of exact penalty terms
        # control: a, delta
        self.n_ctrl = 2

        self.track_coordinates = track_coordinates.to(dev)

        # everything to calculate curvature
        self.track_sigma = self.track_coordinates[2,:]
        self.track_curv = self.track_coordinates[4,:]

        self.track_curv_shift = torch.empty(self.track_curv.size()).to(dev)
        self.track_curv_shift[1:] = self.track_curv[0:-1]
        self.track_curv_shift[0] = self.track_curv[-1]
        self.track_curv_diff = self.track_curv - self.track_curv_shift
        self.track_curv_diff[0] = 0.

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

        self.max_track_width_perc = 0.68

        # # model parameters: l_r, l_f (beta and curv(sigma) are calculated in the dynamics)
        # if params is None:
        #     # l_r, l_f
        #     self.params = Variable(torch.Tensor((0.2, 0.2)))
        # else:
        #     self.params = params
        #     assert len(self.params) == 2
        #
        #     self.delta_threshold_rad = np.pi  #12 * 2 * np.pi / 360
        #     self.v_max = 2
        #     self.max_acceleration = 2
        #
        #     self.dt = 0.05   # name T in document
        #
        #     self.track_width = 0.5
        #
        #     self.lower = -self.track_width/2
        #     self.upper = self.track_width/2
        #
        #     self.mpc_eps = 1e-4
        #     self.linesearch_decay = 0.5
        #     self.max_linesearch_iter = 2

    def curv(self, sigma):
        '''
        This function can stay the same
        '''

        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[0],1)


        sigma_shifted = sigma.reshape(-1,1) - sigma_f_mat
        curv_unscaled = torch.sigmoid(self.smooth_curve*sigma_shifted)
        curv = (curv_unscaled@(self.curv_f.reshape(-1,1))).type(torch.float)

        return curv.reshape(-1)


    def penalty_d(self, d):
        overshoot_pos = (d - 0.5*self.max_track_width_perc*self.track_width).clamp(min=0)
        overshoot_neg = (-d - 0.5*self.max_track_width_perc*self.track_width).clamp(min=0)
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
        softplus_op = torch.nn.Softplus(20)
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        lr = self.l_r # 0.038
        lf = self.l_f #0.052

        tau, delta = torch.unbind(u, dim=1)

        try:
            sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        except:
            sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff = torch.unbind(state, dim=1)

        # car params
        m = 0.181
        I_z = 0.000505

        # lateral force params
        Df = 0.65
        Cf = 1.5
        Bf = 5.2
        Dr = 1.0
        Cr = 1.45
        Br = 8.5

        # longitudinal force params
        Cm1 = 0.98028992
        Cm2 = 0.01814131
        Cd0 = 0.08518052
        Cd1 = 0.01
        Cd2 = 0.02750696
        gamma = 0.5


        a_f = (torch.atan2((- v_y - lf*r),torch.abs(v_x))+delta)
        a_r = (torch.atan2((-v_y + lr*r),torch.abs(v_x)))
        

        # forces on the wheels
        #F_x = (Cm1 - Cm2 * v_x) * tau - Cd * v_x * v_x - Croll  # motor force

        #forces on the wheels
        Fm = (Cm1 - Cm2 * v_x) * tau  # motor force
        Ffriction = torch.sign(v_x) * (-Cd0 - Cd1 * v_x - Cd2 * v_x * v_x)  # friction force

        Fx_f = Fm * (1 - gamma)  # front wheel force, x component
        Fx_r = Fm * gamma  # rear wheel force, x component

        Fy_f = Df*torch.sin(Cf*torch.atan(Bf*a_f))
        Fy_r = Dr*torch.sin(Cr*torch.atan(Br*a_r))


        dsigma = (v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d)
        dd = v_x*torch.sin(phi)+v_y*torch.cos(phi)
        dphi = r-self.curv(sigma)*((v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d))
        dr = 1/I_z*(Fy_f * lf * torch.cos(delta) + Fx_f * lf * torch.sin(delta) - Fy_r * lr)
        dv_x = 1/m*(Fx_r + Fx_f * torch.cos(delta) - Fy_f * torch.sin(delta) + m * v_y * r + Ffriction)
        dv_y = 1/m*(Fy_r + Fx_f * torch.sin(delta) + Fy_f * torch.cos(delta) - m * v_x * r)

        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        r = r + self.dt * dr
        v_x = v_x + self.dt * dv_x
        v_y = v_y + self.dt * dv_y
        sigma_0 = sigma_0
        sigma_diff = sigma - sigma_0

        d_pen = self.penalty_d(d)

        v_ub = self.penalty_v(v_x)

        state = torch.stack((sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub), 1)

        return state


    def forward_(self, state, u):
        softplus_op = torch.nn.Softplus(20)
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        lr = self.l_r # 0.038
        lf = self.l_f #0.052

        tau, delta = torch.unbind(u, dim=1)

        try:
            sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        except:
            sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff = torch.unbind(state, dim=1)

        # car params
        m = 0.181
        I_z = 0.000505

        # lateral force params
        Df = 0.65
        Cf = 1.5
        Bf = 5.2
        Dr = 1.0
        Cr = 1.45
        Br = 8.5

        # longitudinal force params
        Cm1 = 0.98028992
        Cm2 = 0.01814131
        Cd0 = 0.08518052
        Cd1 = 0.01
        Cd2 = 0.02750696
        gamma = 0.5


        a_f = (torch.atan2((- v_y - lf*r),torch.abs(v_x))+delta)
        a_r = (torch.atan2((-v_y + lr*r),torch.abs(v_x)))
        

        # forces on the wheels
        #F_x = (Cm1 - Cm2 * v_x) * tau - Cd * v_x * v_x - Croll  # motor force

        v = torch.sqrt(v_x**2 + v_y**2 + 1e-6)
        Fm = (Cm1 - Cm2 * v) * tau
        F_friction = -torch.sign(v_x) * (Cd0 + Cd1 * v + Cd2 * v**2)
        
        #forces on the wheels
        Fm = (Cm1 - Cm2 * v_x) * tau  # motor force
        Ffriction = torch.sign(v_x) * (-Cd0 - Cd1 * v_x - Cd2 * v_x * v_x)  # friction force

        Fx_f = Fm * (1 - gamma)  # front wheel force, x component
        Fx_r = Fm * gamma  # rear wheel force, x component

        Fy_f = Df*torch.sin(Cf*torch.atan(Bf*a_f))
        Fy_r = Dr*torch.sin(Cr*torch.atan(Br*a_r))

        curv_sigma = self.curv(sigma)
        denominator = torch.clamp(1. - curv_sigma*d, min=0.3)
    
        dsigma = (v_x * torch.cos(phi) - v_y * torch.sin(phi)) / denominator

        dd = v_x*torch.sin(phi)+v_y*torch.cos(phi)

        #curvature_effect = torch.where(torch.abs(d) < 0.03, torch.zeros_like(curv_sigma), curv_sigma)
        #dphi = r - curvature_effect * dsigma
        
        dphi = r-self.curv(sigma)*dsigma
        dr = 1/I_z*(Fy_f * lf * torch.cos(delta) + Fx_f * lf * torch.sin(delta) - Fy_r * lr)
        dv_x = 1/m*(Fx_r + Fx_f * torch.cos(delta) - Fy_f * torch.sin(delta) + m * v_y * r + Ffriction)
        dv_y = 1/m*(Fy_r + Fx_f * torch.sin(delta) + Fy_f * torch.cos(delta) - m * v_x * r)

        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        r = r + self.dt * dr
        v_x = v_x + self.dt * dv_x
        v_y = v_y + self.dt * dv_y
        sigma_0 = sigma_0
        sigma_diff = sigma - sigma_0

        d_pen = self.penalty_d(d)

        v_ub = self.penalty_v(v_x)

        state = torch.stack((sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub), 1)

        return state

    


    def forward_(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
    
        tau, delta = torch.unbind(u, dim=1)
    
        sigma, d, phi, r, v_x, v_y = torch.unbind(state[:, :6], dim=1)
    
        lr, lf = self.l_r, self.l_f
        m, I_z = 0.181, 0.000505
        Df, Cf, Bf = 0.65, 1.5, 5.2
        Dr, Cr, Br = 1.0, 1.45, 8.5
        Cm1, Cm2 = 0.98028992, 0.01814131
        Cd0, Cd1, Cd2 = 0.08518052, 0.01, 0.02750696
        gamma = 0.5
    
        a_f = delta - torch.atan2((v_y + lf * r), torch.abs(v_x))
        a_r = torch.atan2((-v_y + lr * r), torch.abs(v_x))
    
        # Forces
        v = torch.sqrt(v_x**2 + v_y**2 + 1e-6)
        Fm = (Cm1 - Cm2 * v) * tau
        F_friction = -torch.sign(v_x) * (Cd0 + Cd1 * v + Cd2 * v**2)
    
        Fx_f = Fm * (1 - gamma)
        Fx_r = Fm * gamma
    
        Fy_f = Df * torch.sin(Cf * torch.atan(Bf * a_f))
        Fy_r = Dr * torch.sin(Cr * torch.atan(Br * a_r))
    
        curv_sigma = self.curv(sigma)
        denominator = torch.clamp(1. - curv_sigma*d, min=0.3)
    
        dsigma = (v_x * torch.cos(phi) - v_y * torch.sin(phi)) / denominator
        dd = v_x * torch.sin(phi) + v_y * torch.cos(phi)
        

        curvature_effect = torch.where(torch.abs(d) < 0.03, torch.zeros_like(curv_sigma), curv_sigma)
        dphi = r - curvature_effect * dsigma
    
        dr = (lf * Fy_f * torch.cos(delta) + lf * Fx_f * torch.sin(delta) - lr * Fy_r) / I_z
        dv_x = (Fx_r + Fx_f * torch.cos(delta) - Fy_f * torch.sin(delta) + m * v_y * r + F_friction) / m
        dv_y = (Fy_r + Fx_f * torch.sin(delta) + Fy_f * torch.cos(delta) - m * v_x * r) / m
    
        # Euler integration
        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        r = r + self.dt * dr
        v_x = v_x + self.dt * dv_x
        v_y = v_y + self.dt * dv_y
    
        next_state = torch.stack((sigma, d, phi, r, v_x, v_y), dim=1)
    
        if squeeze:
            next_state = next_state.squeeze(0)
    
        return next_state






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

        self.track_curv_np = self.track_coordinates[4,:].numpy()

        self.track_curv_shift = torch.empty(self.track_curv.size())
        self.track_curv_shift[1:] = self.track_curv[0:-1]
        self.track_curv_shift[0] = self.track_curv[-1]
        self.track_curv_diff = self.track_curv - self.track_curv_shift
        self.track_curv_diff[0] = 0.

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

        self.max_track_width_perc = 0.68

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


    def mpc_casadi(self,q,p,x0,dx,du, u0=np.array([0.,0.])):


        # here the q and the p scale the following
        # feature vector [sigma-sigma_0, d, phi, v, penalty_d,penalty_v,a,delta]


        N=self.mpc_T
        l_r = self.l_r
        l_f = self.l_f

        Ts = self.dt

        # car params
        m = 0.181
        I_z = 0.000505

        # lateral force params
        Df = 0.65
        Cf = 1.5
        Bf = 5.2
        Dr = 1.0
        Cr = 1.45
        Br = 8.5

        # longitudinal force params
        Cm1 = 0.98028992
        Cm2 = 0.01814131
        Cd0 = 0.08518052
        Cd1 = 0.01
        Cd2 = 0.02750696
        gamma = 0.5

        x_sym = SX.sym('x_sym',dx,N+1)
        u_sym = SX.sym('u_sym',du,N)

        a_f = (np.arctan2((-x_sym[5,0:N] - l_f*x_sym[3,0:N]),((x_sym[4,0:N])+0.00001))+u_sym[1,0:N])
        a_r = (np.arctan2((-x_sym[5,0:N] + l_r*x_sym[3,0:N]),((x_sym[4,0:N])+0.00001)))

        # forces on the wheels
        #F_x = (Cm1 - Cm2 * x_sym[4,0:N]) * u_sym[0,0:N] - Cd * x_sym[4,0:N]* x_sym[4,0:N] - Croll  # motor force

        #forces on the wheels
        Fm = (Cm1 - Cm2 * x_sym[4,0:N]) * u_sym[0,0:N]  # motor force
        Ffriction = sign(x_sym[4,0:N]) * (-Cd0 - Cd1 * x_sym[4,0:N] - Cd2 * x_sym[4,0:N] * x_sym[4,0:N])  # friction force

        Fx_f = Fm * (1 - gamma)  # front wheel force, x component
        Fx_r = Fm * gamma  # rear wheel force, x component

        Fy_f = Df*np.sin(Cf*np.arctan(Bf*a_f))
        Fy_r = Dr*np.sin(Cr*np.arctan(Br*a_r))

        #solver parameters
        options = {}
        options['ipopt.max_iter'] = 2000
        options['verbose'] = False

        denominator = (1.-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])
        
        dyn1 = horzcat(
            (x_sym[0,0] - x0[0]),
            (x_sym[0,1:N+1] - x_sym[0,0:N] - \
             Ts*((x_sym[4,0:N]*cos(x_sym[2,0:N])-x_sym[5,0:N]*sin(
                x_sym[2,0:N]))/denominator)))

        dyn2 = horzcat(
            (x_sym[1,0] - x0[1]),
            (x_sym[1,1:N+1] - x_sym[1,0:N] - \
             Ts*(x_sym[4,0:N]*sin(x_sym[2,0:N])+x_sym[5,0:N]*cos(
                x_sym[2,0:N]))))

        dyn3 = horzcat(
            (x_sym[2,0] - x0[2]),
            (x_sym[2,1:N+1] - x_sym[2,0:N] - \
             Ts*(x_sym[3,0:N] - self.curv_casadi(
                x_sym[0,0:N])*(x_sym[4,0:N]*cos(
                x_sym[2,0:N])-x_sym[5,0:N]*sin(x_sym[2,0:N]))/denominator)))

        dyn4 = horzcat(
            (x_sym[3,0] - x0[3]),
            (x_sym[3,1:N+1] - x_sym[3,0:N] - \
             Ts*((1/I_z)*(Fy_f * l_f *cos(u_sym[1,0:N]) + Fx_f * l_f * sin(u_sym[1,0:N])- Fy_r * l_r))))

        dyn5 = horzcat(
            (x_sym[4,0] - x0[4]),
            (x_sym[4,1:N+1] - x_sym[4,0:N] - \
             Ts*(1/m)*(Fx_r + Fx_f *cos(u_sym[1,0:N]) - Fy_f *sin(u_sym[1,0:N]) + m *x_sym[5,0:N]* x_sym[3,0:N] + Ffriction)))

        dyn6 = horzcat(
            (x_sym[5,0] - x0[5]),
            (x_sym[5,1:N+1] - x_sym[5,0:N] - \
             Ts*1/m*(Fy_r + Fx_f * sin(u_sym[1,0:N]) + Fy_f * cos(u_sym[1,0:N]) - m *x_sym[4,0:N]* x_sym[3,0:N])))

        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0],x_sym[1:,0:N],u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+du,N)
        p_sym = SX.sym('p_sym',dx+du,N)
        Q_sym = diag(q_sym)

        #import pdb
        #pdb.set_trace()

        tsh = 0.45*self.max_track_width_perc*self.track_width*np.ones((1,N+1))

        #barr1 = if_else(x_sym[1,0:N+1] > tsh, 1000.*np.ones((1,N+1)), 0*np.ones((1,N+1)))
        #barr2 = if_else(x_sym[1,0:N+1] < -tsh, 1000.*np.ones((1,N+1)), 0*np.ones((1,N+1)))
        
        barr1 = if_else(x_sym[1,0:N+1] > tsh, exp(500 * (x_sym[1,0:N+1] - tsh)) - 1, 0)
        barr2 = if_else(x_sym[1,0:N+1] < -tsh, exp(500 * (-tsh - x_sym[1,0:N+1])) - 1, 0)
        
        #barr1 = -log(fmax(1e-4, x_sym[1,0:N+1] + 0.48*self.max_track_width_perc*self.track_width*np.ones((1,N+1))))
        #barr2 = -log(fmax(1e-4, 0.48*self.max_track_width_perc*self.track_width*np.ones((1,N+1)) - x_sym[1,0:N+1]))
        
        barrier = (1/N)* (barr1 + barr2)

        
        l = sum2(sum1(0.5*q_sym*feat*feat + p_sym*feat)) + sum2(sum1(barrier))
        dl = substitute(substitute(l,q_sym,q),p_sym,p)

        const = vertcat(
                transpose(dyn1),
                transpose(dyn2),
                transpose(dyn3),
                transpose(dyn4),
                transpose(dyn5),
                transpose(dyn6),
                transpose(u_sym[0,0:N]),
                transpose(u_sym[1,0:N]),
                transpose(x_sym[1,0:N+1]),
                transpose(x_sym[4,0:N+1]))

        lbg = np.r_[np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    -self.a_max*np.ones(N),
                    -self.delta_max*np.ones(N),
                    -0.5*self.max_track_width_perc*self.track_width*np.ones(N+1),
                    0.1*np.ones(N+1)]

        ubg = np.r_[np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    self.a_max*np.ones(N),
                    self.delta_max*np.ones(N),
                    0.5*self.max_track_width_perc*self.track_width*np.ones(N+1),
                    self.v_max*np.ones(N+1)]

        lbx = -np.inf * np.ones(dx*(N+1)+du*N)
        ubx = np.inf * np.ones(dx*(N+1)+du*N)

        x = vertcat(reshape(x_sym[:,0:N+1],(dx*(N+1),1)),reshape(u_sym[:,0:N],(du*N,1)))
        #w_ws = np.vstack([np.reshape(x_warmstart[:dx,0:N+1],(dx*(N+1),1)),np.reshape(x_warmstart[dx+dc+df:,0:N],(du*(N),1))])

        #options = {
        #            'verbose': False,
        #            'ipopt.print_level': 0,
        #            'print_time': False,
        #            'ipopt.sb': 'yes',
        #            'print_time': 0,
        #            'ipopt.constr_viol_tol': 1e-9,
        #            'ipopt.tol': 1e-6,
        #            'ipopt.max_iter': 1000,
        #            'ipopt.hessian_approximation': 'exact',
        #            'ipopt.mu_init': 1e4,
        #            'ipopt.mu_min': 1e-5,
        #            'ipopt.mu_max': 1e4
        #        }

        options = {
            'verbose': False,
            'ipopt.print_level': 0,
            'print_time': False,
            'ipopt.sb': 'yes',
            'ipopt.constr_viol_tol': 1e-8,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 5e-1,
            'ipopt.acceptable_constr_viol_tol': 1e-4,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.mu_init': 1e-1,
            'ipopt.mu_min': 1e-4,
            'ipopt.max_iter': 1500,
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.hessian_approximation': 'exact'
        }

        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        # create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        #x_warm_full = np.zeros(dx*(N+1)+du*N)

        # Insert your u0 guess at the correct position:
        #x_warm_full[-du*N:] = u0
        #solver_input['x0'] = x_warm_full

        # add initial guess to solver
        #solver_input['x0'] = w_ws

        # solve optimization problem
        solver_output = solver(**solver_input)

        # process ouput
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*N:].reshape(-1,du)
        x = sol_evalf[:-du*N].reshape(-1,dx)

        optimal_status = 0
        status = solver.stats()['return_status']
        if status == 'Solve_Succeeded':
            optimal_status = 1
            
        print("IPOPT status:", status)
        
        return x, u#, optimal_status


def sample_init(BS, dyn, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000
    sigma_sample = torch.randint(int(0.0*di), int(16.0*di), (BS,1), generator=gen)/di
    d_sample = torch.randint(int(-0.14*di), int(0.14*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.5*di), int(0.5*di), (BS,1), generator=gen)/di
    r_sample = torch.randint(int(-0.02*di), int(0.02*di), (BS,1), generator=gen)/di
    vx_sample = torch.randint(int(0.1*di), int(1.8*di), (BS,1), generator=gen)/di
    vy_sample = torch.randint(int(-0.05*di), int(0.05*di), (BS,1), generator=gen)/di

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(vx_sample)

    x_init_sample = torch.hstack((
        sigma_sample, d_sample, phi_sample, r_sample, vx_sample, vy_sample,
        sigma_sample, sigma_diff_sample, d_pen, v_pen))

    return x_init_sample


def sample_init_test(BS, dyn, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000
    sigma_sample = torch.zeros((BS,1))
    d_sample = torch.randint(int(-0.01*di), int(0.01*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.002*di), int(0.002*di), (BS,1), generator=gen)/di
    r_sample = torch.ones((BS,1))*0.0
    vx_sample = torch.ones((BS,1))*0.0
    vy_sample = torch.ones((BS,1))*0.0

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(vx_sample)

    x_init_sample = torch.hstack((
        sigma_sample, d_sample, phi_sample, r_sample, vx_sample, vy_sample,
        sigma_sample, sigma_diff_sample, d_pen, v_pen))

    return x_init_sample



def sample_init_traj_dist(BS, dyn, traj, num_patches, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000

    traj_steps = np.shape(traj)
    #print('traj_steps:',traj_steps)
    patch_steps = np.floor(traj_steps[0]/num_patches)

    traj_ind_sample = torch.zeros([BS,1]).int()
    for i in range(num_patches):
        traj_ind_sample[i*int(BS/num_patches):(i+1)*int(BS/num_patches)] = torch.randint(int(patch_steps*i),int(patch_steps*(i+1)),(int(BS/num_patches),1), generator=gen)
    traj_sample = traj[traj_ind_sample.detach().numpy().flatten(),:]

    d_sample = torch.clamp(torch.from_numpy(traj_sample[:,1].reshape(-1,1))+torch.randint(int(-.02*di), int(.02*di), (BS,1), generator=gen)/di,-0.16,0.16)
    phi_sample = torch.from_numpy(traj_sample[:,2].reshape(-1,1))+torch.randint(int(-0.05*di), int(0.05*di), (BS,1), generator=gen)/di
    r_sample = torch.from_numpy(traj_sample[:,3].reshape(-1,1))+torch.randint(int(-0.001*di), int(0.001*di), (BS,1), generator=gen)/di  # currently fixed
    vx_sample = torch.clamp(torch.from_numpy(traj_sample[:,4].reshape(-1,1))+torch.randint(int(-0.3*di), int(0.1*di), (BS,1), generator=gen)/di,0.07,1.8)
    vy_sample = torch.from_numpy(traj_sample[:,5].reshape(-1,1))+torch.randint(int(-0.00*di), int(0.001*di), (BS,1), generator=gen)/di # currently fixed

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(vx_sample)

    x_init_sample = torch.hstack((
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), d_sample, phi_sample, r_sample, vx_sample, vy_sample,
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), sigma_diff_sample, d_pen, v_pen))

    return x_init_sample



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



def solve_casadi(q_np,p_np,x0_np,dx,du,control,u0=np.array([0,0])):

    mpc_T = q_np.shape[1]

    #x_curr_opt, u_curr_opt, op = control.mpc_casadi(q_np,p_np,x0_np,dx,du,u0)
    x_curr_opt, u_curr_opt = control.mpc_casadi(q_np,p_np,x0_np,dx,du,u0)

    sigzero_curr_opt = np.expand_dims(x_curr_opt[[0],0].repeat(mpc_T+1), 1)
    sigsiff_curr_opt = x_curr_opt[:,[0]]-x_curr_opt[0,0]

    x_curr_opt_plus = np.concatenate((
        x_curr_opt,sigzero_curr_opt,sigsiff_curr_opt), axis = 1)

    x_star = x_curr_opt_plus[:-1]
    u_star = u_curr_opt

    return x_star, u_star#, op

def process_single_casadi(sample, q, p, x0, dx, du, control):
    x, u = solve_casadi(
        q[:,sample], p[:,sample],
        x0[sample], dx, du, control)
    return sample, x, u

def solve_casadi_parallel(q, p, x0, BS, dx, du, control):
    x = np.zeros((q.shape[2],q.shape[1],x0.shape[-1]))
    u = np.zeros((q.shape[2],q.shape[1],du))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            process_single_casadi,
            sample, q, p, x0, dx, du, control) for sample in range(BS)]

        for future in futures:
            sample, x_sample, u_sample = future.result()
            x[:, sample] = x_sample
            u[:, sample] = u_sample

    return x, u




def q_and_p(mpc_T, q_p_pred, Q_manual, p_manual):

    n_Q, BS, _ = q_p_pred.shape

    q_p_pred = q_p_pred.repeat_interleave(mpc_T//n_Q, dim=0)

    e = 1e-9

    q = e*torch.ones((mpc_T,BS,12)) + torch.tensor(Q_manual).unsqueeze(1).float()
    p = torch.zeros((mpc_T,BS,12)) + torch.tensor(p_manual).unsqueeze(1).float()

    #sigma_diff
    p[:,:,7] = p[:,:,7] + q_p_pred[:,:,0]

    #d
    p[:,:,1] = p[:,:,1] + q_p_pred[:,:,1]
    #q[:,:,1] = q[:,:,1] + q_p_pred[:,:,2]

    #phi
    p[:,:,2] = p[:,:,2] + q_p_pred[:,:,2]
    #q[:,:,2] = q[:,:,2] + q_p_pred[:,:,4]

    #a
    p[:,:,10] = p[:,:,10] + q_p_pred[:,:,3]

    #delta
    p[:,:,11] = p[:,:,11] + q_p_pred[:,:,4]
    #q[:,:,11] = q[:,:,11] + q_p_pred[:,:,7]

    return q, p