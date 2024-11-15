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
        p_sym = SX.sym('p_sym',dx+du,mpc_T)
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
                    -0.5*self.max_track_width_perc*self.track_width*np.ones(mpc_T+1),
                    -0.1*np.ones(mpc_T+1)]

        ubg = np.r_[np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    np.zeros(mpc_T+1),
                    self.a_max*np.ones(mpc_T),
                    self.delta_max*np.ones(mpc_T),
                    0.5*self.max_track_width_perc*self.track_width*np.ones(mpc_T+1),
                    self.v_max*np.ones(mpc_T+1)]


        lbx = -np.inf * np.ones(dx*(mpc_T+1)+du*mpc_T)
        ubx = np.inf * np.ones(dx*(mpc_T+1)+du*mpc_T)

        x = vertcat(reshape(x_sym[:,0:mpc_T+1],(dx*(mpc_T+1),1)),
                    reshape(u_sym[:,0:mpc_T],(du*mpc_T,1)))

        options = {
                    'verbose': False,
                    'ipopt.print_level': 0,
                    'print_time': False,
                    'ipopt.sb': 'yes',
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

    def mpc_casadi_dyn(self,q,p,x0,dx,du):


        # here the q and the p scale the following
        # feature vector [sigma-sigma_0, d, phi, v, penalty_d,penalty_v,a,delta]


        N=self.mpc_T
        l_r = self.l_r
        l_f = self.l_f

        Ts = self.dt

        # car params
        m = 0.200
        I_z = 0.0004

        # lateral force params
        Df = 0.43
        Cf = 1.4
        Bf = 0.5
        Dr = 0.6
        Cr = 1.7
        Br = 0.5

        # longitudinal force params
        Cm1 = 0.98028992
        Cm2 = 0.01814131
        Cd = 0.02750696
        Croll = 0.08518052

        x_sym = SX.sym('x_sym',dx,N+1)
        u_sym = SX.sym('u_sym',du,N)

        a_f = -(np.arctan2((-x_sym[5,0:N] - l_f*x_sym[3,0:N]),((x_sym[4,0:N])+0.00001))+u_sym[1,0:N])
        a_r = -(np.arctan2((-x_sym[5,0:N] + l_r*x_sym[3,0:N]),((x_sym[4,0:N])+0.00001)))

        # forces on the wheels
        F_x = (Cm1 - Cm2 * x_sym[4,0:N]) * u_sym[0,0:N] - Cd * x_sym[4,0:N]* x_sym[4,0:N] - Croll  # motor force

        F_f = -Df*np.sin(Cf*np.arctan(Bf*a_f))
        F_r = -Dr*np.sin(Cr*np.arctan(Br*a_r))

        #solver parameters
        options = {}
        options['ipopt.max_iter'] = 2000
        options['verbose'] = False

        dyn1 = horzcat(
            (x_sym[0,0] - x0[0]),
            (x_sym[0,1:N+1] - x_sym[0,0:N] - \
             Ts*((x_sym[4,0:N]*cos(x_sym[2,0:N])-x_sym[5,0:N]*sin(
                x_sym[2,0:N]))/(1.-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N]))))

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
                x_sym[2,0:N])-x_sym[5,0:N]*sin(x_sym[2,0:N]))/(1-self.curv_casadi(
                x_sym[0,0:N])*x_sym[1,0:N]))))

        dyn4 = horzcat(
            (x_sym[3,0] - x0[3]),
            (x_sym[3,1:N+1] - x_sym[3,0:N] - \
             Ts*(1/I_z*(F_f * l_f *cos(u_sym[1,0:N])- F_r * l_r))))

        dyn5 = horzcat(
            (x_sym[4,0] - x0[4]),
            (x_sym[4,1:N+1] - x_sym[4,0:N] - \
             Ts*1/m*(F_x - F_f *sin(u_sym[1,0:N]) + m *x_sym[5,0:N]* x_sym[3,0:N])))

        dyn6 = horzcat(
            (x_sym[5,0] - x0[5]),
            (x_sym[5,1:N+1] - x_sym[5,0:N] - \
             Ts*1/m*(F_r + F_f * cos(u_sym[1,0:N]) - m *x_sym[4,0:N]* x_sym[3,0:N])))

        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0],x_sym[1:,0:N],u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+du,N)
        p_sym = SX.sym('p_sym',dx+du,N)
        Q_sym = diag(q_sym)

        l = sum2(sum1(0.5*q_sym*feat*feat + p_sym*feat))
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

        options = {
                    'verbose': False,
                    'ipopt.print_level': 0,
                    'print_time': False,
                    'ipopt.sb': 'yes',
                    'print_time': 0,
                    'ipopt.tol': 1e-2,
                    'ipopt.max_iter': 400,
                    'ipopt.hessian_approximation': 'limited-memory'
                }

        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        # create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        # add initial guess to solver
        #solver_input['x0'] = w_ws

        # solve optimization problem
        solver_output = solver(**solver_input)

        # process ouput
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*N:].reshape(-1,du)
        x = sol_evalf[:-du*N].reshape(-1,dx)

        #print(sol.solveroutput.info.lambda)

        # print solution
        return x, u


class SimpleNN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K):
        super(SimpleNN, self).__init__()
        input_size = 3 + mpc_H
        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, mpc_T*O)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.K = K
        self.O = O
        self.mpc_T = mpc_T

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        #x = self.output_activation(x) * self.K
        x = x.reshape(self.mpc_T, -1, self.O)
        x = 5*self.output_activation(x/10)
        return x


class ImprovedNN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K):
        super(ImprovedNN, self).__init__()
        input_size = 3  # For the global context variables
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Adding a temporal conv layer
        self.fc1 = nn.Linear(16 * mpc_H + input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, mpc_T * O)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.K = K
        self.O = O
        self.mpc_T = mpc_T

    def forward(self, x):
        global_context, time_series = x[:, :3], x[:, 3:]
        time_series = time_series.unsqueeze(1)  # For Conv1D input [batch_size, channels, seq_len]
        time_series = self.activation(self.conv1(time_series)).view(time_series.size(0), -1)

        x = torch.cat([time_series, global_context], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = x.reshape(self.mpc_T, -1, self.O)
        return x/5



class ImprovedNN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K):
        super(ImprovedNN, self).__init__()
        input_size = 3

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(16 * mpc_H + input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, mpc_T * O)
        self.activation = nn.LeakyReLU(0.1)
        self.output_activation = nn.Tanh()
        self.K = K
        self.O = O
        self.mpc_T = mpc_T


    def forward(self, x):
        global_context, time_series = x[:, :3], x[:, 3:]

        time_series = time_series.unsqueeze(1)

        time_series_res = time_series
        time_series = self.activation(self.conv1(time_series))
        time_series = self.bn1(time_series)
        time_series = self.dropout(time_series)
        time_series += time_series_res

        time_series = time_series.view(time_series.size(0), -1)

        x = torch.cat([time_series, global_context], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        x = x.reshape(self.mpc_T, -1, self.O)
        x = 5*self.output_activation(x/10)
        return x


import torch.nn.functional as F
class TCN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K, num_channels=[1, 25, 50, 50, 25], kernel_size=3):
        super(TCN, self).__init__()
        self.K = K
        self.O = O
        self.mpc_T = mpc_T
        
        layers = []
        for i in range(len(num_channels) - 1):
            layers.append(nn.Conv1d(num_channels[i], num_channels[i+1], kernel_size, padding='same', dilation=2**i))
            layers.append(nn.BatchNorm1d(num_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.tcn = nn.Sequential(*layers)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(num_channels[-1] * mpc_H + 3, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, mpc_T * O)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        global_context, time_series = x[:, :3], x[:, 3:]

        time_series = time_series.unsqueeze(1)
        time_series = self.tcn(time_series)
        time_series = time_series.view(time_series.size(0), -1)

        x = torch.cat([time_series, global_context], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.reshape(self.mpc_T, -1, self.O)
        x = 8 * self.output_activation(x / 10)
        return x




def sample_init(BS, dyn, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000
    sigma_sample = torch.randint(int(0.0*di), int(14.5*di), (BS,1), generator=gen)/di
    d_sample = torch.randint(int(-0.09*di), int(0.09*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.09*di), int(0.09*di), (BS,1), generator=gen)/di
    v_sample = torch.randint(int(.5*di), int(1.8*di), (BS,1), generator=gen)/di

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)

    x_init_sample = torch.hstack((
        sigma_sample, d_sample, phi_sample, v_sample,
        sigma_sample, sigma_diff_sample, d_pen, v_pen))

    return x_init_sample

def sample_init_dyn(BS, dyn, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000
    sigma_sample = torch.randint(int(0.0*di), int(14.5*di), (BS,1), generator=gen)/di
    d_sample = torch.randint(int(-0.06*di), int(0.06*di), (BS,1), generator=gen)/di
    phi_sample = torch.randint(int(-0.06*di), int(0.06*di), (BS,1), generator=gen)/di
    r_sample = torch.randint(int(-0.01*di), int(0.01*di), (BS,1), generator=gen)/di
    vx_sample = torch.randint(int(1.0*di), int(1.5*di), (BS,1), generator=gen)/di
    vy_sample = torch.randint(int(0.0*di), int(0.03*di), (BS,1), generator=gen)/di

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
    v_sample = torch.zeros((BS,1))

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)

    x_init_sample = torch.hstack((
        sigma_sample, d_sample, phi_sample, v_sample,
        sigma_sample, sigma_diff_sample, d_pen, v_pen))

    return x_init_sample

def sample_init_test_dyn(BS, dyn, sn=None):

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

    d_sample = torch.clamp(torch.from_numpy(traj_sample[:,1].reshape(-1,1))+torch.randint(int(-.02*di), int(.02*di), (BS,1), generator=gen)/di,-0.24,0.24)
    phi_sample = torch.from_numpy(traj_sample[:,2].reshape(-1,1))+torch.randint(int(-0.01*di), int(0.01*di), (BS,1), generator=gen)/di
    v_sample = torch.clamp(torch.from_numpy(traj_sample[:,3].reshape(-1,1))+torch.randint(int(-0.01*di), int(.01*di), (BS,1), generator=gen)/di,0.0,2.5)

    # and this part we can actually keep

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)

    x_init_sample = torch.hstack((
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), d_sample, phi_sample, v_sample,
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), sigma_diff_sample, d_pen, v_pen))

    return x_init_sample

def sample_init_traj_dist(BS, dyn, traj, num_patches, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000

    # The idea is to take the trajectory and divide it into num_patches patches.
    # We then sample BS / num_patches samples from each batch such that already
    # in the first training step we the whole track and will not be surprised
    # by a curve later on.

    traj_steps = np.shape(traj)
    #print('traj_steps:',traj_steps)
    patch_steps = np.floor(traj_steps[0]/num_patches)

    # in this step we randomly
    traj_ind_sample = torch.zeros([BS,1]).int()
    for i in range(num_patches):
        traj_ind_sample[i*int(BS/num_patches):(i+1)*int(BS/num_patches)] = torch.randint(int(patch_steps*i),int(patch_steps*(i+1)),(int(BS/num_patches),1), generator=gen)
    #print('traj_ind_sample:',traj_ind_sample)
    traj_sample = traj[traj_ind_sample.detach().numpy().flatten(),:]
    #print('traj_sample:',traj_sample)

    # now we keep sigma as it is and sample d, phi, and v in a small "tube" around the traj points

    # Note that we clamp the sampled d and v values to stay in their constraints. Sampling constraint violating
    # states would not make sense.

    d_sample = torch.clamp(torch.from_numpy(traj_sample[:,1].reshape(-1,1))+torch.randint(int(-.04*di), int(.04*di), (BS,1), generator=gen)/di,-0.17,0.17)
    phi_sample = torch.from_numpy(traj_sample[:,2].reshape(-1,1))+torch.randint(int(-0.05*di), int(0.05*di), (BS,1), generator=gen)/di
    v_sample = torch.clamp(torch.from_numpy(traj_sample[:,3].reshape(-1,1))+torch.randint(int(-0.2*di), int(0.2*di), (BS,1), generator=gen)/di,0.0,1.8)

    # and this part we can actually keep

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(v_sample)

    x_init_sample = torch.hstack((
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), d_sample, phi_sample, v_sample,
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), sigma_diff_sample, d_pen, v_pen))

    return x_init_sample

def sample_init_traj_dist_dyn(BS, dyn, traj, num_patches, sn=None):

    # If sn!=None, we makesure that we always sample the same set of initial states
    # We need that for validation to understand if our model is improving or not

    gen=None
    if sn != None:
        gen = torch.Generator()
        gen.manual_seed(sn)

    di = 1000

    # The idea is to take the trajectory and divide it into num_patches patches.
    # We then sample BS / num_patches samples from each batch such that already
    # in the first training step we the whole track and will not be surprised
    # by a curve later on.

    traj_steps = np.shape(traj)
    #print('traj_steps:',traj_steps)
    patch_steps = np.floor(traj_steps[0]/num_patches)

    # in this step we randomly
    traj_ind_sample = torch.zeros([BS,1]).int()
    for i in range(num_patches):
        traj_ind_sample[i*int(BS/num_patches):(i+1)*int(BS/num_patches)] = torch.randint(int(patch_steps*i),int(patch_steps*(i+1)),(int(BS/num_patches),1), generator=gen)
    #print('traj_ind_sample:',traj_ind_sample)
    traj_sample = traj[traj_ind_sample.detach().numpy().flatten(),:]
    #print('traj_sample:',traj_sample)

    # now we keep sigma as it is and sample d, phi, and v in a small "tube" around the traj points

    # Note that we clamp the sampled d and v values to stay in their constraints. Sampling constraint violating
    # states would not make sense.

    d_sample = torch.clamp(torch.from_numpy(traj_sample[:,1].reshape(-1,1))+torch.randint(int(-.01*di), int(.01*di), (BS,1), generator=gen)/di,-0.17,0.17)
    phi_sample = torch.from_numpy(traj_sample[:,2].reshape(-1,1))+torch.randint(int(-0.02*di), int(0.02*di), (BS,1), generator=gen)/di
    r_sample = torch.from_numpy(traj_sample[:,3].reshape(-1,1))+torch.randint(int(-0.001*di), int(0.001*di), (BS,1), generator=gen)/di  # currently fixed
    vx_sample = torch.clamp(torch.from_numpy(traj_sample[:,4].reshape(-1,1))+torch.randint(int(-0.1*di), int(0.1*di), (BS,1), generator=gen)/di,0.5,1.5)
    vy_sample = torch.from_numpy(traj_sample[:,5].reshape(-1,1))+torch.randint(int(-0.00*di), int(0.001*di), (BS,1), generator=gen)/di # currently fixed

    # and this part we can actually keep

    sigma_diff_sample = torch.zeros((BS,1))

    d_pen = dyn.penalty_d(d_sample)
    v_pen = dyn.penalty_v(vx_sample)

    x_init_sample = torch.hstack((
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), d_sample, phi_sample, r_sample, vx_sample, vy_sample,
        torch.from_numpy(traj_sample[:,0].reshape(-1,1)), sigma_diff_sample, d_pen, v_pen))

    return x_init_sample

def get_curve_hor_from_x(x, track_coord, H_curve):
    idx_track_batch = ((x[:,0]-track_coord[[2],:].T)**2).argmin(0)
    idcs_track_batch = idx_track_batch[:, None] + torch.arange(H_curve)
    try:
        idcs_track_batch = torch.clip(idcs_track_batch, None, track_coord.shape[1]-1)
        curvs = track_coord[4,idcs_track_batch].float()
    except:
        import pdb
        pdb.set_trace()
    return curvs

class FrenetKinBicycleDx(nn.Module):
    def __init__(self, track_coordinates, params, dev):
        super().__init__()

        self.params = params

        # states: sigma, d, phi, v (4) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1)
        self.n_state = 4+2+1+1
        #print('Number of states:', self.n_state)

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

        self.factor_pen = 10000000.

        self.max_track_width_perc = 0.68



    def curv(self, sigma):

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
        overshoot_neg = (-v - 0.001).clamp(min=0)
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

        try:
            sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        except:
            sigma, d, phi, v, sigma_0, sigma_diff = torch.unbind(state, dim=1)

        #beta = torch.atan(self.l_r/(self.l_r+self.l_f)*torch.tan(delta))
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

        #v = torch.clamp(v, 0, self.v_max)

        sigma_diff = sigma - sigma_0

        d_pen = self.penalty_d(d)
        v_ub = self.penalty_v(v)

        state = torch.stack((sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub), 1)

        return state

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
        softplus_op = torch.nn.Softplus(20)
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        lr = self.l_r
        lf = self.l_f

        tau, delta = torch.unbind(u, dim=1)

        sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)

        # car params
        m = 0.200
        I_z = 0.0004

        # lateral force params
        Df = 0.43
        Cf = 1.4
        Bf = 0.5
        Dr = 0.6
        Cr = 1.7
        Br = 0.5

        # longitudinal force params
        Cm1 = 0.98028992
        Cm2 = 0.01814131
        Cd = 0.02750696
        Croll = 0.08518052

        a_f = -(torch.atan2((- v_y - lf*r),torch.abs(v_x))+delta)
        a_r = -(torch.atan2((-v_y + lr*r),torch.abs(v_x)))


        # forces on the wheels
        F_x = (Cm1 - Cm2 * v_x) * tau - Cd * v_x * v_x - Croll  # motor force

        F_f = -Df*torch.sin(Cf*torch.atan(Bf*a_f))
        F_r = -Dr*torch.sin(Cr*torch.atan(Br*a_r))


        dsigma = (v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d)
        dd = v_x*torch.sin(phi)+v_y*torch.cos(phi)
        dphi = r-self.curv(sigma)*((v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d))
        dr = 1/I_z*(F_f * lf * torch.cos(delta) - F_r * lr)
        dv_x = 1/m*(F_x - F_f * torch.sin(delta) + m * v_y * r)
        dv_y = 1/m*(F_r + F_f * torch.cos(delta) - m * v_x * r)

        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        r = r + self.dt * dr
        v_x = v_x + self.dt * dv_x
        v_y = v_y + self.dt * dv_y
        sigma_0 = sigma_0                   # we need to carry it on
        sigma_diff = sigma - sigma_0

        d_pen = self.penalty_d(d)

        v_ub = self.penalty_v(v_x)

        state = torch.stack((sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub), 1)

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

def solve_casadi_dyn(q_np,p_np,x0_np,dx,du,control):

    mpc_T = q_np.shape[1]

    x_curr_opt, u_curr_opt = control.mpc_casadi_dyn(q_np,p_np,x0_np,dx,du)

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

def process_single_casadi_dyn(sample, q, p, x0, dx, du, control):
    x, u = solve_casadi_dyn(
        q[:,sample], p[:,sample],
        x0[sample], dx, du, control)
    return sample, x, u

def solve_casadi_parallel_dyn(q, p, x0, BS, dx, du, control):
    x = np.zeros((q.shape[2],q.shape[1],x0.shape[-1]))
    u = np.zeros((q.shape[2],q.shape[1],du))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            process_single_casadi_dyn,
            sample, q, p, x0, dx, du, control) for sample in range(BS)]

        for future in futures:
            sample, x_sample, u_sample = future.result()
            x[:, sample] = x_sample
            u[:, sample] = u_sample

    return x, u


def q_and_p(mpc_T, q_p_pred, Q_manual, p_manual):
    # Cost order:
    # [for casadi] sigma_diff, d, phi, v, a, delta
    # [for model]  sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_pen, a, delta

    n_Q, BS, _ = q_p_pred.shape

    q_p_pred = q_p_pred.repeat_interleave(mpc_T//n_Q, dim=0)

    e = 1e-9

    q = e*torch.ones((mpc_T,BS,10)) + torch.tensor(Q_manual).unsqueeze(1).float()
    p = torch.zeros((mpc_T,BS,10)) + torch.tensor(p_manual).unsqueeze(1).float()

    #sigma_diff
    #q[:,:,5] = q[:,:,5] + q_p_pred[:,:,0].clamp(e)
    #p[:,:,5] = p[:,:,5] + q_p_pred[:,:,0]

    #d
    #q[:,:,1] = 10*torch.sigmoid((q[:,:,1] + q_p_pred[:,:,1]))
    #p[:,:,1] = p[:,:,1] + q_p_pred[:,:,1]

    #phi
    #q[:,:,2] = 10*torch.sigmoid((q[:,:,2] + q_p_pred[:,:,3]))
    #p[:,:,2] = p[:,:,2] + q_p_pred[:,:,2]

    #a
    #q[:,:,8] = torch.sigmoid((q[:,:,8] + q_p_pred[:,:,5]))
    p[:,:,8] = p[:,:,8] + q_p_pred[:,:,0]

    #delta
    #q[:,:,9] = torch.sigmoid((q[:,:,9] + q_p_pred[:,:,7]))
    p[:,:,9] = p[:,:,9] + q_p_pred[:,:,1]

    return q, p




def q_and_p_dyn(mpc_T, q_p_pred, Q_manual, p_manual):
    # Cost order:
    # [for casadi] sigma_diff, d, phi, v, a, delta
    # [for model]  sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_pen, a, delta

    n_Q, BS, _ = q_p_pred.shape

    q_p_pred = q_p_pred.repeat(mpc_T//n_Q, 1, 1)

    e = 1e-9

    q = e*torch.ones((mpc_T,BS,12)) + torch.tensor(Q_manual).unsqueeze(1).float()
    p = torch.zeros((mpc_T,BS,12)) + torch.tensor(p_manual).unsqueeze(1).float()

    #sigma_diff
    #q[:,:,5] = q[:,:,5] + q_p_pred[:,:,0].clamp(e)
    p[:,:,7] = p[:,:,7] + q_p_pred[:,:,0]

    #d
    #q[:,:,1] = (q[:,:,1] + q_p_pred[:,:,1]).clamp(e)
    p[:,:,1] = p[:,:,1] + q_p_pred[:,:,1]

    #phi
    #q[:,:,2] = (q[:,:,2] + q_p_pred[:,:,3]).clamp(e)
    p[:,:,2] = p[:,:,2] + q_p_pred[:,:,2]

    #a
    #q[:,:,10] = (q[:,:,10] + q_p_pred[:,:,5]).clamp(e)
    p[:,:,10] = p[:,:,10] + q_p_pred[:,:,3]

    #delta
    #q[:,:,11] = (q[:,:,11] + q_p_pred[:,:,7]).clamp(e)
    p[:,:,11] = p[:,:,11] + q_p_pred[:,:,4]

    return q, p
