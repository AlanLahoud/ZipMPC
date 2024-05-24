#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from casadi import *

import os

#import shutil
#FFMPEG_BIN = shutil.which('ffmpeg')

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.style.use('bmh')

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class CasadiControl():
    def __init__(self, track_coordinates, params):
        super().__init__()

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
        self.v_max = params[3]
        self.ac_max = params[4]
        self.dt = params[5] #0.04

        self.a_max = params[6]
        self.delta_max = params[7]


    def sigmoid(self, x):
      	return 1 / (1 + exp(-x))

    def curv_casadi(self, sigma):
        '''
        # create vector of size of vector sigma passed to the function
        sigma_num_dim = sigma.ndimension()
        # The idea for this function is the following
        # 1. step = get dimensions of sigma and replicate the track coordinates of sigma and curv accordingly
        track_sigma = self.track_coordinates[2,:]
        track_curv = self.track_coordinates[4,:]
        track_sigma_rep = track_sigma.repeat(sigma.size()[0],1)
        # extend sigma by one dimension - for next operation
        sigma_ext = sigma.reshape(-1,1)
        # 2. step = calcualate (sigma - sigma_track)**2.argmin(ndimension())
        curv_mask = ((sigma_ext - track_sigma_rep)**2).argmin(sigma_num_dim)
        # 3. step, use vector from before to get curvatures.
        curv = torch.gather(track_curv,0,curv_mask.type(torch.int64))
        print(sigma[:10])
        print(curv_mask[:10])
        print(curv[:10]) '''
        print("curv start")

        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[1],1)

	#convert everything to numpy s.t. it works with casadi
        sigma_f_mat_np = sigma_f_mat.numpy()
        sigma_f_np = self.sigma_f.numpy()
        curv_f_np = self.curv_f.numpy()

        sigma_shifted = reshape(sigma,num_s[1],1)- sigma_f_mat_np
        curv_unscaled = self.sigmoid(5*sigma_shifted)
        curv = reshape((curv_unscaled@(curv_f_np.reshape(-1,1))),1,num_s[1])
        print("curv end")
        return curv

    def penalty_d(self, d, factor=100000.):
        overshoot_pos = np.clip((d - 0.5*self.track_width*0.75),0,None)
        overshoot_neg = np.clip((-d - 0.5*self.track_width*0.75),0,None)
        penalty_pos = np.exp(overshoot_pos) - 1
        penalty_neg = np.exp(overshoot_neg) - 1
        return factor*(penalty_pos + penalty_neg)

    def penalty_v(self, v, factor=100000.):
        penalty_pos = np.clip((v - self.v_max),0,None) ** 2
        return factor*penalty_pos

    def mpc_casadi(self,q,p,x0,horizon,df,dc,dx,du):

        # with: dx, du: number of states/inputs passed from the dynamics
        # dc: number of constraints
        # df: number of states that we do not use
        q_used = np.hstack([q[5],q[1:4],q[6:]])
        p_used = np.hstack([p[5],p[1:4],p[6:]])

        # here the q and the p scale the following
        # feature vector [sigma-sigma_0, d, phi, v, penalty_d,penalty_v,a,delta]

        Ts = self.dt
        N=horizon
        l_r=0.2
        l_f=0.2

        dx = dx - dc - df

        x_sym = SX.sym('x_sym',dx,N+1)
        u_sym = SX.sym('u_sym',du,N)

        #solver parameters
        options = {}
        options['ipopt.max_iter'] = 20000
        options['verbose'] = False

        beta = np.arctan(l_r/(l_r+l_f)*np.tan(u_sym[1,0:N]))

        dyn1 = horzcat((x_sym[0,0] - x0[0,0]), (x_sym[0,1:N+1] - x_sym[0,0:N] - Ts*(x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1.-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn2 = horzcat((x_sym[1,0] - x0[0,1]), (x_sym[1,1:N+1] - x_sym[1,0:N] - Ts*(x_sym[3,0:N]*np.sin(x_sym[2,0:N]+beta))))
        dyn3 = horzcat((x_sym[2,0] - x0[0,2]), (x_sym[2,1:N+1] - x_sym[2,0:N] - Ts*(x_sym[3,0:N]*(1/l_f)*np.sin(beta)-self.curv_casadi(x_sym[0,0:N])*x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn4 = horzcat((x_sym[3,0] - x0[0,3]), (x_sym[3,1:N+1] - x_sym[3,0:N] - Ts*(u_sym[0,0:N])))
        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0,0],
                       x_sym[1:,0:N],
                       self.penalty_d(x_sym[1,0:N]),
                       fmax(np.zeros([1,N]),
                            (x_sym[1,0:N]-track_width)),
                       self.penalty_v(x_sym[3,0:N]),
                       u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+dc+du)
        p_sym = SX.sym('q_sym',dx+dc+du)
        Q_sym = diag(q_sym)

        l = sum2(transpose(diag(transpose(feat)@Q_sym@feat)) + transpose(p_sym)@feat)
        dl = substitute(substitute(l,q_sym,q_used),p_sym,p_used)

        const = vertcat(transpose(dyn1),transpose(dyn2),transpose(dyn3),transpose(dyn4),transpose(u_sym[0,0:N]),transpose(u_sym[1,0:N]))
        lbg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),-2*np.ones(N),-1*np.ones(N)]
        ubg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),2*np.ones(N),1*np.ones(N)]
        lbx = -np.inf * np.ones(dx*(N+1)+du*N)
        ubx = np.inf * np.ones(dx*(N+1)+du*N)

        x = vertcat(reshape(x_sym[:,0:N+1],(dx*(N+1),1)),reshape(u_sym[:,0:N],(du*N,1)))

        # define solver
        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        # create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        # solve optimization problem
        print("solve optimization problem")
        solver_output = solver(**solver_input)

        # process ouput
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*N:]
        x = sol_evalf[:-du*N]
        #print(u)
        #print(x)
        u_applied = u[0:1]

        # print solution
        return sol_evalf


    def mpc_casadi_with_constraints(self,q,p,x0,horizon,df,dc,dx,du):

        # with: dx, du: number of states/inputs passed from the dynamics
        # dc: number of constraints
        # df: number of states that we do not use
        q_used = np.hstack([q[5],q[1:4],q[9:]])
        p_used = np.hstack([p[5],p[1:4],p[9:]])

        # here the q and the p scale the following
        # feature vector [sigma-sigma_0, d, phi, v, penalty_d,penalty_v,a,delta]

        Ts = self.dt
        N=horizon
        l_r=0.2
        l_f=0.2

        dx = dx - dc - df

        x_sym = SX.sym('x_sym',dx,N+1)
        u_sym = SX.sym('u_sym',du,N)

        #solver parameters
        options = {}
        options['ipopt.max_iter'] = 20000
        options['verbose'] = False

        beta = np.arctan(l_r/(l_r+l_f)*np.tan(u_sym[1,0:N]))

        dyn1 = horzcat((x_sym[0,0] - x0[0,0]), (x_sym[0,1:N+1] - x_sym[0,0:N] - Ts*(x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1.-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn2 = horzcat((x_sym[1,0] - x0[0,1]), (x_sym[1,1:N+1] - x_sym[1,0:N] - Ts*(x_sym[3,0:N]*np.sin(x_sym[2,0:N]+beta))))
        dyn3 = horzcat((x_sym[2,0] - x0[0,2]), (x_sym[2,1:N+1] - x_sym[2,0:N] - Ts*(x_sym[3,0:N]*(1/l_f)*np.sin(beta)-self.curv_casadi(x_sym[0,0:N])*x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn4 = horzcat((x_sym[3,0] - x0[0,3]), (x_sym[3,1:N+1] - x_sym[3,0:N] - Ts*(u_sym[0,0:N])))
        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0,0],x_sym[1:,0:N],u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+du)
        p_sym = SX.sym('q_sym',dx+du)
        Q_sym = diag(q_sym)

        l = sum2(transpose(diag(transpose(feat)@Q_sym@feat)) + transpose(p_sym)@feat)
        dl = substitute(substitute(l,q_sym,q_used),p_sym,p_used)

        const = vertcat(
            transpose(dyn1),
            transpose(dyn2),
            transpose(dyn3),
            transpose(dyn4),
            transpose(u_sym[0,0:N]),
            transpose(u_sym[1,0:N]),
            transpose(x_sym[1,0:N+1]),
            transpose(x_sym[3,0:N+1]))

        lbg = np.r_[np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    -self.a_max*np.ones(N),
                    -self.delta_max*np.ones(N),
                    -0.5*self.track_width*0.75*np.ones(N+1),
                    -0.1*np.ones(N+1)]

        ubg = np.r_[np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    np.zeros(N+1),
                    self.a_max*np.ones(N),
                    self.delta_max*np.ones(N),
                    0.5*self.track_width*0.75*np.ones(N+1),
                    self.v_max*np.ones(N+1)]

        lbx = -np.inf * np.ones(dx*(N+1)+du*N)
        ubx = np.inf * np.ones(dx*(N+1)+du*N)

        x = vertcat(reshape(x_sym[:,0:N+1],(dx*(N+1),1)),reshape(u_sym[:,0:N],(du*N,1)))

        # define solver
        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        # create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        # solve optimization problem
        solver_output = solver(**solver_input)

        # process ouput
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*N:]
        x = sol_evalf[:-du*N]
        #print(u)
        #print(x)
        u_applied = u[0:1]

        #print(sol.solveroutput.info.lambda)

        # print solution
        return sol_evalf




    def mpc_casadi_with_constraints_2(self, q, p, x0, horizon, df, dc, dx, du):
        Ts = self.dt
        N = horizon
        l_r = 0.2
        l_f = 0.2

        dx = dx - dc - df

        x_sym = SX.sym('x_sym', dx, N + 1)
        u_sym = SX.sym('u_sym', du, N)

        # Solver parameters
        options = {}
        options['ipopt.max_iter'] = 20000
        options['verbose'] = False

        beta = np.arctan(l_r / (l_r + l_f) * np.tan(u_sym[1, 0:N]))

        dyn1 = horzcat(
            (x_sym[0, 0] - x0[0, 0]),
            (x_sym[0, 1:N + 1] - x_sym[0, 0:N] - Ts * (x_sym[3, 0:N] * (np.cos(x_sym[2, 0:N] + beta) / (1. - self.curv_casadi(x_sym[0, 0:N]) * x_sym[1, 0:N]))))
        )
        dyn2 = horzcat(
            (x_sym[1, 0] - x0[0, 1]),
            (x_sym[1, 1:N + 1] - x_sym[1, 0:N] - Ts * (x_sym[3, 0:N] * np.sin(x_sym[2, 0:N] + beta)))
        )
        dyn3 = horzcat(
            (x_sym[2, 0] - x0[0, 2]),
            (x_sym[2, 1:N + 1] - x_sym[2, 0:N] - Ts * (x_sym[3, 0:N] * (1 / l_f) * np.sin(beta) - self.curv_casadi(x_sym[0, 0:N]) * x_sym[3, 0:N] * (np.cos(x_sym[2, 0:N] + beta) / (1 - self.curv_casadi(x_sym[0, 0:N]) * x_sym[1, 0:N]))))
        )
        dyn4 = horzcat(
            (x_sym[3, 0] - x0[0, 3]),
            (x_sym[3, 1:N + 1] - x_sym[3, 0:N] - Ts * (u_sym[0, 0:N]))
        )

        # Define the objective function
        feat = vertcat(x_sym[0, 0:N] - x0[0, 0], x_sym[1:, 0:N], u_sym[:, 0:N])
        q_sym = SX.sym('q_sym', dx + du)
        p_sym = SX.sym('p_sym', dx + du)

        # Simplify the cost to only use q[5] and p[5]
        q_used = np.zeros(dx + du)
        p_used = np.zeros(dx + du)
        q_used[5] = q[5]
        p_used[5] = p[5]

        Q_sym = diag(q_sym)
        l = sum2(transpose(diag(transpose(feat) @ Q_sym @ feat)) + transpose(p_sym) @ feat)
        dl = substitute(substitute(l, q_sym, q_used), p_sym, p_used)

        const = vertcat(
            transpose(dyn1),
            transpose(dyn2),
            transpose(dyn3),
            transpose(dyn4),
            transpose(u_sym[0, 0:N]),
            transpose(u_sym[1, 0:N]),
            transpose(x_sym[1, 0:N + 1]),
            transpose(x_sym[3, 0:N + 1])
        )

        lbg = np.r_[np.zeros(N + 1),
                    np.zeros(N + 1),
                    np.zeros(N + 1),
                    np.zeros(N + 1),
                    -self.a_max * np.ones(N),
                    -self.delta_max * np.ones(N),
                    -0.5 * self.track_width * 0.75 * np.ones(N + 1),
                    -0.1 * np.ones(N + 1)]

        ubg = np.r_[np.zeros(N + 1),
                    np.zeros(N + 1),
                    np.zeros(N + 1),
                    np.zeros(N + 1),
                    self.a_max * np.ones(N),
                    self.delta_max * np.ones(N),
                    0.5 * self.track_width * 0.75 * np.ones(N + 1),
                    self.v_max * np.ones(N + 1)]

        lbx = -np.inf * np.ones(dx * (N + 1) + du * N)
        ubx = np.inf * np.ones(dx * (N + 1) + du * N)

        x = vertcat(reshape(x_sym[:, 0:N + 1], (dx * (N + 1), 1)), reshape(u_sym[:, 0:N], (du * N, 1)))

        # Define solver
        nlp = {'x': x, 'f': dl, 'g': const}
        solver = nlpsol('solver', 'ipopt', nlp, options)

        # Create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        # Solve optimization problem
        solver_output = solver(**solver_input)

        # Process output
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du * N:]
        x = sol_evalf[:-du * N]
        u_applied = u[0:1]

        # Print solution
        return sol_evalf

    def mpc_casadi_with_constraints_paj(self,q,p,x0,horizon,df,dc,dx,du,x_warmstart):

        # with: dx, du: number of states/inputs passed from the dynamics
        # dc: number of constraints
        # df: number of states that we do not use
        q_used = np.hstack([q[7],q[1:6],q[10:]])
        p_used = np.hstack([p[7],p[1:6],p[10:]])

        # here the q and the p scale the following
        # feature vector [sigma-sigma_0, d, phi, v, penalty_d,penalty_v,a,delta]

        Ts = self.dt
        N=horizon
        l_r=0.2
        l_f=0.2

        # set them like in learned dynamics
        m = 1.5
        g = 9.81
        mu = 0.85
        I_z = m*l_r*l_f # this should be an approximation

        B = 6.0
        C = 1.6
        D = 1.0

        dx = dx - dc - df

        x_sym = SX.sym('x_sym',dx,N+1)
        u_sym = SX.sym('u_sym',du,N)

        a_f = np.arctan((x_sym[5,0:N]  + l_f*x_sym[3,0:N])/(fabs(x_sym[4,0:N])+0.01))-u_sym[1,0:N]
        a_r = np.arctan((x_sym[5,0:N]  - l_f*x_sym[3,0:N])/(fabs(x_sym[4,0:N])+0.01))

        F_yf = -0.5*m*g*mu*(D*np.sin(C*np.arctan(B*a_f)))
        F_yr = -0.5*m*g*mu*(D*np.sin(C*np.arctan(B*a_r)))

        #solver parameters
        options = {}
        options['ipopt.max_iter'] = 200
        options['verbose'] = False

        beta = np.arctan(l_r/(l_r+l_f)*np.tan(u_sym[1,0:N]))

        dyn1 = horzcat((x_sym[0,0] - x0[0,0]), (x_sym[0,1:N+1] - x_sym[0,0:N] - Ts*((x_sym[4,0:N]*np.cos(x_sym[2,0:N])-x_sym[5,0:N]*np.sin(x_sym[2,0:N]))/(1.-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N]))))
        dyn2 = horzcat((x_sym[1,0] - x0[0,1]), (x_sym[1,1:N+1] - x_sym[1,0:N] - Ts*(x_sym[4,0:N]*np.cos(x_sym[2,0:N])+x_sym[5,0:N]*np.sin(x_sym[2,0:N]))))
        dyn3 = horzcat((x_sym[2,0] - x0[0,2]), (x_sym[2,1:N+1] - x_sym[2,0:N] - Ts*(x_sym[3,0:N] - self.curv_casadi(x_sym[0,0:N])*(x_sym[4,0:N]*np.cos(x_sym[2,0:N])-x_sym[5,0:N]*np.sin(x_sym[2,0:N]))/(1-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N]))))
        dyn4 = horzcat((x_sym[3,0] - x0[0,3]), (x_sym[3,1:N+1] - x_sym[3,0:N] - Ts*(1/I_z*(l_f*F_yf -l_r*F_yr))))
        dyn5 = horzcat((x_sym[4,0] - x0[0,4]), (x_sym[4,1:N+1] - x_sym[4,0:N] - Ts*(u_sym[0,0:N]+x_sym[3,0:N]*x_sym[5,0:N])))
        dyn6 = horzcat((x_sym[5,0] - x0[0,5]), (x_sym[5,1:N+1] - x_sym[5,0:N] - Ts*(1/m*(F_yf*np.cos(u_sym[1,0:N])+F_yr)-x_sym[3,0:N]*x_sym[4,0:N])))
        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0,0],x_sym[1:,0:N],u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+du)
        p_sym = SX.sym('p_sym',dx+du)
        Q_sym = diag(q_sym)

        l = sum2(transpose(diag(transpose(feat)@Q_sym@feat)) + transpose(p_sym)@feat)
        dl = substitute(substitute(l,q_sym,q_used),p_sym,p_used)

        const = vertcat(transpose(dyn1),transpose(dyn2),transpose(dyn3),transpose(dyn4),transpose(dyn5),transpose(dyn6),transpose(u_sym[0,0:N]),transpose(u_sym[1,0:N]),transpose(x_sym[1,0:N+1]),transpose(x_sym[4,0:N+1]))
        lbg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),-2*np.ones(N),-1*np.ones(N),-0.5*self.track_width*0.75*np.ones(N+1),-0.1*np.ones(N+1)]
        ubg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),2*np.ones(N),1*np.ones(N),0.5*self.track_width*0.75*np.ones(N+1),self.v_max*0.95*np.ones(N+1)]
        lbx = -np.inf * np.ones(dx*(N+1)+du*N)
        ubx = np.inf * np.ones(dx*(N+1)+du*N)

        x = vertcat(reshape(x_sym[:,0:N+1],(dx*(N+1),1)),reshape(u_sym[:,0:N],(du*N,1)))
        w_ws = np.vstack([np.reshape(x_warmstart[:dx,0:N+1],(dx*(N+1),1)),np.reshape(x_warmstart[dx+dc+df:,0:N],(du*(N),1))])

        # define solver
        nlp = {'x':x,'f':dl, 'g':const}
        solver = nlpsol('solver','ipopt', nlp, options)

        # create solver input
        solver_input = {}
        solver_input['lbx'] = lbx
        solver_input['ubx'] = ubx
        solver_input['lbg'] = lbg
        solver_input['ubg'] = ubg

        # add initial guess to solver
        solver_input['x0'] = w_ws

        # solve optimization problem
        solver_output = solver(**solver_input)

        # process ouput
        sol = solver_output['x']
        sol_evalf = np.squeeze(evalf(sol))
        u = sol_evalf[-du*N:]
        x = sol_evalf[:-du*N]
        #print(u)
        #print(x)
        u_applied = u[0:1]

        #print(sol.solveroutput.info.lambda)

        # print solution
        return sol_evalf
