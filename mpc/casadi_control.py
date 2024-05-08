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
    def __init__(self, track_coordinates=None, params=None):
        super().__init__()

        # states: sigma, d, phi, v (4) + d_lb, d_ub (2)
        self.n_state = 4 + 2           # here add amount of states plus amount of exact penalty terms
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


        # model parameters: l_r, l_f (beta and curv(sigma) are calculated in the dynamics)
        if params is None:
            # l_r, l_f
            self.params = Variable(torch.Tensor((0.2, 0.2)))
        else:
            self.params = params
            assert len(self.params) == 2

            self.delta_threshold_rad = np.pi  #12 * 2 * np.pi / 360
            self.d_threshold = 2
            self.max_acceleration = 2

            self.dt = 0.05   # name T in document

            self.track_width = 0.5

            self.lower = -self.track_width/2
            self.upper = self.track_width/2

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

    def mpc_casadi(self,q,p,x0,horizon,df,dc,dx,du,track_width,v_max):

        # with: dx, du: number of states/inputs passed from the dynamics
        # dc: number of constraints
        # df: number of states that we do not use

        Ts = 0.05
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
        dyn2 = horzcat((x_sym[1,0] - x0[0,1]), (x_sym[1,1:N+1] - x_sym[1,0:N] - Ts*(x_sym[3,0:N]*np.sin(x_sym[2,0:N])+beta)))
        dyn3 = horzcat((x_sym[2,0] - x0[0,2]), (x_sym[2,1:N+1] - x_sym[2,0:N] - Ts*(x_sym[3,0:N]*(1/l_f)*np.sin(beta)-self.curv_casadi(x_sym[0,0:N])*x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn4 = horzcat((x_sym[3,0] - x0[0,3]), (x_sym[3,1:N+1] - x_sym[3,0:N] - Ts*(u_sym[0,0:N])))
        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0,0],x_sym[1:,0:N],fmax(np.zeros([1,N]),(-x_sym[1,0:N]-track_width)),fmax(np.zeros([1,N]),(x_sym[1,0:N]-track_width)),fmax(np.zeros([1,N]),(-x_sym[3,0:N] + 0)),fmax(np.zeros([1,N]),(x_sym[3,0:N]-v_max)),u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+dc+du)
        p_sym = SX.sym('q_sym',dx+dc+du)
        Q_sym = diag(q_sym)

        l = sum2(transpose(diag(transpose(feat)@Q_sym@feat)) + transpose(p_sym)@feat)
        dl = substitute(substitute(l,q_sym,q),p_sym,p)

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


    def mpc_casadi_with_constraints(self,q,p,x0,horizon,df,dc,dx,du,track_width,v_max):

        # with: dx, du: number of states/inputs passed from the dynamics
        # dc: number of constraints
        # df: number of states that we do not use

        Ts = 0.05
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
        dyn2 = horzcat((x_sym[1,0] - x0[0,1]), (x_sym[1,1:N+1] - x_sym[1,0:N] - Ts*(x_sym[3,0:N]*np.sin(x_sym[2,0:N])+beta)))
        dyn3 = horzcat((x_sym[2,0] - x0[0,2]), (x_sym[2,1:N+1] - x_sym[2,0:N] - Ts*(x_sym[3,0:N]*(1/l_f)*np.sin(beta)-self.curv_casadi(x_sym[0,0:N])*x_sym[3,0:N]*(np.cos(x_sym[2,0:N]+beta)/(1-self.curv_casadi(x_sym[0,0:N])*x_sym[1,0:N])))))
        dyn4 = horzcat((x_sym[3,0] - x0[0,3]), (x_sym[3,1:N+1] - x_sym[3,0:N] - Ts*(u_sym[0,0:N])))
        # think about how to integrate the curvature function

        # define symbolic variables for cost parameters
        feat = vertcat(x_sym[0,0:N]-x0[0,0],x_sym[1:,0:N],u_sym[:,0:N])
        q_sym = SX.sym('q_sym',dx+du)
        p_sym = SX.sym('q_sym',dx+du)
        Q_sym = diag(q_sym)

        l = sum2(transpose(diag(transpose(feat)@Q_sym@feat)) + transpose(p_sym)@feat)
        dl = substitute(substitute(l,q_sym,q),p_sym,p)

        const = vertcat(transpose(dyn1),transpose(dyn2),transpose(dyn3),transpose(dyn4),transpose(u_sym[0,0:N]),transpose(u_sym[1,0:N]),transpose(x_sym[1,0:N+1]),transpose(x_sym[3,0:N+1]))
        lbg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),-2*np.ones(N),-1*np.ones(N),-track_width*np.ones(N+1),0*np.ones(N+1)]
        ubg = np.r_[np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),np.zeros(N+1),2*np.ones(N),1*np.ones(N),track_width*np.ones(N+1),v_max*np.ones(N+1)]
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
