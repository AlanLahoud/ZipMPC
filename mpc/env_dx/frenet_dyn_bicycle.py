import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class FrenetDynBicycleDx(nn.Module):
    def __init__(self, track_coordinates=None, params=None):
        super().__init__()

        # states: sigma, d, phi, r, v_x, v_y (6) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1)
        self.n_state = 6+2+1+1
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
        curv_unscaled = torch.sigmoid(5000*sigma_shifted)
        curv = (curv_unscaled@(self.curv_f.reshape(-1,1))).type(torch.float)


        return curv.reshape(-1)


    def penalty_d(self, d, factor=100000.):
        overshoot_pos = (d - 0.5*self.track_width*0.75).clamp(min=0)
        overshoot_neg = (-d - 0.5*self.track_width*0.75).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1
        return factor*(penalty_pos + penalty_neg)

    def penalty_v(self, v, factor=100000.):
        # as v_x defines the forward direction - this function is applied with
        # respect to v_x.
        penalty_pos = (v - self.v_max*0.95).clamp(min=0) ** 2
        return factor*penalty_pos

    def forward(self, state, u):
        softplus_op = torch.nn.Softplus(20)
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        l_r = self.l_r
        l_f = self.l_f

        a, delta = torch.unbind(u, dim=1)

        sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        #fP(α) = D sin(C arctan(Bα)) B = 10.0, C = 1.9, D = 1.0
        # αf = arctan( vy + lf ˙ψ|vx|)− δf
        a_f = torch.atan((v_y + l_f*r)/v_x)-delta
        a_r = torch.atan((v_y - l_f*r)/v_x)

        m = 0.5
        g = 9.81
        mu = 0.5
        I_z = m*l_r*l_f # this should be an approximation

        B = 10.0
        C = 1.9
        D = 1.0

        F_yf = -0.5*m*g*mu*(D*torch.sin(C*torch.atan(B*a_f)))
        F_yr = -0.5*m*g*mu*(D*torch.sin(C*torch.atan(B*a_r)))

        dsigma = (v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d)
        dd = v_x*torch.cos(phi)+v_y*torch.sin(phi)
        dphi = r-self.curv(sigma)*((v_x*torch.cos(phi)-v_y*torch.sin(phi))/(1.-self.curv(sigma)*d))
        dr = 1/I_z*(l_f*F_yf -l_r*F_yr)
        dv_x = a + r*v_y
        dv_y = 1/m*(F_yf*torch.cos(delta)+F_yr)-r*v_x
        #print(sigma)
        #print(self.curv(sigma))

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

        #d_lb = softplus_op(-d - 0.5*self.track_width)
        #d_ub = softplus_op(d - 0.5*self.track_width)
        #v_lb = softplus_op(-v + 0)
        #v_ub = softplus_op(v - self.v_max)

        state = torch.stack((sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub), 1)

        return state


    # This function is for plotting
    # def get_frame(self, state, ax=None):
    #     state = util.get_data_maybe(state.view(-1))
    #     assert len(state) == 10
    #     sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
    #     l_r,l_f = torch.unbind(self.params)
    #
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(6,6))
    #     else:
    #         fig = ax.get_figure()
    #
    #     # here I need to figure out what we would like to plot
    #     ax.plot(d,d, color='k')
    #     ax.set_xlim((-2, 2))
    #     ax.set_ylim((-2, 2))
    #     return fig, ax

    def get_true_obj(self):
        # dimensions adjusted

    	# 0  	 1   2   3  4	 5	   6        7          8      9     10  11
        # sigma, d, phi, r, v_x, v_y, sigma_0, sigma_diff, d_pen, v_ub	a   delta
        q = torch.Tensor([ 0.,  2.,  1.,  0., 0.,   0., 0., 0., 0., 0., 1., 2.])
        assert not hasattr(self, 'mpc_lin')
        p = torch.Tensor([ 0.,  0.,  0.,  0., 0., 0., 0., -2., 100., 100., -1,  0.])
        return Variable(q), Variable(p)

if __name__ == '__main__':
    dx = FrenetKinBicycleDx()
    n_batch, T = 8, 50
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit= torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
