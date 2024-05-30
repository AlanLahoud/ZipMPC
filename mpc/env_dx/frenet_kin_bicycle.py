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

class FrenetKinBicycleDx(nn.Module):
    def __init__(self, track_coordinates, params):
        super().__init__()

        # states: sigma, d, phi, v (4) + sigma_0, sigma_diff (2) + d_pen (1) + v_ub (1)
        self.n_state = 4+2+1+1
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
        #self.ac_max = params[4]
        self.dt = params[5] #0.04

        
        # model parameters: l_r, l_f (beta and curv(sigma) are calculated in the dynamics)
        #if params is None:
        #    # l_r, l_f
        #    self.params = Variable(torch.Tensor((0.2, 0.2)))
        #else:
        #    self.params = params
        #    assert len(self.params) == 2

            #self.track_width = 4 # here we need to check how we do that as our d is not with respect to center line

            #self.delta_threshold_rad = np.pi  #12 * 2 * np.pi / 360
            #self.v_max = 2
            #self.ac_max = ((0.7*self.v_max)**2)*3.33
            
            #self.max_acceleration = 2

            #self.dt = 0.05   # name T in document

            #self.track_width = 0.5

            #self.lower = -self.track_width/2
            #self.upper = self.track_width/2

            # Still need to think about how I do this here

            # 0  	1   2    3  4	5
            # sigma d  phi   v  a	delta
            #self.goal_state = torch.Tensor([ 0.,  0.,  1., 0.,   0.])
            #self.goal_weights = torch.Tensor([0.1, 0.1,  1., 1., 0.1])
            #self.ctrl_penalty = 0.001

            ###############################################
            #self.mpc_eps = 1e-4
            #self.linesearch_decay = 0.5
            #self.max_linesearch_iter = 2

    def curv(self, sigma):
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

        num_sf = self.sigma_f.size()
        num_s = sigma.size()

        sigma_f_mat = self.sigma_f.repeat(num_s[0],1)


        sigma_shifted = sigma.reshape(-1,1) - sigma_f_mat
        curv_unscaled = torch.sigmoid(10*sigma_shifted)
        curv = (curv_unscaled@(self.curv_f.reshape(-1,1))).type(torch.float)

        '''
        curv = torch.zeros(sigma.size())
        num_s = self.sigma_f.size()
        num_s_i = num_s[0]

        for i in range(num_s_i):
            new_sig = (sigma - self.sigma_f[i])*5
            curv = curv + self.curv_f[i]*torch.sigmoid(new_sig)'''

        return curv.reshape(-1)

    
    def penalty_d(self, d, factor=10000.):  
        overshoot_pos = (d - 0.4*self.track_width).clamp(min=0)
        overshoot_neg = (-d - 0.4*self.track_width).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1 
        return factor*(penalty_pos + penalty_neg)
    
    def penalty_v(self, v, factor=10000.):          
        overshoot_pos = (v - self.v_max*0.95).clamp(min=0)
        overshoot_neg = (-v + 0.001).clamp(min=0)
        penalty_pos = torch.exp(overshoot_pos) - 1
        penalty_neg = torch.exp(overshoot_neg) - 1 
        return factor*(penalty_pos + penalty_neg)
            
    #def penalty_ac(self, ac, factor=1000.):  
    #    penalty = (ac - self.ac_max).clamp(min=0) ** 2
    #    return factor*penalty
    
    def forward(self, state, u):
        softplus_op = torch.nn.Softplus(20)
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

        dsigma = v*(torch.cos(phi+beta)/(1.-k*d))
        dd = v*torch.sin(phi+beta)
        dphi = v/self.l_f*torch.sin(beta)-k*v*(torch.cos(phi+beta)/(1-k*d))
        dv = a

        sigma = sigma + self.dt * dsigma
        d = d + self.dt * dd
        phi = phi + self.dt * dphi
        v = v + self.dt * dv
        sigma_0 = sigma_0                   # we need to carry it on
        sigma_diff = sigma - sigma_0
        
        #Ackerman theory
        # http://www.ingveh.ulg.ac.be/uploads/education/MECA0525/11_MECA0525_VEHDYN1_SSTURN_2021-2022.pdf
        #ac = v**2 * delta / (self.l_r+self.l_f)
                
        d_pen = self.penalty_d(d)        
        v_ub = self.penalty_v(v)
        #ac_ub = self.penalty_ac(ac)
        
        #d_lb = softplus_op(-d - 0.5*self.track_width)
        #d_ub = softplus_op(d - 0.5*self.track_width)
        #v_lb = softplus_op(-v + 0)
        #v_ub = softplus_op(v - self.v_max)

        state = torch.stack((sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub), 1)

        return state


    # This function is for plotting
    #WE ARE NOT USING THIS
    def get_frame(self, state, ax=None):
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 10
        sigma, d, phi, v, sigma_0, sigma_diff, d_pen, v_ub = torch.unbind(state, dim=1)
        l_r,l_f = torch.unbind(self.params)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        # here I need to figure out what we would like to plot
        ax.plot(d,d, color='k')
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        return fig, ax

    def get_true_obj(self):
    	# this I need to rewrite carefully, as not all of them have q unequal to 0
    	# I think we just need to set them individually

    	# 0  	1   2    3  4	  5	6    7
        # sigma d  phi   v  d_lb  d_ub	a    delta
        q = torch.Tensor([ 0.,  2.,  1.,  0., 0., 0., 0., 0., 0., 0., 1., 2.])
        assert not hasattr(self, 'mpc_lin')
        p = torch.Tensor([ 0.,  0.,  0.,  0., 0., -2., 100., 100., 100., 100., -1,  0.])
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
        #fig, ax = dx.get_frame(x[0])
        #fig.savefig('{:03d}.png'.format(t))
        #plt.close(fig)

    #vid_file = 'cartpole_vid.mp4'
    #if os.path.exists(vid_file):
    #    os.remove(vid_file)
    #cmd = ('{} -loglevel quiet '
    #        '-r 32 -f image2 -i %03d.png -vcodec '
    #        'libx264 -crf 25 -pix_fmt yuv420p {}').format(
    #    FFMPEG_BIN,
    #    vid_file
    #)
    #os.system(cmd)
    #for t in range(T):
    #    os.remove('{:03d}.png'.format(t))
