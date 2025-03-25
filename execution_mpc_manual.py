import torch
import torch.nn as nn
import numpy as np

import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

import time

import utils

import argparse

import sys
from sys import exit

import utils_pac_hardware as utils_car

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




def plot_sim(x_simulated, u_simulated, vc, output_path, lab_text='Velocity', time_lap=0.0):
    
    x_list = []
    y_list = []

    color_data = vc
    
    for i in range(x_simulated.shape[0]):
        xy = utils.frenet_to_cartesian(x_simulated[i,:2], track_coord)
        x_list.append(xy[0].numpy())
        y_list.append(xy[1].numpy())
    
    x_plot = np.array(x_list)
    y_plot = np.array(y_list)
    
    fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=250)
    gen.plotPoints(ax)

    custom_cmap = plt.get_cmap('winter').reversed()
    norm = Normalize(vmin=color_data.min(), vmax=color_data.max())
    sm = ScalarMappable(cmap=custom_cmap, norm=norm)


    for i in range(len(x_plot) - 1):
        ax.plot(x_plot[i:i+2], y_plot[i:i+2], color=custom_cmap(norm(color_data[i])), alpha=0.5)

    #plt.axis('off')
    ax.annotate(time_lap, xy=(0, -0.5))
    
    cbar = plt.colorbar(sm, ax=ax)
    
    cbar.set_label(lab_text) 
    
    print('x_init: ' + str(gen.xCoords[0]))
    print('y_init: ' + str(gen.yCoords[0]))
    print('yaw_init: ' + str(gen.tangentAngle[0]))
    print('Total Arc Length: ' + str(gen.arcLength[-1]/2))
    #plt.show()

    plt.tight_layout()
    #plt.show()
    
    plt.savefig(output_path, format='png', dpi=300)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--NL', type=int, default=60)

    return parser.parse_args()
    


args = parse_arguments()

NL = args.NL

out_path = f'_{NL}'

p_sigma_manual = 8.

l_r = 0.038 
l_f = 0.052  
delta_max = 0.40

# Curve smoothness
k_curve = 25.

#discretization
dt = 0.02

n_steps_dt = 1

# Maximum v and a
#v_max=1.8
v_max = 2.0
a_max = 1.0


# Track parameters
track_name = 'DEMO_TRACK'
track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68
bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width
t_track = 0.3
init_track = [0,0,0]


params_dx = torch.tensor(
    [l_r, l_f, track_width, dt/n_steps_dt, k_curve, v_max, delta_max, a_max, NL])
params_casadi = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NL])

# Generating track
gen = simple_track_generator.trackGenerator(track_density,track_width)

track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'TEST_TRACK'    : track_functions.test_track,
    'TEST_TRACK_2'    : track_functions.test_track2,
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


print('PACEJKA HARDWARE')
dx=6
du=2
lqr_iter = 35
eps=0.00001
true_dx = utils_car.FrenetDynBicycleDx(track_coord, params_dx, 'cpu')

control_H = utils_car.CasadiControl(track_coord, params_casadi)
Q_manual_H = (1/NL)*np.repeat(np.expand_dims(
    np.array([0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NL, 0)
p_manual_H = (1/NL)*np.repeat(np.expand_dims(
    np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)
#p_manual_H[-1,7] = -p_sigma_manual*NL

idx_to_casadi = [7,1,2,3,4,5,10,11]
idx_to_NN = [1,2,4]


# This sampling should bring always the same set of initial states
x0_lap = utils_car.sample_init_test(1, true_dx, sn=0).numpy()

x0_lap_manual = x0_lap[:,:dx+4]

finished = 0
crashed = 0
steps = 0
max_steps=1200

x0_b_manual = x0_lap_manual[0].copy()

x0_b_manual = x0_lap_manual[0].copy()
x_manual_full_H = []
x_frenet_full = []

u_full = []

x0_cart = torch.tensor([0., -1., 1., 0,])

u_step = np.array([0.5, 0.])

while finished==0 and crashed==0:
    q_lap_manual_casadi = Q_manual_H[:,idx_to_casadi].T
    p_lap_manual_casadi = p_manual_H[:,idx_to_casadi].T

    x_b_manual, u_b_manual = utils_car.solve_casadi(
        q_lap_manual_casadi, p_lap_manual_casadi,
        x0_b_manual, dx, du, control_H, u_step)

    u_step = u_b_manual[0:1].mean(0)

    for ss in range(n_steps_dt): 
  
        x0_b_manual = true_dx.forward((torch.tensor(x0_b_manual)).unsqueeze(0), 
                                                  torch.tensor(u_step).unsqueeze(0)).squeeze()[:dx+4].detach().numpy()
        x_frenet_full.append(x0_b_manual)
        u_full.append(u_step)

    print('Sigma', round(x0_b_manual[0], 3), 'Vx', round(x0_b_manual[4], 3), 'r', round(x0_b_manual[3], 3))

    
    if x0_b_manual[0]>track_coord[2].max().numpy()/2:
        finished=1

    if x0_b_manual[1]>bound_d_casadi+0.04 or x0_b_manual[1]<-bound_d_casadi-0.04 or steps>max_steps:
        crashed=1

    steps = steps+1
    #print(steps)

#x_manual_full_H = np.array(x_manual_full_H)
x_frenet_full = np.array(x_frenet_full)
lap_time = dt*steps

print(f'Manual extended NL = {NL}, lap time: {lap_time}, finished: {finished}')


plot_sim(torch.tensor(x_frenet_full), torch.tensor(x_frenet_full), np.array(u_full)[:,1], 
         f'./outs_imgs/steer_{out_path}.png', lab_text='Steering', time_lap=lap_time)

plot_sim(torch.tensor(x_frenet_full), torch.tensor(x_frenet_full), np.array(x_frenet_full)[:,4], 
         f'./outs_imgs/vel_{out_path}.png', lab_text='Velocity', time_lap=lap_time)