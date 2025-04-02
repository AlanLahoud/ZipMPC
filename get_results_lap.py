import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from mpc.env_dx import frenet_dyn_bicycle, frenet_kin_bicycle  #changed
from mpc.track.src import simple_track_generator, track_functions

import utils

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from tqdm import tqdm

import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--param_model', type=str, default='lcredh')  #'lcredh' or 'bo'
    parser.add_argument('--dyn', type=str, default='kin')
    parser.add_argument('--seed_n', type=int, default=0)
    parser.add_argument('--NS', type=int, default=5)
    parser.add_argument('--NL', type=int, default=18)
    parser.add_argument('--n_Q', type=int, default=1)
    parser.add_argument('--p_sigma_manual', type=float, default=8.0)
    parser.add_argument('--track', type=str, default='TEST_TRACK')

    return parser.parse_args()


args = parse_arguments()

##########################################################################################
################### P A R A M E T E R S ##################################################
##########################################################################################

dyn_model = args.dyn
param_model = args.param_model

assert dyn_model in ['kin','pac']

if dyn_model=='kin':
    import utils_kin as utils_car
else:
    import utils_pac as utils_car

NS = args.NS # Short horizon Length 
NL = args.NL # Long Horizon Length
n_Q = args.n_Q # Number of learnable parameters through the short horizon

assert n_Q<=NS


# Manual progress cost parameter (initial guess)
p_sigma_manual = args.p_sigma_manual

track_name = args.track

# Seed for reproducibility
seed_n= args.seed_n
torch.manual_seed(seed_n)
np.random.seed(seed_n)

# Car axis length
l_r = 0.05
l_f = l_r


if dyn_model=='kin':
    delta_max = 0.40

else:
    delta_max = 0.50
    

# Curve smoothness
k_curve = 25.

#discretization
dt = 0.03

# Maximum v and a
v_max=1.8
a_max = 1.0

# Clip learnable parameters (TanH, check NN)
max_p = 10

# Model path to save
str_model = f'{dyn_model}_{NS}_{NL}_{n_Q}_{p_sigma_manual}'

# Track parameters
track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68
max_track_width_perc = 0.68
bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width
bound_d = 0.5*max_track_width_perc*track_width
t_track = 0.3
init_track = [0,0,0]

# Parameters as tensors for both Short and Long horizons
params = torch.tensor(
    [l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, NS])
params_H = torch.tensor(
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

# Setting the learnable model dynamics (short horizon)
# Here, we also set manual cost parameters / initial guess
if dyn_model=='kin':
    print('KINEMATICS')
    dx=4
    du=2
    lqr_iter = 20
    eps=0.00001
    true_dx = utils_car.FrenetKinBicycleDx(track_coord, params, 'cpu')
    control = utils_car.CasadiControl(track_coord, params)
    Q_manual = np.repeat(np.expand_dims(
        np.array([0.0, 3.0, 1.0, 0.01, 0.01, 0.01, 1, 1, 0.01, 1.0]), 0), NS, 0)
    p_manual = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NS, 0)
    
    control_H = utils_car.CasadiControl(track_coord, params_H)
    Q_manual_H = np.repeat(np.expand_dims(
        np.array([0.0, 3.0, 1.0, 0.01, 0.01, 0.01, 1, 1, 0.01, 1.0]), 0), NL, 0)
    p_manual_H = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)

    idx_to_casadi = [5,1,2,3,8,9]
    idx_to_NN = [1,2,3]
    
    
    
else:
    print('PACEJKA')
    dx=6
    du=2
    lqr_iter = 35
    eps=0.00001
    true_dx = utils_car.FrenetDynBicycleDx(track_coord, params, 'cpu')
    control = utils_car.CasadiControl(track_coord, params)
    Q_manual = np.repeat(np.expand_dims(
        np.array([0, 50.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NS, 0)
    p_manual = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NS, 0)
    
    control_H = utils_car.CasadiControl(track_coord, params_H)
    Q_manual_H = np.repeat(np.expand_dims(
        np.array([0, 50.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1]), 0), NL, 0)
    p_manual_H = np.repeat(np.expand_dims(
        np.array([0, 0, 0, 0, 0., 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), NL, 0)
    
    idx_to_casadi = [7,1,2,3,4,5,10,11]
    idx_to_NN = [1,2,4]



def plot_sim(x_simulated, u_simulated, vc, output_path, lab_text='Velocity'):
    
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

    plt.axis('off')
    
    cbar = plt.colorbar(sm, ax=ax)
    
    cbar.set_label(lab_text) 
    
    print('x_init: ' + str(gen.xCoords[0]))
    print('y_init: ' + str(gen.yCoords[0]))
    print('yaw_init: ' + str(gen.tangentAngle[0]))
    print('Total Arc Length: ' + str(gen.arcLength[-1]/2))
    #plt.show()

    plt.tight_layout()
    
    plt.savefig(output_path, format='png', dpi=300)



def plot_sim_all(x_simulateds, output_path):
    dict_colors = {0: 'red', 1: 'blue', 2: 'limegreen'}
    labels = {0: 'MCSH', 1: 'MCLH', 2: 'Our'}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=250)
    gen.plotPoints(ax)
    
    for s, x_simulated in enumerate(x_simulateds):
        x_list = []
        y_list = []
        
        for i in range(x_simulated.shape[0]):
            xy = utils.frenet_to_cartesian(x_simulated[i, :2], track_coord)
            x_list.append(xy[0].numpy())
            y_list.append(xy[1].numpy())
        
        x_plot = np.array(x_list)
        y_plot = np.array(y_list)       
    
        # Plotting the segments and adding a label only for the first segment
        for i in range(len(x_plot) - 1):
            if i == 0:
                ax.plot(x_plot[i:i+2], y_plot[i:i+2], color=dict_colors[s], label=labels[s], linewidth=1.5, alpha=0.5)
            else:
                ax.plot(x_plot[i:i+2], y_plot[i:i+2], color=dict_colors[s], linewidth=1.5, alpha=0.5)

    plt.axis('off')
    
    print('x_init: ' + str(gen.xCoords[0]))
    print('y_init: ' + str(gen.yCoords[0]))
    print('yaw_init: ' + str(gen.tangentAngle[0]))
    print('Total Arc Length: ' + str(gen.arcLength[-1] / 2))

    # Add legend with horizontal orientation at the top
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)



def plot_data(curv_full, var_p, y_label, output_path):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    average_curv = curv_full.mean(axis=-1)
    ax.scatter(average_curv, var_p, color='blue', alpha=0.7, edgecolor='k', s=50)
    
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel('Average Curvature (Context)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, format='png', dpi=300)
    plt.close(fig)



def eval_lap(x0, Q_manual, p_manual, control, model=None):
    
    finished = 0
    crashed = 0
    steps = 0
    max_steps=700

    x_full = x0.reshape(-1,1).copy()[:dx+2]
    u_full = np.zeros((2,1))
    q_p_full = []
    curv_full = []

    while finished==0 and crashed==0:
        if model==None:
            q_lap_np_casadi = np.expand_dims((Q_manual[:,idx_to_casadi].T), 1)
            p_lap_np_casadi = np.expand_dims((p_manual[:,idx_to_casadi].T), 1)
        
        else:
            x0_lap_pred_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
            curv_lap = utils.get_curve_hor_from_x(x0_lap_pred_torch, track_coord, NL)    
            inp_lap = torch.hstack((x0_lap_pred_torch[:,idx_to_NN], curv_lap))

            q_p_pred_lap = model(inp_lap)

            q_p_full.append(q_p_pred_lap.squeeze().detach().numpy())
            curv_full.append(curv_lap.squeeze().detach().numpy())
        
            q_lap, p_lap = utils_car.q_and_p(NS, q_p_pred_lap, Q_manual, p_manual)
            
            q_lap_np_casadi = torch.permute(q_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
            p_lap_np_casadi = torch.permute(p_lap[:,:,idx_to_casadi], (2, 1, 0)).detach().numpy()
      
        x_b_pred, u_b_pred = utils_car.solve_casadi(
            q_lap_np_casadi[:,0,:], p_lap_np_casadi[:,0,:],
            x0, dx, du, control)
        
        x0_new = true_dx.forward((torch.tensor(x0)).unsqueeze(0), torch.tensor(u_b_pred)[0:1]).squeeze()[:dx+2].detach().numpy()
        x0 = x0_new.copy()

        x_full = np.append(x_full, x0.reshape(-1,1), axis=1)
        u_full = np.append(u_full, u_b_pred[0].reshape(-1,1), axis=1)
        
        if x0[0]>=track_coord[2].max().numpy()/2-0.1:
            finished=1
        
        if x0[1]>bound_d + 0.04 or x0[1]<-bound_d - 0.04 or steps>max_steps:
            crashed=1
    
        steps = steps+1
    
    lap_time = dt*steps

    return lap_time, finished, x_full, u_full, np.array(q_p_full), np.array(curv_full)


def eval_lap_bo(x0, Q_manual, p_manual, Q_bo, p_bo_app, control, test_bo):
    
    finished = 0
    crashed = 0
    steps = 0
    max_steps=700

    x_full = x0.reshape(-1,1).copy()[:dx+2]
    u_full = np.zeros((2,1))
    q_p_full = []
    curv_full = []

    while finished==0 and crashed==0:
        if not test_bo:
            q_lap_np_casadi = np.expand_dims((Q_manual[:,idx_to_casadi].T), 1)
            p_lap_np_casadi = np.expand_dims((p_manual[:,idx_to_casadi].T), 1)
        
        else:
            # currently hard programmed to kinematic model.
            p_bo = np.repeat(np.expand_dims(p_bo_app, 0), NS, 0)

            q_lap_np_casadi = np.expand_dims((Q_bo[:,idx_to_casadi].T), 1)
            p_lap_np_casadi = np.expand_dims((p_bo[:,idx_to_casadi].T), 1)

        x_b_pred, u_b_pred = utils_car.solve_casadi(
            q_lap_np_casadi[:,0,:], p_lap_np_casadi[:,0,:],
            x0, dx, du, control)
        
        x0_new = true_dx.forward((torch.tensor(x0)).unsqueeze(0), torch.tensor(u_b_pred)[0:1]).squeeze()[:dx+2].detach().numpy()
        x0 = x0_new.copy()

        x_full = np.append(x_full, x0.reshape(-1,1), axis=1)
        u_full = np.append(u_full, u_b_pred[0].reshape(-1,1), axis=1)
        
        if x0[0]>=track_coord[2].max().numpy()/2-0.1:
            finished=1
        
        if x0[1]>bound_d + 0.04 or x0[1]<-bound_d - 0.04 or steps>max_steps:
            crashed=1
    
        steps = steps+1
    
    lap_time = dt*steps

    return lap_time, finished, x_full, u_full


##########################################################################################
################### I N F E R E N C E  ###################################################
##########################################################################################

if param_model == 'lcredh':
    model = utils.TCN(NL, n_Q, 5, max_p)
    model.load_state_dict(torch.load(f'./models/model_{str_model}.pkl'))
    model.eval()

    x0_lap = utils_car.sample_init_test(1, true_dx, sn=2).numpy().squeeze()
    x0_lap_pred = x0_lap[:dx+4]
    x0_lap_manual = x0_lap[:dx+4]


    lap_time, finished, x_full, u_full, q_p_full, curv_full = eval_lap(x0_lap_pred, Q_manual, p_manual, control, model=model)
    lap_time_H, finished_H, x_H_full, u_H_full, _, _ = eval_lap(x0_lap_pred, Q_manual_H, p_manual_H, control_H)
    lap_time_T, finished_T, x_full_T, u_full_T, _, _ = eval_lap(x0_lap_pred, Q_manual, p_manual, control)

    print('LAP TIMES:', lap_time, lap_time_H, lap_time_T)

    if q_p_full.ndim == 3:
        q_p_full = q_p_full.mean(1)

    plot_data(curv_full, q_p_full[:,0], r'Sigmadiff Linear Cost: $p_{\sigma}$', f'./imgs_paper/plot_sdf_{str_model}_{track_name}.png')
    plot_data(curv_full, q_p_full[:,1], r'Lateral Deviation Linear Cost: $p_{d}$', f'./imgs_paper/plot_lat_{str_model}_{track_name}.png')
    plot_data(curv_full, q_p_full[:,2], r'Heading Angle Linear Cost: $p_{\phi}$', f'./imgs_paper/plot_phi_{str_model}_{track_name}.png')
    plot_data(curv_full, q_p_full[:,3], r'Acceleration Linear Cost: $p_{a}$', f'./imgs_paper/plot_a_{str_model}_{track_name}.png')
    plot_data(curv_full, q_p_full[:,4], r'Steering Angle Linear Cost: $p_{\delta}$', f'./imgs_paper/plot_delta_{str_model}_{track_name}.png')

    plot_sim(x_full.T, u_full.T, q_p_full[:,1], f'./imgs_paper/traj_lat_{str_model}_{track_name}.png', r'Lateral Deviation Linear Cost: $p_{d}$')
    plot_sim(x_full.T, u_full.T, q_p_full[:,2], f'./imgs_paper/traj_phi_{str_model}_{track_name}.png', r'Heading Angle Linear Cost: $p_{\phi}$')
    plot_sim(x_full.T, u_full.T, q_p_full[:,4], f'./imgs_paper/traj_delta_{str_model}_{track_name}.png', r'Steering Angle Linear Cost: $p_{\delta}$')

    plot_sim(x_full_T.T, u_full_T.T, x_full_T[idx_to_NN[2]], f'./imgs_paper/traj_vel_T_{str_model}_{track_name}.png', r'Velocity$')
    plot_sim(x_H_full.T, u_H_full.T, x_H_full[idx_to_NN[2]], f'./imgs_paper/traj_vel_H_{str_model}_{track_name}.png', r'Velocity$')
    plot_sim(x_full.T, u_full.T, x_full[idx_to_NN[2]], f'./imgs_paper/traj_vel_{str_model}_{track_name}.png', r'Velocity$')

    plot_sim_all([x_full_T.T, x_H_full.T, x_full.T], f'./imgs_paper/plot_traj_all_{str_model}_{track_name}.png')

    lap_times = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time, finished, x_full, _, _, _ = eval_lap(x0_s, Q_manual, p_manual, control, model=model)
        lap_times.append(lap_time)

    lap_times_H = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time_H, finished_H, x_H_full, _, _, _ = eval_lap(x0_s, Q_manual_H, p_manual_H, control_H)
        lap_times_H.append(lap_time_H)

    lap_times_T = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time_T, finished_T, x_full_T, _, _, _ = eval_lap(x0_s, Q_manual, p_manual, control)
        lap_times_T.append(lap_time_T)

elif param_model == 'bo':
    x0_lap = utils_car.sample_init_test(1, true_dx, sn=2).numpy().squeeze()
    x0_lap_pred = x0_lap[:dx+4]
    x0_lap_manual = x0_lap[:dx+4]

    p_bo_base = np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0])
    p_bo_add = np.zeros(10)
    Q_bo = Q_manual
    idx_to_learned_param = [5,1,2,8,9]
    p_bo_add[idx_to_learned_param] = np.array([-0.175, -0.4, 0.0, -0.1, -0.175])
    p_bo_app = p_bo_base + p_bo_add

    lap_time, finished, x_full, u_full = eval_lap_bo(x0_lap_pred, Q_manual, p_manual, Q_bo, p_bo_app, control, True)
    lap_time_H, finished_H, x_H_full, u_H_full = eval_lap_bo(x0_lap_pred, Q_manual_H, p_manual_H, Q_bo, p_bo_app, control_H, False)
    lap_time_T, finished_T, x_full_T, u_full_T = eval_lap_bo(x0_lap_pred, Q_manual, p_manual, Q_bo, p_bo_app, control, False)

    print('LAP TIMES:', lap_time, lap_time_H, lap_time_T)

    # plot_sim(x_full_T.T, u_full_T.T, x_full_T[idx_to_NN[2]], f'./imgs_paper/traj_vel_T_{str_model}_{track_name}.png', r'Velocity$')
    # plot_sim(x_H_full.T, u_H_full.T, x_H_full[idx_to_NN[2]], f'./imgs_paper/traj_vel_H_{str_model}_{track_name}.png', r'Velocity$')
    # plot_sim(x_full.T, u_full.T, x_full[idx_to_NN[2]], f'./imgs_paper/traj_vel_{str_model}_{track_name}.png', r'Velocity$')

    # plot_sim_all([x_full_T.T, x_H_full.T, x_full.T], f'./imgs_paper/plot_traj_all_{str_model}_{track_name}.png')

    lap_times = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time, finished, x_full,_ = eval_lap_bo(x0_s, Q_manual, p_manual, Q_bo, p_bo_app, control, True)
        lap_times.append(lap_time)

    lap_times_H = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time_H, finished_H, x_H_full,_ = eval_lap_bo(x0_s, Q_manual_H, p_manual_H, Q_bo, p_bo_app, control_H, False)
        lap_times_H.append(lap_time_H)

    lap_times_T = []
    for i in tqdm(range(10)):
        x0_s = x0_lap_pred.copy()
        x0_s[1] = x0_s[1] + 0.03*torch.randn((1,))
        x0_s[2] = x0_s[2] + 0.04*torch.randn((1,))
        lap_time_T, finished_T, x_full_T,_ = eval_lap_bo(x0_s, Q_manual, p_manual, Q_bo, p_bo_app, control, False)
        lap_times_T.append(lap_time_T)



print(lap_times)
print(lap_times_H)
print(lap_times_T)