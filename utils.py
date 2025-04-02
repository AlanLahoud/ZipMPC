import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx




class TCN(nn.Module):
    def __init__(self, mpc_H, mpc_T, O, K):
        super(TCN, self).__init__()
        input_size = 3

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.2) 

        # Fully connected layers for shared representation
        self.fc1 = nn.Linear(8 * mpc_H + input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)

        # Global representation layer
        self.fc_global = nn.Linear(512, O)

        # Modulation layer for time-varying outputs
        self.fc_modulation = nn.Linear(512, mpc_T * O)

        # Activation functions
        self.activation = nn.LeakyReLU(0.1)
        self.output_activation = nn.Tanh()

        # Model parameters
        self.mpc_T = mpc_T
        self.O = O
        self.K = K

    def forward(self, x):
        global_context, time_series = x[:, :3], x[:, 3:]

        time_series = time_series.unsqueeze(1) 
        time_series = self.activation(self.bn1(self.conv1(time_series)))
        time_series = self.dropout(self.activation(self.bn2(self.conv2(time_series))))

        time_series = time_series.view(time_series.size(0), -1) 

        x = torch.cat([time_series, global_context], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        global_cost = self.fc_global(x) 

        modulation = self.fc_modulation(x) 
        modulation = modulation.view(self.mpc_T, -1, self.O) 

        global_cost = global_cost.unsqueeze(0) 
        outputs = global_cost + modulation 

        outputs = self.K * self.output_activation(outputs / self.K)
        return outputs



def get_curve_hor_from_x(x, track_coord, NL, v_max, dt):
    idx_track_batch = ((x[:, 0] - track_coord[[2], :].T) ** 2).argmin(0)
    
    # Calculate the maximum allowed value based on sigma
    max_sigma = v_max * dt * NL + x[:, 0]
    idx_track_batch_max = ((max_sigma - track_coord[[2], :].T) ** 2).argmin(0)
    
    stepsize = torch.clamp((idx_track_batch_max - idx_track_batch) // NL, min=1)
    
    range_indices = torch.arange(NL).unsqueeze(0)  # Shape (1, NL)
    batch_arange = idx_track_batch.unsqueeze(1) + range_indices * stepsize.unsqueeze(1)
    
    idcs_track_batch = torch.clip(batch_arange, 0, track_coord.shape[1] - 1)
    
    curvs = track_coord[4, idcs_track_batch].float()
    
    # Ensure the result has exactly H_curve elements for each batch
    assert curvs.shape[1] == NL, f"Curvs does not match NL: {curvs.shape[1]} != {NL}"
    
    return curvs


def compute_x_coord(point_f, ref_path, nearest_index):
    return ref_path[0,nearest_index] - point_f[1]*torch.sin(ref_path[3,nearest_index])

def compute_y_coord(point_f, ref_path, nearest_index):
    return ref_path[1,nearest_index] + point_f[1]*torch.cos(ref_path[3,nearest_index])

def get_nearest_index(point_f, ref_path):
    return ((point_f[0] - ref_path[2,:])**2).argmin()

def frenet_to_cartesian(point_f, ref_path):
    nearest_index = get_nearest_index(point_f, ref_path)
    x = compute_x_coord(point_f, ref_path, nearest_index)
    y = compute_y_coord(point_f, ref_path, nearest_index)
    return torch.tensor([x, y])