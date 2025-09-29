# ZipMPC

This repository contains scripts for training and evaluating neural networks combined with Model Predictive Control (MPC) to reduce the MPC horizon length. The code represents the experimental part of our paper "ZipMPC: Compressed Context-Dependent MPC Cost via Imitation Learning". 

## Files and Descriptions

### `reduce_horizon_mpc.py`
This is the main file used to train and save the learned neural networks for our proposed method, ZipMPC.

**Main Arguments:**
1. `dyn`: Choose between `"kin"` and `"pac"` for the dynamic car model.
2. `NS`: The number of time steps in the short horizon, representing the horizon length for the learned cost.
3. `NL`: The number of time steps in the long horizon, representing the horizon for the reference trajectory optimization.
4. `n_Q`: The number of learnable variables through the time dimension.
5. `p_sigma_manual`: The main variable for the given manual cost to enforce progress in the track.

---

### `get_results_lap.py`
This script evaluates the lap time using the saved models.

**Additional Argument:**
- `track`: The name of the track. Check the `mpc.track.src` directory to see the available track options.

---

### `get_results_imitation.py`
This script evaluates the imitation loss based on the control variables using the saved models.

**Additional Argument:**
- `empc` (boolean): Set to `True` if you want to evaluate the explicit MPC instead of the learned cost model.

---

## Sharing and access information
Copyright (c) 2025 ETH Zurich, Institute for Dynamics Systems and Control , and Örebro University, Center for Applied Autonomous Sensor Systems (AASS), Rahel Rickenbach, Alan Lahoud, Erik Schaffernicht, Melanie N. Zeilinger Johannes A. Stork. No rights reserved. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
