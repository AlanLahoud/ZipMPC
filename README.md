# Reduce Horizon MPC

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
