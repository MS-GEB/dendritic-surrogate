import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuron import h
from single import Single
from tqdm import tqdm


random_seed = 1
rng = np.random.default_rng(seed=random_seed)
h.Random().Random123_globalindex(random_seed)
h.load_file('stdgui.hoc')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='pas')          # output voltage type, 'pas' for subthreshold, 
                                                                # 'single' for single burst, 'multi' for multiple bursts
parser.add_argument('--device', type=str, default=None)         # PyTorch device
parser.add_argument('--adam', action='store_true')              # whether to use Adam optimizer
args, _ = parser.parse_known_args()

MODE = args.mode
assert MODE in ['pas', 'single', 'multi'], "'pas' for subthreshold, 'single' for single burst, 'multi' for multiple bursts"
OUTPUT_PATH = os.path.join('output', 'syn', MODE)
os.makedirs(OUTPUT_PATH, exist_ok=True)
if args.device:                                 # Specified PyTorch device
    DEVICE = args.device
elif torch.cuda.is_available():                 # Default PyTorch device
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
K_max_t = 75                                    # transfer impedance maximum time window for curve fitting (ms)
K_filename = os.path.join(OUTPUT_PATH, 'K.npy')
w_e_min, w_e_max = 0., None                     # restrict polarity
w_i_min, w_i_max = None, -0.                    # restrict polarity
N_s, N_e, N_i = 2, 400, 100

### Simulation Parameters ###
bin_on = 50                             # stimulus start earliest (ms)
bin_off = 350                           # stimulus start latest (ms)
stim_dur = 100                          # stimulus duration (ms)
tstop = 500                             # simulation end (ms)
lr_on = 50                              # learning start (ms)
lr_off = 450                            # learning end (ms)
rand_freq = 10                          # random input firing rate (Hz)
bg_freq = 2                             # background noise firing rate (Hz)
v_rest = -75                            # resting potential (mV)
dt = 0.1                                # time step for curve fitting (ms)
epochs = 100       					    # maximum training epochs
ADAM = args.adam                        # whether to use Adam optimizer
if ADAM:
    alpha = 1e-5                        # learning rate for Adam optimizer
else:
    alpha = 3e-8                        # learning rate for SGD optimizer
spk_thrsh = -40                         # Spike detection threshold (mV)


def gen_target(cell: Single, inputs):
    cell.set_stim(inputs)
    t_rec = h.Vector().record(h._ref_t)

    h.t = 0
    h.tstop = tstop
    h.finitialize(v_rest)
    h.fcurrent()
    tstep = 0
    with tqdm(desc="Running", total=tstop, unit='ms') as pbar:
        pbar.bar_format = "{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        while h.t < h.tstop:
            h.fadvance()
            tstep += 1
            cell.update_pred(tstep)
            pbar.update(dt)

    return np.array(t_rec), np.array(cell.v_out), np.array(cell.v_pred)


def train(cell: Single, inputs, v_tgt):    
    lr_start, lr_end = int(lr_on / dt), int(lr_off / dt)
    t_rec = h.Vector().record(h._ref_t)
    loss_opt = 1e60

    if ADAM:            # Adam params
        adam_m = 0.
        adam_v = 0.
        beta_1, beta_2 = 0.9, 0.999
        beta_1_t, beta_2_t = 1., 1.
        epsilon = 1e-7

    for iepoch in range(epochs):
        cell.set_stim(inputs)

        h.t = 0
        h.tstop = tstop
        h.finitialize(v_rest)
        h.fcurrent()
        tstep = 0
        with tqdm(desc=f"Epoch {iepoch:d}", total=tstop, unit='ms') as pbar:
            pbar.bar_format = "{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            while h.t < h.tstop:
                h.fadvance()
                tstep += 1
                cell.update_dvdw(tstep)
                pbar.update(dt)

        v_out = np.array(cell.v_out)
        v_tgt_clip = np.clip(v_tgt, a_min=None, a_max=spk_thrsh)[lr_start:lr_end]
        v_out_clip = np.clip(v_out, a_min=None, a_max=spk_thrsh)[lr_start:lr_end]

        loss = 0.5 * np.sum((v_tgt_clip - v_out_clip)**2) / (lr_end - lr_start)
        
        if loss < loss_opt:
            loss_opt = loss

            ### Plot optimal results ###
            fig, ax = plt.subplots()
            ax.plot(t_rec, v_out, label='Learned')
            ax.plot(t_rec, v_tgt, label='Target', linestyle='dashed')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Voltage (mV)')
            ax.legend(loc='upper left', frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'epoch: {iepoch:03d}')
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_PATH, 'Optimal.png'))
            plt.close(fig)

        ### Compute dw ###
        dLtdv = -(v_tgt_clip - v_out_clip)
        dw = cell.get_dw(dLtdv, lr_start, lr_end)
        if ADAM:
            adam_m = beta_1 * adam_m + (1. - beta_1) * dw
            adam_v = beta_2 * adam_v + (1. - beta_2) * dw * dw
            beta_1_t = beta_1_t * beta_1
            beta_2_t = beta_2_t * beta_2
            m_hat = adam_m / (1. - beta_1_t)
            v_hat = adam_v / (1. - beta_2_t)
            dw = m_hat / (np.sqrt(v_hat) + epsilon)
        dw *= alpha
        cell.update_weights(dw)


if __name__ == '__main__':
    h.dt = dt
    h.celsius = 36

    config = {
        'seed': random_seed,
        'N_s': N_s, 'N_e': N_e, 'N_i': N_i,
        'v_rest': v_rest,
        'K_max_t': K_max_t, 'K_filename': K_filename,
        'w_e_min': w_e_min, 'w_e_max': w_e_max,
        'w_i_min': w_i_min, 'w_i_max': w_i_max,
        'device': DEVICE,
    }
    cell = Single(config)

    inputs = []
    for _ in range(N_e + N_i):
        bin_start = rng.uniform(bin_on, bin_off)
        # background noise before stimulus
        bg_list_1 = np.where(rng.random(int(bin_start / dt)) < dt * bg_freq / 1000)[0] * dt
        # stimulus input
        t_list = np.where(rng.random(int(stim_dur / dt)) < dt * rand_freq / 1000)[0] * dt + bin_start
        # background noise after stimulus
        bg_list_2 = np.where(rng.random(int((tstop - bin_start - stim_dur) / dt)) < dt * bg_freq / 1000)[0] * dt + bin_start + stim_dur
        # combine
        t_list = np.sort(np.concatenate([bg_list_1, t_list, bg_list_2]))
        inputs.append(t_list)

    ### Set target weights ###
    match MODE:
        case 'pas':
            w_s = np.array([1.,] * N_s)
            w_e = rng.uniform(0.1*1e-3, 1.4*1e-3, (N_e,))
            w_i = rng.uniform(-1.4*1e-3, -0.1*1e-3, (N_i,))
        case 'single':
            w_s = np.array([1.,] * N_s)
            w_e = rng.uniform(0.1*1e-3, 1.7*1e-3, (N_e,))
            w_i = rng.uniform(-1.7*1e-3, -0.1*1e-3, (N_i,))
        case 'multi':
            w_s = np.array([1.,] * N_s)
            w_e = rng.uniform(0.1*1e-3, 2.*1e-3, (N_e,))
            w_i = rng.uniform(-2.*1e-3, -0.1*1e-3, (N_i,))
    cell.set_weights(np.concatenate((w_s, w_e, w_i)))

    t_rec, v_tgt, _ = gen_target(cell, inputs)

    ### Reinitialize weights ###
    w_s = np.array([1.,] * N_s)
    w_e = rng.uniform(0.*1e-3, 0.3*1e-3, (N_e,))
    w_i = rng.uniform(-0.3*1e-3, -0.*1e-3, (N_i,))
    cell.set_weights(np.concatenate((w_s, w_e, w_i)))

    ### Train ###
    train(cell, inputs, v_tgt)

    print('Done')
    h.quit()
