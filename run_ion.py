import os
import numpy as np
import matplotlib.pyplot as plt
from single_syn import Single_syn as Single
from neuron import h


random_seed = 1
rng = np.random.default_rng(seed=random_seed)
h.Random().Random123_globalindex(random_seed)
h.load_file('stdgui.hoc')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='')               # neuron id
parser.add_argument('--device', type=str, default='cuda:0')     # PyTorch computing device
parser.add_argument('--adam', action='store_true')              # whether to use Adam optimizer
args, _ = parser.parse_known_args()

ID = args.id
OUTPUT_PATH = os.path.join('output', 'ion', ID)
os.makedirs(OUTPUT_PATH, exist_ok=True)
DEVICE = args.device                            # PyTorch computing device
K_max_t = 75                                    # transfer impedance maximum time window for curve fitting (ms)
K_filename = os.path.join(OUTPUT_PATH, "K.npy")
w_min, w_max = -1, 1                            # restrict polarity

### Simulation Parameters ###
stim_on = 50                            # stimulus onset (ms)
stim_dur = 100                          # stimulus duration (ms)
tstop = 500                             # simulation end (ms)
lr_on = 50                              # learning start (ms)
lr_off = 450                            # learning end (ms)
v_rest = -75                            # resting potential (mV)
dt = 0.1                                # time step for curve fitting (ms)
epochs = 100       					    # maximum training epochs
ADAM = args.adam                        # whether to use Adam optimizer
if ADAM:
    alpha = 1e-5                        # initial learning rate for Adam optimizer
else:
    alpha = 3e-8                        # initial learning rate for SGD optimizer
spk_threshold = -40                     # Spike detection threshold (mV)


def train(cell: Single, data):
    inputs, v_target = data
    lr_start = int(lr_on / dt)
    lr_end = int(lr_off / dt)
    t_rec = h.Vector().record(h._ref_t)
    loss_opt = 1e60

    if ADAM:            # Adam params
        adam_m = 0.
        adam_v = 0.
        beta_1, beta_2 = 0.9, 0.999
        beta_1_t, beta_2_t = 1., 1.
        epsilon = 1e-7

    loss_train = []
    for iepoch in range(epochs):
        cell.set_stim(inputs)

        h.t = 0
        h.tstop = tstop
        h.finitialize(v_rest)
        h.fcurrent()
        tstep = 0
        while h.t < h.tstop:
            print(f"epoch: {iepoch:d}, t: {h.t:f}", end='\r')
            h.fadvance()
            tstep += 1
            cell.update_dvdw(tstep)
        print('')

        v_out = np.array(cell.v_out)
        v_target_clip = np.clip(v_target, a_min=None, a_max=spk_threshold)[lr_start:lr_end]
        v_output_clip = np.clip(v_out, a_min=None, a_max=spk_threshold)[lr_start:lr_end]

        loss = np.mean(0.5*(np.abs(v_target_clip - v_output_clip))**2)
        print(f"loss: {loss:.5g}")
        loss_train.append(loss)
        
        if loss < loss_opt:
            loss_opt = loss

            ### Plot optimal results ###
            fig, ax = plt.subplots()
            ax.plot(t_rec, v_out, label='Learned')
            ax.plot(t_rec, v_target, label='Target', linestyle='dashed')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Voltage (mV)')
            ax.legend(loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'epoch: {iepoch:03d}')
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_PATH, "Optimal.png"))
            plt.close(fig)

        ### Compute dw ###
        dLtdv = -(v_target_clip - v_output_clip)
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

        ### Plot ###
        fig, ax = plt.subplots()
        ax.plot(range(1, len(loss_train) + 1), loss_train)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Training error', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_PATH, "loss.png"))
        plt.close(fig)
    
    return loss_train


if __name__ == '__main__':
    h.dt = dt
    h.celsius = 36

    config = {
        'seed': random_seed,
        'v_rest': v_rest,
        'K_max_t': K_max_t, 'K_filename': K_filename,
        'w_min': w_min, 'w_max': w_max,
        'device': DEVICE,
    }
    cell = Single(config)

    ### Initialize weights ###
    w = rng.uniform(-1.*1e-3, 1.*1e-3, (cell.N_ion,))
    cell.set_weights(w)

    data = load_data()
    
    ### Train ###
    loss_train = train(cell, data)

    print('Done')
    h.quit()
