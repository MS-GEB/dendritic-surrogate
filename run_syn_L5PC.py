import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuron import h
from single_L5PC import Single_L5PC as Single
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=None)     # PyTorch device
parser.add_argument('--seed', type=int, default=0)          # global random seed
parser.add_argument('--init_seed', type=int, default=0)     # generate init weights from seed
args, _ = parser.parse_known_args()

random_seed = args.seed
rng = np.random.default_rng(seed=random_seed)
h.Random().Random123_globalindex(random_seed)
h.load_file('stdgui.hoc')

OUTPUT_PATH = os.path.join('output', f'syn_L5PC')
os.makedirs(OUTPUT_PATH, exist_ok=True)
if args.device:                                 # Specified PyTorch device
    DEVICE = args.device
elif torch.cuda.is_available():                 # Default PyTorch device
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
K_max_t = 75                                    # transfer impedance maximum time window for numerical acc (ms)
K_filename = os.path.join(OUTPUT_PATH, 'K.npy')
N_syn = 100                                     # number of input synapses
seg_outs = [
    ('soma', 0, 0.5),
    ('apic', 36, 0.9),
    ]
N_out = len(seg_outs)

### Simulation Parameters ###
N_apic_cluster = 30                     # number of distal apical synapses
N_apic_bg = 10                          # number of apical background firing synapses
N_apic = N_apic_cluster + N_apic_bg
low_freq = 10                           # random input firing rate (Hz)
burst_freq = 30                         # high freq firing rate (Hz)
bg_freq = 2                             # background noise firing rate (Hz)
stim_on = 100                           # stimulus on (ms)
stim_dur = 800                          # duration of high freq input (ms)
tstop = 1000                            # simulation end (ms)
lr_on = 0                               # learning start (ms)
lr_stop = 1000                          # learning end (ms)
w_target_scale_apic_bg = 1e-4           # normal distribution sigma, target apical cluster weights
w_target_scale_dend = 4e-3              # normal distribution sigma, target basal weights
w_target_apic_cluster_min, w_target_apic_cluster_max = 0., 1e-3   # uniform distribution range, target apical weights
w_init_scale_apic = 1e-4                # normal distribution sigma, initial apical weights
w_init_scale_dend = 1e-3                # normal distribution sigma, initial basal weights
alpha = 3e-9                            # learning rate
alpha_dend_mul = 10                     # learning rate multiplier for basal synapses

w_clip_min, w_clip_max = None, None         # clip weights at min/max
dw_apic_min, dw_apic_max = -1e-4, 1e-4      # clip apical dw at min/max
dw_dend_min, dw_dend_max = -1e-3, 1e-3      # clip basal dw at min/max

### Simulation Parameters ###
v_rest = -80                            # resting potential (mV)
dt = 0.025                              # time step (ms)
epochs = 100                            # max training epochs
v_clip_max = ((-40,), (0,))             # max clip for voltage when computing MSE (mV)


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

    return np.array(t_rec), np.array(cell.v_outs), np.array(cell.v_preds)

def train(cell: Single, inputs, v_tgts):
    lr_start, lr_end = int(lr_on / dt), int(lr_stop / dt)
    t_rec = h.Vector().record(h._ref_t)
    loss_opt = 1e60

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

        v_outs = np.array(cell.v_outs)
        v_tgts_clip = np.clip(v_tgts, a_min=None, a_max=v_clip_max)[:, lr_start:lr_end]
        v_outs_clip = np.clip(v_outs, a_min=None, a_max=v_clip_max)[:, lr_start:lr_end]
        
        loss = 0.5 * np.sum((v_tgts_clip - v_outs_clip)**2) / (N_out * (lr_end - lr_start))

        if loss < loss_opt:
            loss_opt = loss

            ### Plot optimal results ###
            for i in range(N_out):
                fig, ax = plt.subplots()
                ax.plot(t_rec, v_outs[i], label='Learned')
                ax.plot(t_rec, v_tgts[i], label='Target', linestyle='dashed')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Voltage (mV)')
                ax.legend(loc='upper center', frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title(f'epoch: {iepoch:03d}')
                fig.tight_layout()
                fig.savefig(os.path.join(OUTPUT_PATH, f"Optimal_{i}.png"))
                plt.close(fig)

        ### Compute dw ###
        dLtdv = -(v_tgts_clip - v_outs_clip)
        dw = cell.get_dw(dLtdv, lr_start, lr_end)
        dw *= alpha
        dw[N_apic:] *= alpha_dend_mul
        if dw_apic_min is not None or dw_apic_max is not None:
            dw_apic = np.clip(dw[:N_apic], a_min=dw_apic_min, a_max=dw_apic_max)
        else:
            dw_apic = dw[:N_apic]
        if dw_dend_min is not None or dw_dend_max is not None:
            dw_dend = np.clip(dw[N_apic:], a_min=dw_dend_min, a_max=dw_dend_max)
        else:
            dw_dend = dw[N_apic:]
        dw = np.append(dw_apic, dw_dend)
        cell.update_weights(dw)
    

if __name__ == '__main__':
    h.dt = dt
    h.celsius = 36

    config = {
        'seed': random_seed,
        'N_syn': N_syn,
        'seg_outs': seg_outs,
        'v_rest': v_rest,
        'K_max_t': K_max_t, 'K_filename': K_filename,
        'w_clip_min': w_clip_min, 'w_clip_max': w_clip_max,
        'device': DEVICE,
    }
    cell = Single(config)

    inputs = []
    for _ in range(N_apic_cluster):
        # background noise before high-freq
        bg_list_1 = np.where(rng.random(int(stim_on / dt)) < dt * bg_freq / 1000)[0] * dt
        # high-freq input
        t_list = np.where(rng.random(int(stim_dur / dt)) < dt * burst_freq / 1000)[0] * dt + stim_on
        # background noise after high-freq
        bg_list_2 = np.where(rng.random(int((tstop - stim_on - stim_dur) / dt)) < dt * bg_freq / 1000)[0] * dt + stim_on + stim_dur
        # combine
        t_list = np.sort(np.concatenate([bg_list_1, t_list, bg_list_2]))
        inputs.append(t_list)
    for _ in range(N_apic_bg):
        # background noise before low-freq
        bg_list_1 = np.where(rng.random(int(stim_on / dt)) < dt * bg_freq / 1000)[0] * dt
        # low-freq input
        t_list = np.where(rng.random(int(stim_dur / dt)) < dt * low_freq / 1000)[0] * dt + stim_on
        # background noise after low-freq
        bg_list_2 = np.where(rng.random(int((tstop - stim_on - stim_dur) / dt)) < dt * bg_freq / 1000)[0] * dt + stim_on + stim_dur
        # combine
        t_list = np.sort(np.concatenate([bg_list_1, t_list, bg_list_2]))
        inputs.append(t_list)
    for _ in range(N_syn - N_apic):
        # background noise before low-freq
        bg_list_1 = np.where(rng.random(int(stim_on / dt)) < dt * bg_freq / 1000)[0] * dt
        # low-freq input
        t_list = np.where(rng.random(int(stim_dur / dt)) < dt * low_freq / 1000)[0] * dt + stim_on
        # background noise after low-freq
        bg_list_2 = np.where(rng.random(int((tstop - stim_on - stim_dur) / dt)) < dt * bg_freq / 1000)[0] * dt + stim_on + stim_dur
        # combine
        t_list = np.sort(np.concatenate([bg_list_1, t_list, bg_list_2]))
        inputs.append(t_list)
    # burst cluster on distal apical, others randomly scatter on apic & basal
    isegs = [cell.seg2iseg(cell.cell, cell.cell.apic[37](0.5)),
             cell.seg2iseg(cell.cell, cell.cell.apic[51](0.5)),
             cell.seg2iseg(cell.cell, cell.cell.apic[60](0.5)),]
    conn_apic_cluster = rng.choice(isegs, N_apic_cluster)
    conn_apic_bg = rng.integers(cell.nseg_dend, cell.nseg_dend + cell.nseg_apic, N_apic_bg)
    conn_apic = np.append(conn_apic_cluster, conn_apic_bg)
    conn_dend = rng.integers(0, cell.nseg_dend, N_syn - N_apic)
    conn = np.append(conn_apic, conn_dend)
    # target weights
    weights_target_apic_cluster = rng.uniform(w_target_apic_cluster_min, w_target_apic_cluster_max, N_apic_cluster)
    weights_target_apic_bg = rng.normal(0., w_target_scale_apic_bg, N_apic_bg)
    weights_target_apic = np.append(weights_target_apic_cluster, weights_target_apic_bg)
    weights_target_dend = rng.normal(0., w_target_scale_dend, N_syn - N_apic)
    weights_target = np.append(weights_target_apic, weights_target_dend)

    cell.init_conn(conn, weights_target)

    t_rec, v_tgts, _ = gen_target(cell, inputs)

    ### Reinitialize weights ###
    rng = np.random.default_rng(seed=args.init_seed)
    # init weights
    weights_init_apic = rng.normal(0., w_init_scale_apic, N_apic)
    weights_init_dend = rng.normal(0., w_init_scale_dend, N_syn - N_apic)
    weights_init = np.append(weights_init_apic, weights_init_dend)
    cell.set_weights(weights_init)

    ### Train ###
    train(cell, inputs, v_tgts)

    print("Done")
    h.quit()
