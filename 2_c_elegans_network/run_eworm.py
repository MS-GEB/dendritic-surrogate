import os
import copy
import numpy as np
from neuron import h
from network_eworm import Network
import matplotlib.pyplot as plt
import time
import json


def load_json(file_name):
    with open(file_name, 'r+') as f:
        data_dic = json.load(f)
    return data_dic

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', 
                    help="'train' or 'test'")
parser.add_argument('--ngpu', type=int, default=1, 
                    help="number of gpus to use")
parser.add_argument('--percise', action='store_true', 
                    help="whether to additionally use global gradients")
parser.add_argument('--k_mul', type=int, default=5, 
                    help="temporal downsampling rate for gradient steps")
parser.add_argument('--adam', action='store_true', 
                    help="whether to use Adam optimizer")
args, _ = parser.parse_known_args()

OUTPUT_PATH = 'output'
os.makedirs(OUTPUT_PATH, exist_ok=True)
RUN_MODE = args.mode            # 'test' or 'train'

config_file = "000_circuit_search_config.json"
connection_file = "sample_#0_circuit_old.pkl"
config = load_json(config_file)
sim_config = config["sim_config"]

dt = sim_config["dt"]           # time step
v_r = sim_config["v_init"]      # resting potential
ngpu = args.ngpu                # number of GPUs used
K_max_t = 120                   # K time window
K_mul = args.k_mul              # 
K_filename = os.path.join(OUTPUT_PATH, 'K.pkl')   # transfer impedance file name
K_nblock = 136                  # number of diagnal sub-blocks of K
tstop = sim_config["tstop"]     # simulation time
epochs = 50                     # max training epochs
lr_start = int(0 / dt)          # learning start step
lr_end = int(tstop / dt)        # learning end step
alpha_w0 = 1e-5                 # initial learning rate for w
alpha_x0 = 3e-2                 # initial learning rate for x
alpha_d = 1/125	                # learning rate decay constant
w_gap_max = None
w_gap_min = 1e-9
w_syn_max = None
w_syn_min = -2
random_seed = 42
PERCISE = args.percise
ADAM_W = args.adam
ADAM_X = args.adam
PRINT_TIMESTEP = True


def cal_corrcoef_dLtdv(output_vs, target_corrcoef, lr_start, lr_end):
    output_corrcoef = np.corrcoef(output_vs[:, lr_start: lr_end])
    dLdcorr = np.array(-(target_corrcoef - output_corrcoef))  # shape (N_output, N_output)
    dLdcorr[np.isnan(dLdcorr)] = 0.
    N_output = output_vs.shape[0]
    dcorrdxt = np.zeros((N_output, N_output, lr_end - lr_start))
    for i in range(N_output):
        x = output_vs[i, lr_start: lr_end]
        mean_x = np.mean(x)
        std_x = np.std(x)
        for j in range(N_output):
            y = output_vs[j, lr_start: lr_end]
            mean_y = np.mean(y)
            std_y = np.std(y)
            cov_xy = np.mean((x - mean_x) * (y - mean_y))
            dcov_xydxt = ((y - mean_y) - np.mean(y - mean_y)) / (lr_end - lr_start)
            dstd_xdxt = ((x - mean_x) - np.mean(x - mean_x)) / (std_x * (lr_end - lr_start))
            dcorrxydxt = (dcov_xydxt * std_x * std_y - cov_xy * (dstd_xdxt * std_y)) / (std_x * std_y) ** 2
            dcorrdxt[i, j, :] = dcorrxydxt
    dLtdv = np.sum(dLdcorr[:, :, np.newaxis] * dcorrdxt, axis=1)   # shape (N_output, lr_end - lr_start)
    return dLtdv


def train(net: Network, output_names, input_is, target):
    net.set_outputs(output_names)
    x = np.copy(input_is)

    # Adam params
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-9
    if ADAM_W:
        adam_m_w = 0.
        adam_v_w = 0.
        beta_1_t_w = 1.
        beta_2_t_w = 1.
    if ADAM_X:
        adam_m_x = 0.
        adam_v_x = 0.
        beta_1_t_x = 1.
        beta_2_t_x = 1.

    opt_epoch = -1
    opt_w = net.w.numpy()
    opt_x = np.copy(x)
    opt_mean_error = 1e100
    if ADAM_W:
        opt_adam_params_w = copy.deepcopy((adam_m_w, adam_v_w, beta_1_t_w, beta_2_t_w))
    if ADAM_X:
        opt_adam_params_x = copy.deepcopy((adam_m_x, adam_v_x, beta_1_t_x, beta_2_t_x))
    alpha_multiplier = 1.

    train_error = []
    for epoch in range(epochs):
        start_time = time.time()
        alpha_w = alpha_w0 / (1 + alpha_d * epoch)
        alpha_x = alpha_x0 / (1 + alpha_d * epoch)

        net._reset_lr_records()

        h.t = 0
        h.tstop = tstop
        h.secondorder = 0
        h.finitialize(v_r)
        h.fcurrent()

        # stimulation
        tstep = 0
        pre_time = time.time()
        while h.t < h.tstop:
            if PRINT_TIMESTEP:
                now_time = time.time()
                print(f'epoch: {epoch}, t: {h.t}, used: {now_time-pre_time:.3f}s', end='\r')
                pre_time = now_time
            for ind, id in enumerate(net.input_ids):
                net.input_synlist[id].amp = x[ind, tstep]
            h.fadvance()
            tstep += 1
            if tstep % K_mul == 0:
                net.update_dvdw(tstep // K_mul, percise=PERCISE)
        print('')

        output_vs = []
        for cn in output_names:
            id = net.cells_id_dic[cn]
            output_vs.append(np.array(net.output_vlist[id]))
        output_vs = np.array(output_vs)     # shape (N_output, tstep)
        output_corrcoef = np.corrcoef(output_vs[:, lr_start: lr_end])
        mean_error = np.abs(target - output_corrcoef)
        mean_error[np.isnan(mean_error)] = 0.
        mean_error = np.mean(mean_error ** 2)
        dLtdv = cal_corrcoef_dLtdv(output_vs, target, lr_start, lr_end)

        train_error.append(mean_error)
        logger.info(f'epoch: {epoch}, mean error: {mean_error:.5g}')

        if mean_error < opt_mean_error:
            opt_epoch = epoch
            opt_w = net.w.numpy().copy()
            opt_x = np.copy(x)
            opt_mean_error = mean_error
            if ADAM_W:
                opt_adam_params_w = copy.deepcopy((adam_m_w, adam_v_w, beta_1_t_w, beta_2_t_w))
            if ADAM_X:
                opt_adam_params_x = copy.deepcopy((adam_m_x, adam_v_x, beta_1_t_x, beta_2_t_x))
            
            # plot corr
            fig, ax = plt.subplots(figsize=(24,24), dpi=200)
            ax.set_yticks(range(len(output_names)))
            ax.set_yticklabels(output_names, fontsize=20)
            ax.set_xticks(range(len(output_names)))
            ax.set_xticklabels(output_names, fontsize=18)
            im = ax.imshow(np.corrcoef(output_vs[:, lr_start: lr_end]), cmap=plt.cm.cool, vmin=-1, vmax=1)
            plt.xticks(rotation=45)
            plt.rcParams['font.size'] = 30
            plt.colorbar(im, fraction=0.0452)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_PATH, "Optimal_corr.png"))
            plt.close(fig)

            np.save(os.path.join(OUTPUT_PATH, "w_optimal.npy"), opt_w)
            np.save(os.path.join(OUTPUT_PATH, "x_optimal.npy"), opt_x)
            logger.info('optimal weights & x saved')
        elif epoch - opt_epoch >= 10:
            # retreat
            logger.info(f'no improvement since epoch {opt_epoch} for 10 epochs, restore weights & x, learning rate *= 0.8')
            net.set_weights(opt_w)
            x = np.copy(opt_x)
            if ADAM_W:
                adam_m_w, adam_v_w, beta_1_t_w, beta_2_t_w = copy.deepcopy(opt_adam_params_w)
            if ADAM_X:
                adam_m_x, adam_v_x, beta_1_t_x, beta_2_t_x = copy.deepcopy(opt_adam_params_x)
            opt_epoch = epoch + 1
            alpha_multiplier *= 0.8
            continue
        alpha_w *= alpha_multiplier
        alpha_x *= alpha_multiplier

        dw, dx = net.get_dw_dx(dLtdv, lr_start, lr_end)

        if ADAM_W:
            adam_m_w = beta_1 * adam_m_w + (1. - beta_1) * dw
            adam_v_w = beta_2 * adam_v_w + (1. - beta_2) * dw * dw
            beta_1_t_w = beta_1_t_w * beta_1
            beta_2_t_w = beta_2_t_w * beta_2
            m_hat_w = adam_m_w / (1. - beta_1_t_w)
            v_hat_w = adam_v_w / (1. - beta_2_t_w)
            dw = m_hat_w / (np.sqrt(v_hat_w) + epsilon)
        if ADAM_X:
            adam_m_x = beta_1 * adam_m_x + (1. - beta_1) * dx
            adam_v_x = beta_2 * adam_v_x + (1. - beta_2) * dx * dx
            beta_1_t_x = beta_1_t_x * beta_1
            beta_2_t_x = beta_2_t_x * beta_2
            m_hat_x = adam_m_x / (1. - beta_1_t_x)
            v_hat_x = adam_v_x / (1. - beta_2_t_x)
            dx = m_hat_x / (np.sqrt(v_hat_x) + epsilon)
        dw *= alpha_w
        dx *= alpha_x
        dx = np.array([np.interp(np.arange(lr_start, lr_end), np.arange(lr_start, lr_end, K_mul), dxi) for dxi in dx])
        dx += -3e-2 * x[:, lr_start: lr_end]

        net.update_weights(dw)
        x[:, lr_start:lr_end] += dx
        x = np.clip(x, a_min=-0.2, a_max=0.2)

        logger.info(f'time cost: {time.time()-start_time}')

    return train_error


def test(net: Network, output_names, input_is):
    net.set_outputs(output_names)
    x = np.copy(input_is)

    h.t = 0
    h.tstop = tstop
    h.secondorder = 0
    h.finitialize(v_r)
    h.fcurrent()

    # stimulation
    tstep = 0
    while h.t < h.tstop:
        if PRINT_TIMESTEP:
            print(f't: {h.t}/{h.tstop}', end='\r')
        for ind, cid in enumerate(net.input_ids):
            net.input_synlist[cid].amp = x[ind, tstep]
        h.fadvance()
        tstep += 1
    print('')

    output_vs = []
    for cn in output_names:
        id = net.cells_id_dic[cn]
        output_vs.append(np.array(net.output_vlist[id]))
    output_vs = np.array(output_vs)     # shape (N_output, tstep)
    return output_vs


if __name__ == "__main__":
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(OUTPUT_PATH, f'log.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('\n###################################################################################################\n')

    np.random.seed(random_seed)

    # setup h
    h.load_file('stdrun.hoc')
    h.dt = dt

    input_names = config["search_config"]["input_cell_names"]
    print("input names:", input_names)

    output_names_config = config["search_config"]["output_cell_names"]
    with open(os.path.join("components", "cb2022_data", "Ca_corr_mat_cell_name.txt")) as f:
        output_names_target = f.read().split("\t")
    ca_corr = np.loadtxt(os.path.join("components", "cb2022_data", "Ca_corr_mat.txt"))
    output_names = []
    output_ids = []
    for i, cn in enumerate(output_names_target):
        if cn in output_names_config:
            output_names.append(cn)
            output_ids.append(i)
    target = ca_corr[output_ids,:][:,output_ids]
    print("output names:", output_names)
    print("target:", target)

    # plot target corr
    fig, ax = plt.subplots(figsize=(24,24), dpi=200)
    ax.set_yticks(range(len(output_names)))
    ax.set_yticklabels(output_names, fontsize=20)
    ax.set_xticks(range(len(output_names)))
    ax.set_xticklabels(output_names, fontsize=18)
    im = ax.imshow(target, cmap=plt.cm.cool, vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH, "target_corr.png"))
    plt.close(fig)

    net_config = config["config"]
    lr_config = {
        'v_r': v_r,
        'ngpu': ngpu,
        'K_max_t': K_max_t,
        'K_filename': K_filename,
        'K_nblock': K_nblock,
        'K_mul': K_mul,
        'w_gap_max': w_gap_max,
        'w_gap_min': w_gap_min,
        'w_syn_max': w_syn_max,
        'w_syn_min': w_syn_min
    }

    eworm_net = Network(net_config, lr_config, random_seed)
    eworm_net.read_cells_neurite_connection(connection_file, input_names)

    ### train ###
    if RUN_MODE == 'train':
        init_weight_path = "w_init.npy"
        init_input_path = "x_init.npy"
        if os.path.exists(init_weight_path) and os.path.exists(init_input_path):
            init_w = np.load(init_weight_path)
            init_x = np.load(init_input_path)
            eworm_net.set_weights(init_w)
            input_is = np.copy(init_x)
            print("Initial weights and inputs are loaded")
        train_error = train(eworm_net, output_names, input_is, target)

    ### test ###
    opt_w_path = os.path.join(OUTPUT_PATH, "w_optimal.npy")
    opt_x_path = os.path.join(OUTPUT_PATH, "x_optimal.npy")
    if os.path.exists(opt_w_path) and os.path.exists(opt_x_path):
        opt_w = np.load(opt_w_path)
        opt_x = np.load(opt_x_path)
        eworm_net.set_weights(opt_w)
        input_is = np.copy(opt_x)
        print("Optimal weights and inputs are loaded")

    output_vs = test(eworm_net, output_names, input_is)
    sim_corr = np.corrcoef(output_vs[:, lr_start: lr_end])
    fig = plt.figure(figsize=(48,24), dpi=200)
    ax = fig.add_subplot(121)
    ax.set_title("Simulation", fontsize=40)
    ax.set_yticks(range(len(output_names)))
    ax.set_yticklabels(output_names, fontsize=20)
    ax.set_xticks(range(len(output_names)))
    ax.set_xticklabels(output_names, fontsize=18)
    im = ax.imshow(sim_corr, cmap=plt.cm.cool, vmin = -1, vmax = 1)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    ax = fig.add_subplot(122)
    mean_error = np.abs(target - sim_corr)
    mean_error[np.isnan(mean_error)] = 0.
    mean_error = 0.5 * np.mean(mean_error ** 2)
    ax.set_title(f"Experiment mse {mean_error * 2}", fontsize=40)
    ax.set_yticks(range(len(output_names)))
    ax.set_yticklabels(output_names, fontsize=20)
    ax.set_xticks(range(len(output_names)))
    ax.set_xticklabels(output_names, fontsize=18)
    im = ax.imshow(target, cmap=plt.cm.cool, vmin = -1, vmax = 1)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH, "test_corr.png"))
    plt.close(fig)
