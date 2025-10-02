import numpy as np
import torch
from neuron import h
from tqdm import tqdm

class Single_hh:
    def __init__(self, config):

        self.rng = np.random.default_rng(seed=config['seed'])

        self.N_out = 1                          # soma[0](0.5)
        self.N_hh = config['N_hh']              # soma[0](0.5)
        self.N_e, self.N_i = config['N_e'], config['N_i']
        self.N_syn = self.N_hh + self.N_e + self.N_i
        self.N = self.N_out + self.N_syn

        self.v_rest = config['v_rest']          # resting potential
        self.K_max_t = config['K_max_t']        # transfer impedance maximum time window
        self.K_filename = config['K_filename']
        self.w_e_min, self.w_e_max = config['w_e_min'], config['w_e_max']
        self.w_i_min, self.w_i_max = config['w_i_min'], config['w_i_max']

        self.device = config['device']

        self._create_cell()
        self._connect_cells()

    def _setup_hpc(self, model, morph):
        cell = getattr(h, model)()
        nl = h.Import3d_Neurolucida3()
        nl.quiet = 1
        nl.input(morph)
        imprt = h.Import3d_GUI(nl, 0)
        imprt.instantiate(cell)
        cell.indexSections(imprt)
        cell.geom_nsec()
        cell.geom_nseg()
        cell.delete_axon()
        cell.insertChannel()
        # cell.init_biophys()
        cell.init_rc()
        cell.biophys()
        return cell

    def _create_cell(self):
        h.load_file("import3d.hoc")
        h.load_file("PassiveHPC.hoc")

        self.cell = self._setup_hpc("PassiveHPC", "2013_03_06_cell11_1125_H41_06.asc")
        for sec in self.cell.all:
            sec.e_pas = self.v_rest
        self.v_out = h.Vector().record(self.cell.soma[0](0.5)._ref_v)
    
    def _cal_K(self):
        self.K_len = int(self.K_max_t / h.dt)
        try:
            tmp_K = np.load(self.K_filename)
            print(f"read K from {self.K_filename}")
            assert tmp_K.shape == (self.N, self.N_syn, self.K_len), "Unexpected shape of K"
        except FileNotFoundError:
            print(f"{self.K_filename} not found, computing K")
            tmp_cell = self._setup_hpc("PassiveHPC", "2013_03_06_cell11_1125_H41_06.asc")
            for sec in tmp_cell.all:
                sec.e_pas = self.v_rest
            tmp_K = np.zeros((self.N, self.N_syn, self.K_len), dtype=np.float32)
            v_list = []
            for i in range(self.N):
                dend_id = self.conn[i]
                loc = self.conn_loc[i]
                v = h.Vector()
                if i < self.N_out:                                  # somatic output
                    v.record(tmp_cell.soma[dend_id](loc)._ref_v)
                elif i < self.N_out + self.N_hh:                    # soma hh
                    v.record(tmp_cell.soma[dend_id](loc)._ref_v)
                elif dend_id < self.total_dend:                     # dend
                    v.record(tmp_cell.dend[dend_id](loc)._ref_v)
                else:                                               # apic
                    apic_id = dend_id - self.total_dend
                    v.record(tmp_cell.apic[apic_id](loc)._ref_v)
                v_list.append(v)

            for j in tqdm(range(self.N_syn)):
                dend_id = self.conn[self.N_out + j]
                loc = self.conn_loc[self.N_out + j]
                if j < self.N_hh:                                   # soma hh
                    clamp = h.IClamp(tmp_cell.soma[dend_id](loc))
                elif dend_id < self.total_dend:                     # dend
                    clamp = h.IClamp(tmp_cell.dend[dend_id](loc))
                else:                                               # apic
                    apic_id = dend_id - self.total_dend
                    clamp = h.IClamp(tmp_cell.apic[apic_id](loc))
                clamp.delay = 0
                clamp.dur = h.dt
                clamp.amp = 1. / h.dt
                h.finitialize(self.v_rest)
                h.continuerun(self.K_max_t)
                for i in range(self.N):
                    tmp_K[i, j] = np.array(v_list[i])[1: 1 + self.K_len][::-1] - self.v_rest   # already reversed
                clamp.amp = 0

            print(f"save K to {self.K_filename}")
            np.save(self.K_filename, tmp_K.astype(np.float16))

            del tmp_cell
        
        self.K = torch.tensor(tmp_K, dtype=torch.float32, device=torch.device(self.device))

    def _connect_cells(self):
        self.total_dend = len(self.cell.dend)
        self.total_apic = len(self.cell.apic)
        # connection matrix input-to-cell dendrites
        self.conn_out = np.array([0,], dtype=np.int32)
        self.conn_loc_out = np.array([0.5,])
        self.conn_hh = np.array([0,] * self.N_hh, dtype=np.int32)
        self.conn_loc_hh = np.array([0.5,] * self.N_hh)
        self.conn_e = self.rng.integers(0, self.total_dend + self.total_apic, (self.N_e,))
        self.conn_loc_e = self.rng.random((self.N_e,))
        self.conn_i = self.rng.integers(0, self.total_dend + self.total_apic, (self.N_i,))
        self.conn_loc_i = self.rng.random((self.N_i,))
        self.conn = np.concatenate((self.conn_out, self.conn_hh, self.conn_e, self.conn_i))
        self.conn_loc = np.concatenate((self.conn_loc_out, self.conn_loc_hh, self.conn_loc_e, self.conn_loc_i))

        # transfer impedance matrix
        self._cal_K()

        # weight matrix input-to-cell dendrites
        w_hh = np.ones((self.N_hh,))
        w_e = self.rng.uniform(0.1*1e-3, 1.*1e-3, (self.N_e,))
        w_i = self.rng.uniform(-1.*1e-3, -0.1*1e-3, (self.N_i,))
        self.w = np.concatenate((w_hh, w_e, w_i))

        # soma hh
        self.synlist_hh = []
        for i in range(self.N_hh):
            soma_id = self.conn_hh[i]                               # 0
            loc = self.conn_loc_hh[i]                               # 0.5
            if i == 0:  # hh
                syn = h.HH2_modified(self.cell.soma[soma_id](loc))
                syn.gnabar = 0.08
                syn.gkbar = 0.04
                syn.vtraub = -60
                syn.ena = 50
                syn.ek = -80
            else:   # im
                syn = h.IM_modified(self.cell.soma[soma_id](loc))
                syn.gkbar = 0.003
                syn.taumax = 200
                syn.ek = -80
            syn.dv = 1e-3
            syn.w = w_hh[i]                                             # 1.
            syn.seg_area = self.cell.soma[soma_id](loc).area() * 1e-8   # um2 to cm2
            syn.didv_clip = 1e3                                         # no clip
            self.synlist_hh.append(syn)

        # input to cell
        # AMPA + NMDA
        self.synlist_e = []
        self.stimlist_e = []
        self.nclist_e = []
        for i in range(self.N_e):
            dend_id = self.conn_e[i]
            loc = self.conn_loc_e[i]
            if dend_id < self.total_dend:
                syn = h.Exp2Syn_exc(self.cell.dend[dend_id](loc))
            else:
                apic_id = dend_id - self.total_dend
                syn = h.Exp2Syn_exc(self.cell.apic[apic_id](loc))
            syn.AMPA_tau1, syn.AMPA_tau2 = 0.1, 2
            syn.NMDA_tau1, syn.NMDA_tau2 = 2, 75
            syn.AMPA_e, syn.NMDA_e = 0, 0
            syn.NMDA_C = 0.267
            syn.NMDA_rho = 0.062
            syn.r_na = 2
            syn.w = w_e[i]
            self.synlist_e.append(syn)

            stim = h.VecStim()
            self.stimlist_e.append(stim)
            nc = h.NetCon(stim, syn)
            nc.delay = 1
            nc.weight[0] = 1
            self.nclist_e.append(nc)

        # GABA
        self.synlist_i = []
        self.stimlist_i = []
        self.nclist_i = []
        for i in range(self.N_i):
            dend_id = self.conn_i[i]
            loc = self.conn_loc_i[i]
            if dend_id < self.total_dend:
                syn = h.Exp2Syn_inh(self.cell.dend[dend_id](loc))
            else:
                apic_id = dend_id - self.total_dend
                syn = h.Exp2Syn_inh(self.cell.apic[apic_id](loc))
            syn.GABA_tau1, syn.GABA_tau2 = 1, 5
            syn.GABA_e = -75
            syn.w = w_i[i]
            self.synlist_i.append(syn)

            stim = h.VecStim()
            self.stimlist_i.append(stim)
            nc = h.NetCon(stim, syn)
            nc.delay = 1
            nc.weight[0] = 1
            self.nclist_i.append(nc)

        self.synlist = self.synlist_hh + self.synlist_e + self.synlist_i

    def set_stim(self, inputs):
        assert len(inputs) == self.N_e + self.N_i
        for t_list, stim in zip(inputs[:self.N_e], self.stimlist_e):
            t_vec = h.Vector(t_list)
            stim.play(t_vec)
        for t_list, stim in zip(inputs[self.N_e:], self.stimlist_i):
            t_vec = h.Vector(t_list)
            stim.play(t_vec)
        self._reset_records()
    
    def _reset_records(self):
        self.It = torch.tensor(np.array([]).reshape((self.N_syn, 0)), 
                               dtype=torch.float32, device=torch.device(self.device))       # (N_syn, min(t, K_max_t))
        self.dItdv = torch.tensor(np.array([]).reshape((self.N_syn, 0)), 
                                  dtype=torch.float32, device=torch.device(self.device))    # (N_syn, min(t, K_max_t))
        self.dVtdw = torch.tensor(np.zeros((self.N_syn, self.N, 1)), 
                                  dtype=torch.float32, device=torch.device(self.device))    # (N_syn, N, min(t, K_max_t)), [i, j] = dvjdwi
        self.dVouttdw = np.zeros((self.N_syn, 1))                                           # (N_syn, t)
        self.v_pred = [self.v_rest,]

    def update_pred(self, tstep):
        # called after each timestep advance
        it = torch.tensor([syn.pure_i for syn in self.synlist], 
                          dtype=torch.float32, device=torch.device(self.device))
        self.It = torch.hstack((self.It, it[:, torch.newaxis]))[:, -self.K_len:]            # (N_syn, min(t, K_max_t))
        
        K_out_conv = self.K[0, :, -tstep:]
        w_device = torch.tensor(self.w[:, np.newaxis], device=self.device)
        self.v_pred.append(torch.sum(K_out_conv * self.It * torch.abs(w_device)).cpu() * h.dt + self.v_rest)

    def update_dvdw(self, tstep):
        # called after each timestep advance
        it = torch.tensor([syn.pure_i for syn in self.synlist_hh + self.synlist_e] +
                          [-syn.pure_i for syn in self.synlist_i],
                          dtype=torch.float32, device=torch.device(self.device))
        self.It = torch.hstack((self.It, it[:, torch.newaxis]))[:, -self.K_len:]                    # (N_syn, min(t, K_max_t))
        ditdv = torch.tensor([syn.didv for syn in self.synlist],
                             dtype=torch.float32, device=torch.device(self.device))
        self.dItdv = torch.hstack((self.dItdv, ditdv[:, torch.newaxis]))[:, -self.K_len:]           # (N_syn, min(t, K_max_t))
        
        K_conv = self.K[:, :, -tstep:]                                                              # (N, N_syn, min(t, K_max_t))
        dvtdw = torch.einsum('jit,it->ij', K_conv, self.It) * h.dt                                  # (N_syn, N)
        dItdw_conv_0tot_1 = self.dItdv[torch.newaxis, :, :] * self.dVtdw[:, self.N_out:]            # (N_syn, N_syn, min(t, K_max_t))
        dvtdw += torch.einsum('jkt,ikt->ij', K_conv, dItdw_conv_0tot_1) * h.dt

        self.dVtdw = torch.dstack((self.dVtdw, dvtdw[:, :, torch.newaxis]))[:, :, -self.K_len:]     # (N_syn, N, min(t, K_max_t))
        dvouttdw = dvtdw[:, 0].cpu().numpy()                                                        # (N_syn,)
        self.dVouttdw = np.hstack((self.dVouttdw, dvouttdw[:, np.newaxis]))                         # (N_syn, t)

    def get_dw(self, dLtdv, lr_start, lr_end):
        # called after each run
        assert(lr_end - lr_start == len(dLtdv))
        dw = -np.sum(dLtdv * self.dVouttdw[:, lr_start:lr_end], axis=1) / (lr_end - lr_start)
        # mask
        dw[:self.N_hh] = 0.
        return dw

    def update_weights(self, dw):
        assert dw.shape == (self.N_syn,)
        dw = np.array(dw)
        # mask
        dw[:self.N_hh] = 0
        self.w += dw
        # clip w
        w_e = self.w[self.N_hh: self.N_hh + self.N_e]
        w_i = self.w[self.N_hh + self.N_e:]
        w_e = np.clip(w_e, a_min=self.w_e_min, a_max=self.w_e_max)
        w_i = np.clip(w_i, a_min=self.w_i_min, a_max=self.w_i_max)
        self.w[self.N_hh: self.N_hh + self.N_e] = w_e
        self.w[self.N_hh + self.N_e:] = w_i
        self.set_weights()
    
    def set_weights(self, w=None):
        # update weights to syns
        if w is None:
            w = self.w
        else:
            assert w.shape == (self.N_syn,)
        w = np.array(w)
        w_hh = w[:self.N_hh]
        w_e = w[self.N_hh: self.N_hh + self.N_e]
        w_i = w[self.N_hh + self.N_e:]
        w_e = np.clip(w_e, a_min=self.w_e_min, a_max=self.w_e_max)
        w_i = np.clip(w_i, a_min=self.w_i_min, a_max=self.w_i_max)
        self.w[self.N_hh: self.N_hh + self.N_e] = w_e
        self.w[self.N_hh + self.N_e:] = w_i
        for i in range(self.N_hh):
            self.synlist_hh[i].w = w_hh[i]
        for i in range(self.N_e):
            self.synlist_e[i].w = w_e[i]
        for i in range(self.N_i):
            self.synlist_i[i].w = w_i[i]
    
    def save_weights(self, path):
        np.save(path, self.w)
