import numpy as np
import torch
from neuron import h
from tqdm import tqdm

MODEL_NAME = 'PassiveHPC'
MODEL_PATH = f'models/{MODEL_NAME}.hoc'
MORPH_PATH = 'morphs/2013_03_06_cell11_1125_H41_06.asc'

class Single_hh:
    def __init__(self, config):
        self.rng = np.random.default_rng(seed=config['seed'])

        self.seg_outs = config['seg_outs']
        self.N_out = len(self.seg_outs)
        self.N_syn = config['N_syn']

        self.v_rest = config['v_rest']                      # resting potential
        self.K_max_t = config['K_max_t']                    # transfer impedance maximum time window
        self.K_filename = config['K_filename']
        self.K_mul = config['K_mul']
        self.w_clip_min, self.w_clip_max = config['w_clip_min'], config['w_clip_max']

        self.device = torch.device(config['device'])

        self._create_cell()

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
        for sec in cell.all:
            sec.e_pas = self.v_rest
        return cell

    def _create_cell(self):
        h.load_file('import3d.hoc')
        h.load_file(MODEL_PATH)

        self.cell = self._setup_hpc(MODEL_NAME, MORPH_PATH)
        self.mechs = {'soma': ['HH2_modified', 'IM2_modified',],}
        for sec in self.cell.soma:
            sec.insert('HH2_modified')
            sec.insert('IM2_modified')
        self.iseg_mech = []
        for sec in self.cell.soma:
            for seg in sec:
                seg.HH2_modified.gnabar = 0.08
                seg.HH2_modified.gkbar = 0.04
                seg.HH2_modified.vtraub = -60
                seg.HH2_modified.ena = 50
                seg.HH2_modified.ek = -80
                seg.IM2_modified.gkbar = 0.003
                seg.IM2_modified.taumax = 200
                seg.IM2_modified.ek = -80
                self.iseg_mech.append(self.seg2iseg(self.cell, seg))
        self.N_mech = len(self.iseg_mech)

        self.v_outs = []
        self.iseg_out = []
        for (secname, i, x) in self.seg_outs:
            seg = getattr(self.cell, secname)[int(i)](x)
            self.v_outs.append(h.Vector().record(seg._ref_v))
            self.iseg_out.append(self.seg2iseg(self.cell, seg))

        self.nsec_soma, self.nsec_dend, self.nsec_apic = len(self.cell.soma), len(self.cell.dend), len(self.cell.apic)
        self.nsec = len(list(self.cell.all))
        self.nseg_soma = np.sum([sec.nseg for sec in self.cell.soma])
        self.nseg_dend = np.sum([sec.nseg for sec in self.cell.dend])
        self.nseg_apic = np.sum([sec.nseg for sec in self.cell.apic])
        self.nseg = np.sum([sec.nseg for sec in self.cell.all])
        print(f"soma nsec: {self.nsec_soma}, dend nsec: {self.nsec_dend}, apic nsec: {self.nsec_apic}")
        print(f"total nsec:", self.nsec)
        print(f"soma nseg: {self.nseg_soma}, dend nseg: {self.nseg_dend}, apic nseg: {self.nseg_apic}")
        print(f"total nseg:", self.nseg)

        self.N = self.N_out + self.N_mech + self.N_syn
    
    def _cal_K(self):
        self.K_len = int(self.K_max_t / (h.dt * self.K_mul))
        try:
            tmp_K = np.load(self.K_filename)
            print(f"read K from {self.K_filename}")
            assert tmp_K.shape == (self.N, self.N, self.K_len), \
            f"Unexpected shape of K, , expect {(self.N, self.N, self.K_len)} got {tmp_K.shape}"
        except FileNotFoundError:
            print(f"{self.K_filename} not found, computing K")
            tmp_cell = self._setup_hpc(MODEL_NAME, MORPH_PATH)
            tmp_K = np.zeros((self.N, self.N, self.K_len), dtype=np.float32)
            
            self.isegs = self.iseg_out + self.iseg_mech + self.iseg_syn
            v_list = []
            for iseg in self.isegs:
                seg = self.iseg2seg(tmp_cell, iseg)
                v_list.append(h.Vector().record(seg._ref_v))

            old_dt = h.dt
            h.dt = old_dt * self.K_mul
            for j in tqdm(range(self.N)):
                seg = self.iseg2seg(tmp_cell, self.isegs[j])
                clamp = h.IClamp(seg)
                clamp.delay = 0
                clamp.dur = h.dt
                clamp.amp = 1. / h.dt
                h.finitialize(self.v_rest)
                h.continuerun(self.K_max_t)
                for i in range(self.N):
                    tmp_K[i, j] = np.array(v_list[i])[1: 1 + self.K_len][::-1] - self.v_rest   # already reversed
                clamp.amp = 0
            h.dt = old_dt

            print(f"save K to {self.K_filename}")
            np.save(self.K_filename, tmp_K.astype(np.float16))

            del tmp_cell
        
        self.K = torch.tensor(tmp_K, dtype=torch.float32, device=self.device)

    def iseg2seg(self, cell, iseg):
        nseg_cumsum = 0
        for sec in list(cell.soma) + list(cell.dend) + list(cell.apic):
            if nseg_cumsum + sec.nseg > iseg:
                loc = (1 + 2 * (iseg - nseg_cumsum)) / (2 * sec.nseg)
                return sec(loc)
            nseg_cumsum += sec.nseg
        raise ValueError(f"iseg {iseg} >= soma_nseg + dend_nseg + apic_nseg {self.nseg_soma + self.nseg_dend + self.nseg_apic}")
    
    def seg2iseg(self, cell, seg):
        iseg = 0
        for sec in list(cell.soma) + list(cell.dend) + list(cell.apic):
            if seg.sec.hname() == sec.hname():
                for i in range(sec.nseg):
                    if i / sec.nseg <= seg.x and (i + 1) / sec.nseg >= seg.x:
                        return iseg
                    iseg += 1
            else:
                iseg += sec.nseg
        raise ValueError(f"seg: {seg.sec.hname()}({seg.x:.3g}) is not a segment in cell's soma, dend or apic")

    def init_conn(self, conn, w):
        assert conn.shape == (self.N_syn,)
        self.iseg_syn = np.array(conn).tolist()
        assert w.shape == (self.N_syn,)
        self.w = np.array(w)

        # transfer impedance matrix
        self._cal_K()

        # input to cell
        self.synlist = []
        self.stimlist = []
        self.nclist = []
        for i in range(self.N_syn):
            iseg = self.iseg_syn[i]
            seg = self.iseg2seg(self.cell, iseg)
            if self.w[i] >= 0:
                syn = h.Exp2Syn_exc(seg)
                syn.AMPA_tau1, syn.AMPA_tau2 = 0.1, 2
                syn.NMDA_tau1, syn.NMDA_tau2 = 2, 75
                syn.AMPA_e, syn.NMDA_e = 0, 0
                syn.NMDA_C = 0.267
                syn.NMDA_rho = 0.062
                syn.r_na = 2
            else:
                syn = h.Exp2Syn_inh(seg)
                syn.GABA_tau1, syn.GABA_tau2 = 1, 5
                syn.GABA_e = -75
            syn.w = w[i]
            self.synlist.append(syn)
            stim = h.VecStim()
            self.stimlist.append(stim)
            nc = h.NetCon(stim, syn)
            nc.delay = 1
            nc.weight[0] = 1
            self.nclist.append(nc)

    def set_stim(self, inputs):
        assert len(inputs) == self.N_syn
        for t_list, stim in zip(inputs, self.stimlist):
            t_vec = h.Vector(t_list)
            stim.play(t_vec)
        self._reset_records()
    
    def _reset_records(self):
        self.It = torch.zeros((self.N, self.K_len), dtype=torch.float32, device=self.device)
        self.dItdv = torch.zeros((self.N, self.K_len), dtype=torch.float32, device=self.device)
        self.dVtdw = torch.zeros((self.N, self.N, self.K_len), dtype=torch.float32, device=self.device)
        self.dVouttdw = torch.zeros((self.N_syn, self.N_out, 1), dtype=torch.float32, device=self.device)
        self.v_preds = self.v_rest * np.ones((self.N_out, 1))
        w_out = torch.zeros((self.N_out,), dtype=torch.float32, device=self.device)
        w_mech = torch.ones((self.N_mech,), dtype=torch.float32, device=self.device)
        w_syn = torch.tensor(self.w, dtype=torch.float32, device=self.device)
        self.w_device = torch.cat((w_out, w_mech, w_syn))

        self.t_ptr = 0
        self.indices = torch.arange(self.K_len)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def update_pred(self, tstep):
        # called after timestep advance
        it_out = torch.zeros((self.N_out,), dtype=torch.float32, device=self.device)
        it_mech = []
        for secname in ['soma',]:
            for sec in getattr(self.cell, secname):
                for seg in sec:
                    it = 0
                    seg_area = seg.area()
                    for mechname in self.mechs[secname]:
                        it += getattr(seg, mechname).pure_i
                    it_mech.append(it * seg_area * 1e-2)
        it_mech = torch.tensor(it_mech, dtype=torch.float32, device=self.device)
        it_syn = torch.tensor([syn.pure_i for syn in self.synlist],
                              dtype=torch.float32, device=self.device)
        it = torch.cat((it_out, it_mech, it_syn))
        self.It[:, self.t_ptr] = it

        view_idx = torch.flip(self.t_ptr - self.indices, [0,])

        vt_preds = torch.sum(self.K[:self.N_out] * self.It[:, view_idx] * torch.abs(self.w_device[:, torch.newaxis]), axis=(1, 2)).cpu().numpy() * h.dt + self.v_rest
        self.v_preds = np.hstack([self.v_preds, vt_preds[:, np.newaxis]])

        self.t_ptr = (self.t_ptr + 1) % self.K_len

    def update_dvdw(self, tstep, percise=True):
        # called after timestep advance
        it_out = torch.zeros((self.N_out,), dtype=torch.float32, device=self.device)
        it_mech = []
        for secname in ['soma',]:
            for sec in getattr(self.cell, secname):
                for seg in sec:
                    it = 0
                    seg_area = seg.area()
                    for mechname in self.mechs[secname]:
                        it += getattr(seg, mechname).pure_i
                    it_mech.append(it * seg_area * 1e-2)
        it_mech = torch.tensor(it_mech, dtype=torch.float32, device=self.device)
        it_syn = torch.tensor([syn.pure_i if syn.w >= 0 else -syn.pure_i for syn in self.synlist],
                              dtype=torch.float32, device=self.device)
        it = torch.cat((it_out, it_mech, it_syn))
        self.It[:, self.t_ptr] = it

        if percise:
            ditdv_out = torch.zeros((self.N_out,), dtype=torch.float32, device=self.device)
            ditdv_mech = []
            for secname in ['soma',]:
                for sec in getattr(self.cell, secname):
                    for seg in sec:
                        ditdv = 0
                        seg_area = seg.area()
                        for mechname in self.mechs[secname]:
                            ditdv += getattr(seg, mechname).didv
                        ditdv_mech.append(ditdv * seg_area * 1e-2)
            ditdv_mech = torch.tensor(ditdv_mech, dtype=torch.float32, device=self.device)
            ditdv_syn = torch.tensor([syn.didv for syn in self.synlist],
                                     dtype=torch.float32, device=self.device)
            ditdv = torch.cat((ditdv_out, ditdv_mech, ditdv_syn))
            self.dItdv[:, self.t_ptr] = ditdv
        
        view_idx = torch.flip(self.t_ptr - self.indices, [0,])

        dvtdw = torch.einsum('jit,it->ij', self.K, self.It[:, view_idx]) * self.K_mul * h.dt

        if percise:
            dItdw_conv_0tot_1 = self.dItdv[torch.newaxis, :, :] * self.dVtdw
            dvtdw += torch.einsum('jkt,ikt->ij', self.K, dItdw_conv_0tot_1[:, :, view_idx]) * self.K_mul * h.dt
        
        self.dVtdw[:, :, self.t_ptr] = dvtdw
        self.dVouttdw = torch.dstack([self.dVouttdw, dvtdw[-self.N_syn:, :self.N_out, torch.newaxis]])
        self.t_ptr = (self.t_ptr + 1) % self.K_len

    def get_dw(self, dLtdv, lr_start, lr_end):
        # called after each run
        assert dLtdv.shape == (self.N_out, lr_end - lr_start)
        dLtdv = torch.asarray(dLtdv[:, ::self.K_mul], dtype=torch.float32, device=self.device)
        min_t_len = min(dLtdv.shape[-1], self.dVouttdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul].shape[-1])
        dw = -torch.sum(dLtdv[np.newaxis, :, :min_t_len] * self.dVouttdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul][:, :, :min_t_len], axis=(1, 2)) / (self.N_out * min_t_len)
        return dw.cpu().numpy()

    def update_weights(self, dw):
        assert dw.shape == (self.N_syn,)
        dw = np.array(dw)
        w_e_idx = self.w >= 0
        w_i_idx = self.w < 0
        self.w += dw
        self.w[np.logical_and(w_e_idx, self.w < 0)] = 0
        self.w[np.logical_and(w_i_idx, self.w >= 0)] = 0
        self.set_weights()
    
    def set_weights(self, w=None):
        # update weights to syns
        if w is not None:
            assert w.shape == (self.N_syn,)
            self.w = np.array(w)
        if self.w_clip_min is not None or self.w_clip_max is not None:
            self.w = np.clip(self.w, a_min=self.w_clip_min, a_max=self.w_clip_max)
        for i in range(self.N_syn):
            self.synlist[i].w = self.w[i]
    
    def save_weights(self, path):
        np.save(path, self.w)
