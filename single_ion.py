import numpy as np
import torch
from neuron import h
from tqdm import tqdm

class Single_ion:
    def __init__(self, config):

        self.rng = np.random.default_rng(seed=config['seed'])

        self.mode = config['mode']              # 'soma', 'all'
        self.N_out = 1                          # soma[0](0.5)
        self.ionconfig = config['ion_config']   # section name: ion name
        self.v_rest = config['v_rest']          # resting potential
        self.K_max_t = config['K_max_t']        # transfer impedance maximum time window
        self.K_filename = config['K_filename']
        self.w_min, self.w_max = config['w_min'], config['w_max']
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
        h.load_file("./model/PassiveHPC.hoc")

        self.cell = self._setup_hpc("PassiveHPC", "./morph/2013_03_06_cell11_1125_H41_06.asc")

        self.N_segs, self.N_iontypes, self.N_ions = []
        for sectype in ['soma', 'axon', 'dend', 'apic']:
            self.N_segs.append(np.sum([sec.nseg for sec in getattr(self.cell, sectype)]))
            self.N_iontypes.append(len(self.ionconfig['soma']) if sectype in self.ionconfig.keys() else 0)
            self.N_ions.append(self.N_segs[-1] * self.N_iontypes[-1])
        self.N_ion = np.sum(self.N_ions)
        self.N = self.N_out + self.N_ion

        for sec in self.cell.all:
            sec.e_pas = self.v_rest
        self.v_out = h.Vector().record(self.cell.soma[0](0.5)._ref_v)
    
    def _cal_K(self):
        self.K_len = int(self.K_max_t / h.dt)
        try:
            tmp_K = np.load(self.K_filename)
            print(f"read K from {self.K_filename}")
            assert tmp_K.shape == (self.N, self.N_ion, self.K_len), "Unexpected shape of K"
        except FileNotFoundError:
            print(f"{self.K_filename} not found, computing K")
            tmp_cell = self._setup_hpc("PassiveHPC", "2013_03_06_cell11_1125_H41_06.asc")
            for sec in tmp_cell.all:
                sec.e_pas = self.v_rest
            tmp_K = np.zeros((self.N, self.N_ion, self.K_len), dtype=np.float32)
            v_list = []
            for i in range(self.N):
                dend_id = self.conn[i]
                loc = self.conn_loc[i]
                v = h.Vector()
                if i < self.N_out:                                  # somatic output
                    v.record(tmp_cell.soma[dend_id](loc)._ref_v)
                elif dend_id < self.total_dend:                     # dend
                    v.record(tmp_cell.dend[dend_id](loc)._ref_v)
                else:                                               # apic
                    apic_id = dend_id - self.total_dend
                    v.record(tmp_cell.apic[apic_id](loc)._ref_v)
                v_list.append(v)

            for j in tqdm(range(self.N_ion)):
                dend_id = self.conn[self.N_out + j]
                loc = self.conn_loc[self.N_out + j]
                if dend_id < self.total_dend:                     # dend
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
        # transfer impedance matrix
        self._cal_K()

        # weight matrix input-to-cell dendrites
        self.w = self.rng.uniform(-1*1e-3, 1*1e-3, (self.N_ion,))

        # ion
        self.ionlist = []
        p = 0
        for sectype in ['soma', 'axon', 'dend', 'apic']:
            for iontype in self.ionconfig[sectype].keys():
                ionconfig = self.ionconfig[sectype][iontype]
                for sec in getattr(self.cell, sectype):
                    for seg in sec:
                        ion = getattr(h, iontype)(seg)
                        for (param, value) in ionconfig.items():
                            setattr(ion, param, value)
                        ion.w = self.w[p]
                        p += 1
                        self.ionlist.append(ion)
        assert p == self.N_ion

        self.stim = h.IClamp(self.cell.soma[0](0.5))

    def set_stim(self, inputs):
        amp, delay, dur = inputs
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur
        self._reset_records()
    
    def _reset_records(self):
        self.It = torch.tensor(np.array([]).reshape((self.N_ion, 0)), 
                               dtype=torch.float32, device=torch.device(self.device))       # (N_ion, min(t, K_max_t))
        self.dItdv = torch.tensor(np.array([]).reshape((self.N_ion, 0)), 
                                  dtype=torch.float32, device=torch.device(self.device))    # (N_ion, min(t, K_max_t))
        self.dVtdw = torch.tensor(np.zeros((self.N_ion, self.N, 1)), 
                                  dtype=torch.float32, device=torch.device(self.device))    # (N_ion, N, min(t, K_max_t)), [i, j] = dvjdwi
        self.dVouttdw = np.zeros((self.N_ion, 1))                                           # (N_ion, t)
        self.v_pred = [self.v_rest,]

    def update_pred(self, tstep):
        # called after each timestep advance
        it = torch.tensor([syn.pure_i for syn in self.ionlist], 
                          dtype=torch.float32, device=torch.device(self.device))
        self.It = torch.hstack((self.It, it[:, torch.newaxis]))[:, -self.K_len:]            # (N_ion, min(t, K_max_t))
        
        K_out_conv = self.K[0, :, -tstep:]
        w_device = torch.tensor(self.w[:, np.newaxis], device=self.device)
        self.v_pred.append(torch.sum(K_out_conv * self.It * torch.abs(w_device)).cpu() * h.dt + self.v_rest)

    def update_dvdw(self, tstep):
        # called after each timestep advance
        it = torch.tensor([syn.pure_i for syn in self.ionlist],
                          dtype=torch.float32, device=torch.device(self.device))
        self.It = torch.hstack((self.It, it[:, torch.newaxis]))[:, -self.K_len:]                    # (N_ion, min(t, K_max_t))
        ditdv = torch.tensor([syn.didv for syn in self.ionlist],
                             dtype=torch.float32, device=torch.device(self.device))
        self.dItdv = torch.hstack((self.dItdv, ditdv[:, torch.newaxis]))[:, -self.K_len:]           # (N_ion, min(t, K_max_t))
        
        K_conv = self.K[:, :, -tstep:]                                                              # (N, N_ion, min(t, K_max_t))
        dvtdw = torch.einsum('jit,it->ij', K_conv, self.It) * h.dt                                  # (N_ion, N)
        dItdw_conv_0tot_1 = self.dItdv[torch.newaxis, :, :] * self.dVtdw[:, self.N_out:]            # (N_ion, N_ion, min(t, K_max_t))
        dvtdw += torch.einsum('jkt,ikt->ij', K_conv, dItdw_conv_0tot_1) * h.dt

        self.dVtdw = torch.dstack((self.dVtdw, dvtdw[:, :, torch.newaxis]))[:, :, -self.K_len:]     # (N_ion, N, min(t, K_max_t))
        dvouttdw = dvtdw[:, 0].cpu().numpy()                                                        # (N_ion,)
        self.dVouttdw = np.hstack((self.dVouttdw, dvouttdw[:, np.newaxis]))                         # (N_ion, t)

    def get_dw(self, dLtdv, lr_start, lr_end):
        # called after each run
        assert(lr_end - lr_start == len(dLtdv))
        dw = -np.sum(dLtdv * self.dVouttdw[:, lr_start:lr_end], axis=1) / (lr_end - lr_start)
        return dw

    def update_weights(self, dw):
        assert dw.shape == (self.N_ion,)
        dw = np.array(dw)
        self.w += dw
        self.set_weights()
    
    def set_weights(self, w=None):
        # update weights to syns
        if w is None:
            w = self.w
        else:
            assert w.shape == (self.N_ion,)
        self.w = np.clip(np.array(w), a_min=self.w_min, a_max=self.w_max)
        for i in range(self.N_ion):
            self.ionlist[i].w = self.w[i]
    
    def save_weights(self, path):
        np.save(path, self.w)
