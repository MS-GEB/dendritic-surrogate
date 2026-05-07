import numpy as np
import torch
from neuron import h
import pickle
import json


def load_json(file_name):
    with open(file_name, 'r+') as f:
        data_dic = json.load(f)
    return data_dic

class SynInfo(object):
    def __init__(self, id, point, syn, p=-1):
        self.id = id
        self.point = point
        self.syn = syn
        self.p = p


class Network(object):
    def __init__(self, net_config, lr_config, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.load_network_info(net_config)
        self.create_cells()
        self.set_cell_segments()
        self.define_biophysics()

        # learning configs
        self.v_r = lr_config['v_r']             # resting potential
        self.ngpu = lr_config['ngpu']
        self.K_max_t = lr_config['K_max_t']     # transfer impedance maximum time length
        self.K_filename = lr_config['K_filename']
        self.K_nblock = lr_config['K_nblock']
        self.K_nblock = np.clip(self.K_nblock, a_min=1, a_max=self.ncell)
        self.K_mul = lr_config['K_mul']
        self.w_gap_max = lr_config['w_gap_max']
        self.w_gap_min = lr_config['w_gap_min']
        self.w_syn_max = lr_config['w_syn_max']
        self.w_syn_min = lr_config['w_syn_min']


    def load_network_info(self, net_config):
        cell_info = net_config["cell_info"]
        self.cells_id_sim = cell_info["cells_id_sim"]
        self.cells_name_dic = cell_info["cells_name_dic"]   # find name, key-id, value-name
        self.cells_id_dic = {v : k for k, v in self.cells_name_dic.items()} # find id, key-name, value-id
        self.length_per_seg = net_config["cnt_info"]["length_per_seg"]
        print("length_per_seg", self.length_per_seg)

        dir_info = net_config["dir_info"]
        self.model_dir = dir_info["model_dir"]
        self.cell_param_dir = dir_info["cell_param_dir"]


    def create_cells(self):
        self.ncell = len(self.cells_id_sim)
        self.cells = {}
        for id in self.cells_id_sim:
            cn = self.cells_name_dic[id]
            h.load_file(self.model_dir + cn + ".hoc")
            self.cells[cn] = getattr(h, cn)()


    def define_biophysics(self):
        self.mech_list = ['slo1_unc2_lr', 
                          'egl2_lr', 
                          'shl1_lr', 
                          'kqt3_lr',  
                          'unc2_lr', 
                          'kvs1_lr', 
                          'slo1_egl19_lr', 
                          'slo2_unc2_lr', 
                          'irk_lr', 
                          'egl36_lr', 
                          'egl19_lr', 
                          'cca1_lr', 
                          'shk1_lr', 
                          'slo2_egl19_lr', 
                          'nca_lr', 
                          'kcnl_lr',
                          'cainternm_lr',
                          ]
        for id in self.cells_id_sim:
            cn = self.cells_name_dic[id]
            cell = self.cells[cn]
            cell_param = load_json(self.cell_param_dir + cn + ".json")
            
            # set biophysical mechanism
            sec = cell.Soma
            sec.Ra = cell_param["soma"]["Ra"]  # (Ohm*cm)
            sec.cm = cell_param["soma"]["cm"]  # (uF/cm2)
            
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = cell_param["soma"]["gpas"]  # Passive conductance in S/cm2
                seg.pas.e = cell_param["soma"]["epas"]  # Leak reversal potential mV

            for sec in cell.all: # check neuron_name.hoc
                if "Soma" in sec.name():
                    continue
                sec.Ra = cell_param["neurite"]["Ra"]  # (Ohm*cm)
                sec.cm = cell_param["soma"]["cm"]  # (uF/cm2)
                
                sec.insert('pas')
                for seg in sec:
                    seg.pas.g = cell_param["neurite"]["gpas"]  # Passive conductance in S/cm2 
                    seg.pas.e = cell_param["neurite"]["epas"]  # Leak reversal potential mV

            for m in self.mech_list:
                cell.Soma.insert(m)
            for seg in cell.Soma:
                seg.shl1_lr.gbshl1 = cell_param["soma"]["gbshl1"]  # (nS/um2)
                seg.shk1_lr.gbshk1 = cell_param["soma"]["gbshk1"]
                seg.kvs1_lr.gbkvs1 = cell_param["soma"]["gbkvs1"]
                seg.egl2_lr.gbegl2 = cell_param["soma"]["gbegl2"]
                seg.egl36_lr.gbegl36 = cell_param["soma"]["gbegl36"]
                seg.kqt3_lr.gbkqt3 = cell_param["soma"]["gbkqt3"]
                seg.egl19_lr.gbegl19 = cell_param["soma"]["gbegl19"]
                seg.unc2_lr.gbunc2 = cell_param["soma"]["gbunc2"]
                seg.cca1_lr.gbcca1 = cell_param["soma"]["gbcca1"]
                seg.slo1_egl19_lr.gbslo1 = cell_param["soma"]["gbslo1_egl19"]
                seg.slo1_unc2_lr.gbslo1 = cell_param["soma"]["gbslo1_unc2"]
                seg.slo2_egl19_lr.gbslo2 = cell_param["soma"]["gbslo2_egl19"]
                seg.slo2_unc2_lr.gbslo2 = cell_param["soma"]["gbslo2_unc2"]
                seg.kcnl_lr.gbkcnl = cell_param["soma"]["gbkcnl"]
                seg.nca_lr.gbnca = cell_param["soma"]["gbnca"]
                seg.irk_lr.gbirk = cell_param["soma"]["gbirk"]


    def set_cell_segments(self):
        """section, segment separation, prepare self.segments"""
        for _, cell in self.cells.items():
            for section in cell.all:
                soma_flag = 'Soma' in section.name()
                section.nseg = 1 if soma_flag else int(np.ceil(section.L / self.length_per_seg))


    def get_3dp_segment(self, nrn_id, seg_id):
        cn = self.cells_name_dic[nrn_id]
        cell = self.cells[cn]
        seg_cnt = 0
        for section in cell.all:
            if seg_id < seg_cnt + section.nseg:
                loading_bar = np.linspace(0, 1, section.nseg + 1)
                loading_cnt = seg_id - seg_cnt
                return section((loading_bar[loading_cnt] + loading_bar[loading_cnt + 1]) / 2)
            seg_cnt += section.nseg


    def read_cells_neurite_connection(self, infile, input_names):
        import pickle
        connection_info = pickle.load(open(infile, 'rb'))
        # connection_info: a dictionary
        # key: id_post_cell
        # value: [[id_pre_cell, id_pre_point, id_post_point, type, weight],[...]]
        #        type = 0/1/2  0-gap junction, 1-excitatory synapse, 2-inhibitory synapse
        # for id in self.cells_id_sim:
        #     print(id)
        #     if id in connection_info.keys():
        #         print(connection_info[id])
        #     print('')
        
        self.relpointlist = {}      # id -> [point, ...] (relative segments, non-repetitive)
        self.synlist = {}           # id_post_cell -> id_post_point -> [SynInfo(id_pre_cell, id_pre_point, syn, p), ...]
        self.synapse_list = []
        for id in self.cells_id_sim:
            cn = self.cells_name_dic[id]
            cell = self.cells[cn]
            self.relpointlist[id] = list(range(cell.Soma.nseg))
            self.synlist[id] = {}
        for id_post_cell in self.cells_id_sim:  # for each post neuron
            # cn_post = self.cells_name_dic[id_post_cell]
            if id_post_cell not in connection_info.keys():
                continue
            connection = connection_info[id_post_cell]
            for [id_pre_cell, id_pre_point, id_post_point, syntype, w] in connection:
                if id_pre_cell in self.cells_id_sim:  # for each pre neuron
                    cn_pre = self.cells_name_dic[id_pre_cell]
                    if id_post_point not in self.synlist[id_post_cell].keys():
                        self.synlist[id_post_cell][id_post_point] = []
                    if id_post_point not in self.relpointlist[id_post_cell]:
                        self.relpointlist[id_post_cell].append(id_post_point)
                    if id_pre_point not in self.relpointlist[id_pre_cell]:
                        self.relpointlist[id_pre_cell].append(id_pre_point)
                    pre_seg = self.get_3dp_segment(id_pre_cell, id_pre_point)
                    post_seg = self.get_3dp_segment(id_post_cell, id_post_point)
                    if syntype == 0: # construct gap junction
                        gapjunction = h.gapjunction_lr(post_seg)
                        gapjunction.w = w
                        gapjunction._ref_vpre = pre_seg._ref_v
                        self.synapse_list.append(gapjunction)
                        self.synlist[id_post_cell][id_post_point].append(SynInfo(id_pre_cell, id_pre_point, gapjunction))
                    elif syntype == 1 or syntype == 2:
                        # construct synapse
                        synapse = h.neuron_to_neuron_syn_lr(post_seg)
                        synapse.w = w * 1e-4 if syntype == 1 else -w * 1e-4
                        synapse._ref_vpre = pre_seg._ref_v
                        self.synapse_list.append(synapse)
                        self.synlist[id_post_cell][id_post_point].append(SynInfo(id_pre_cell, id_pre_point, synapse))
                    else:
                        raise ValueError("Synapse type unrecognized, expect 0/1/2, get", syntype)
        
        self.N_input = len(input_names)
        self.input_ids = [self.cells_id_dic[cn] for cn in input_names]
        self.input_synlist = {}
        for id in self.input_ids:
            seg = self.get_3dp_segment(id, 0)   # Soma 0
            syn = h.IClamp(seg)
            syn.amp = 0.
            syn.delay = 0.
            syn.dur = 1e9
            self.input_synlist[id] = syn

        self.N = 0              # total number of relative segments
        self.K_start = {}       # id -> pstart in self.K
        self.K_n = {}           # id -> number of relative segments
        self.pinput = []        # list of input p in self.K
        self.point2p = {}       # id_cell -> id_point -> [p, ...] in self.K
        self.pwmaskall = []     # list of w fixed p (active channels & pure pre-synaptic) in self.K
        self.pwmask = {}        # id -> [p, ...], w fixed ps for each id
        self.allpointlist = {}  # id -> [point, ...] (can be repetitive, including multiple synapses onto single segment senario)
        for id in self.cells_id_sim:
            cn = self.cells_name_dic[id]
            self.K_start[id] = self.N
            self.relpointlist[id].sort()
            self.allpointlist[id] = []

            # w fixed points, including active channel segments, input segments & pure pre-synaptic segments
            wmask_points = list(range(self.cells[cn].Soma.nseg))    # somatic active channel segments
            if id in self.input_ids:
                self.pinput.append(self.N + len(wmask_points))
                wmask_points.append(0)                              # input onto soma seg 0
            for point in self.relpointlist[id]:
                if point not in self.synlist[id].keys() and point not in wmask_points:  # pure pre-synaptic segments
                   wmask_points.append(point)
            pwmask = list(range(self.N, self.N + len(wmask_points)))
            self.pwmaskall += pwmask
            self.pwmask[id] = pwmask
            self.allpointlist[id] += wmask_points
            
            n_id = len(wmask_points)
            # gap junction & post-synaptic segments
            for point in self.synlist[id].keys():
                n_id += len(self.synlist[id][point])
                self.allpointlist[id] += [point] * len(self.synlist[id][point])
            
            self.K_n[id] = n_id
            self.N += n_id

            self.point2p[id] = {}
            p = self.K_start[id]
            for point in wmask_points:
                if point not in self.point2p[id].keys():
                    self.point2p[id][point] = []
                self.point2p[id][point].append(p)
                p += 1
            for point in self.synlist[id].keys():
                if point not in self.point2p[id].keys():
                    self.point2p[id][point] = []
                for syninfo in self.synlist[id][point]:
                    self.point2p[id][point].append(p)
                    syninfo.p = p
                    p += 1

        self.w = torch.zeros((self.N,), dtype=torch.float32)
        self.w[self.pwmaskall] = 1.
        self.pgap = []
        self.psyn = []
        for id in self.cells_id_sim:
            for point in self.synlist[id].keys():
                for syninfo in self.synlist[id][point]:
                    pw = syninfo.p
                    syn = syninfo.syn
                    self.w[pw] = syn.w
                    if 'gapjunction' in syn.hname():
                        self.pgap.append(pw)
                    elif 'syn' in syn.hname():
                        self.psyn.append(pw)
        
        print(self.N)
        # print(self.K_n.items())

        self.postplist = []
        self.preplist = []
        for id_post in self.cells_id_sim:
            for point_post in self.synlist[id_post].keys():
                for syninfo in self.synlist[id_post][point_post]:
                    id_pre = syninfo.id
                    point_pre = syninfo.point
                    p_post = syninfo.p
                    self.postplist.append(p_post)
                    self.preplist.append(self.point2p[id_pre][point_pre][0])    # first occurance

        self._cal_K()


    def _cal_K(self):
        self.K_len = int(self.K_max_t / (h.dt * self.K_mul))

        def split_integer(m, n):
            assert n > 0
            quotient = m // n
            remainder = m % n
            if remainder > 0:
                return [quotient] * (n - remainder) + [quotient + 1] * remainder
            if remainder < 0:
                return [quotient - 1] * -remainder + [quotient] * (n + remainder)
            return  [quotient] * n
        
        id_split = split_integer(self.ncell, self.K_nblock)     # split ncell evenly into K_nblock
        self.K_block_n = []         # number of relative segments in each block
        self.K_block_n_start = []   # start of each split for block
        self.id2block = {}          # cell id -> block id
        self.block2ids = {}         # block id -> [cell id, ...]
        cell_id = 0
        block_start = 0
        for i in range(self.K_nblock):
            self.K_block_n_start.append(block_start)
            self.block2ids[i] = []
            block_n = 0
            for _ in range(id_split[i]):
                block_n += self.K_n[self.cells_id_sim[cell_id]]
                self.id2block[self.cells_id_sim[cell_id]] = i
                self.block2ids[i].append(self.cells_id_sim[cell_id])
                cell_id += 1
            self.K_block_n.append(block_n)
            block_start += block_n
        print(id_split)
        # print(self.id2block)
        # print(self.block2ids)
        print(self.K_block_n)
        print(self.K_block_n_start)

        self.K_gpu_n = split_integer(self.N, self.ngpu)   # split N evenly into ngpu
        print(self.K_gpu_n)
        gpu_start = 0
        self.K_gpu_n_start = []     # start of each split for gpu
        for i in range(self.ngpu):
            self.K_gpu_n_start.append(gpu_start)
            gpu_start += self.K_gpu_n[i]
        print(self.K_gpu_n_start)
        
        try:
            print('read K from ' + self.K_filename)
            with open(self.K_filename, 'rb') as f:
                # tmp_K = np.load(f, allow_pickle=True)['K']
                tmp_K = pickle.load(f)
                print("load K done!")
            assert len(tmp_K) == self.K_nblock, 'Unexpected number of blocks read from K'
            tmp_shape = 0
            for i in range(self.K_nblock):
                ki = tmp_K[i]
                assert ki.shape[0] == ki.shape[1] and ki.shape[2] == self.K_len, f'Unexpected shape of block {i} read from K'
                tmp_shape += ki.shape[0]
            assert tmp_shape == self.N, 'Unexpected total number of relative segments read from K'
        except FileNotFoundError:
            print(self.K_filename + ' not found, calculating K')
            
            def get_3dp_segment_tmp(cell, seg_id):
                seg_cnt = 0
                for section in cell.all:
                    if seg_id < seg_cnt + section.nseg:
                        loading_bar = np.linspace(0, 1, section.nseg + 1)
                        loading_cnt = seg_id - seg_cnt
                        return section((loading_bar[loading_cnt] + loading_bar[loading_cnt + 1]) / 2)
                    seg_cnt += section.nseg
            
            tmp_K = []
            for block_n in self.K_block_n:
                tmp_K.append(np.zeros((block_n, block_n, self.K_len), dtype=np.float16))
            for id in self.cells_id_sim:  # for each post neuron
                print('cell ' + id + '/302', end='\r')
                tmpcn = self.cells_name_dic[id]
                tmpcell = getattr(h, tmpcn)()
                cell_param = load_json(self.cell_param_dir + tmpcn + ".json")
                for section in tmpcell.all:
                    soma_flag = 'Soma' in section.name()
                    section.nseg = 1 if soma_flag else int(np.ceil(section.L / self.length_per_seg))
                
                # set biophysical mechanism
                sec = tmpcell.Soma
                sec.Ra = cell_param["soma"]["Ra"]  # (Ohm*cm)
                sec.cm = cell_param["soma"]["cm"]  # (uF/cm2)
                sec.insert('pas')
                for seg in sec:
                    seg.pas.g = cell_param["soma"]["gpas"]  # Passive conductance in S/cm2
                    seg.pas.e = cell_param["soma"]["epas"]  # Leak reversal potential mV

                for sec in tmpcell.all: # check neuron_name.hoc
                    if "Soma" in sec.name():
                        continue
                    sec.Ra = cell_param["neurite"]["Ra"]  # (Ohm*cm)
                    sec.cm = cell_param["neurite"]["cm"]  # (uF/cm2)
                    sec.insert('pas')
                    for seg in sec:
                        seg.pas.g = cell_param["neurite"]["gpas"]  # Passive conductance in S/cm2 
                        seg.pas.e = cell_param["neurite"]["epas"]  # Leak reversal potential mV
                
                tmpseglist = []
                tmpvlist = []
                for point in self.allpointlist[id]:
                    seg = get_3dp_segment_tmp(tmpcell, point)
                    tmpseglist.append(seg)
                    tmpvlist.append(h.Vector().record(seg._ref_v))
                
                block_id = self.id2block[id]
                block_ids = self.block2ids[block_id]
                offset_in_block = 0
                for id_same_block in block_ids:
                    if int(id_same_block) < int(id):
                        offset_in_block += self.K_n[id_same_block]
                old_dt = h.dt
                h.dt = old_dt * self.K_mul
                for i in range(self.K_n[id]):
                    tmpclamp = h.IClamp(tmpseglist[i])
                    tmpclamp.delay = 500.
                    tmpclamp.dur = h.dt
                    tmpclamp.amp = 1. / h.dt
                    h.finitialize(cell_param["soma"]["epas"])
                    h.continuerun(self.K_max_t + tmpclamp.delay)
                    for j in range(self.K_n[id]):
                        v = np.array(tmpvlist[j], dtype=np.float16)
                        i_start = int(tmpclamp.delay / h.dt)    # assert divisible
                        v_rev = v[i_start]
                        tmp_K[block_id][offset_in_block + i, offset_in_block + j] = v[i_start + 1: i_start + 1 + self.K_len][::-1] - v_rev  # already reversed
                    tmpclamp.amp = 0.
                h.dt = old_dt

            print('save K to ' + self.K_filename)
            with open(self.K_filename, 'wb') as f:
                pickle.dump(tmp_K, f, protocol=4)
            print('save K done! ')
        
        for i in range(self.ngpu):
            setattr(self, 'K' + str(i), [torch.asarray(ki, dtype=torch.float32, device=f'cuda:{i}') for ki in tmp_K])  # copy of self.Ki on gpu i

    
    def set_outputs(self, output_names):
        self.N_output = len(output_names)
        self.output_ids = [self.cells_id_dic[cn] for cn in output_names]
        print(output_names, self.output_ids)
        self.poutput = []
        self.output_vlist = {}
        for id in self.output_ids:
            self.poutput.append(self.point2p[id][0][0])     # first occurance
            seg = self.get_3dp_segment(id, 0)   # Soma 0
            vec = h.Vector().record(seg._ref_v)
            self.output_vlist[id] = vec
    

    def _reset_lr_records(self):
        self.It = torch.zeros((self.N, self.K_len), dtype=torch.float32, device='cuda:0')
        if self.ngpu == 1:
            self.dItdv = torch.zeros((self.N, self.K_len), dtype=torch.float32, device='cuda:0')
            self.dItdvpre = torch.zeros((self.N, self.K_len), dtype=torch.float32, device='cuda:0')
            self.dVtdw = torch.zeros((self.N, self.N, self.K_len), dtype=torch.float32, device='cuda:0')
            self.dVpretdw = torch.zeros((self.N, self.N, self.K_len), dtype=torch.float32, device='cuda:0')
        else:
            if hasattr(self, 'dItdvlist'):
                del self.dItdvlist, self.dItdvprelist, self.dVtdwlist, self.dVpretdwlist, self.tmp_dvtdwlist
            self.dItdvlist = [torch.zeros((self.N, self.K_len), dtype=torch.float32, device=f'cuda:{i}') for i in range(self.ngpu)]
            self.dItdvprelist = [torch.zeros((self.N, self.K_len), dtype=torch.float32, device=f'cuda:{i}') for i in range(self.ngpu)]
            self.dVtdwlist = [torch.zeros((self.K_gpu_n[i], self.N, self.K_len), dtype=torch.float32, device=f'cuda:{i}') for i in range(self.ngpu)]
            self.dVpretdwlist = [torch.zeros((self.K_gpu_n[i], self.N, self.K_len), dtype=torch.float32, device=f'cuda:{i}') for i in range(self.ngpu)]
            self.tmp_dvtdwlist = [torch.zeros((self.K_gpu_n[i], self.N), dtype=torch.float32, device=f'cuda:{i}') for i in range(self.ngpu)]

        self.t_ptr = 0
        self.dVoutputtdw = torch.zeros((self.N, self.N_output, 1), dtype=torch.float32) # shape (N, N_output, tstep)
        self.dVinputtdw = torch.zeros((self.N, self.N_input, 1), dtype=torch.float32)   # shape (N, N_input, tstep)

        self.grad_l2norm_thresold = 1.e6
        self.grad_scale = 1.

        self.streams = [torch.cuda.Stream(f'cuda:{i}') for i in range(self.ngpu)]
        self.indices = torch.arange(self.K_len)

        for i in range(self.ngpu):
            torch.cuda.synchronize(f'cuda:{i}')
        torch.cuda.empty_cache()


    def update_dvdw(self, tstep, percise=True):
        # called after timestep advance
        itlist = []
        for id in self.cells_id_sim:
            cn = self.cells_name_dic[id]
            it = [0.] * len(self.pwmask[id])
            for id_seg, seg in enumerate(self.cells[cn].Soma):
                seg_area = seg.area()
                for ch_name in self.mech_list:
                    it[id_seg] += getattr(seg, ch_name).pure_i * seg_area * 1e-2
            if id in self.input_ids:
                syn = self.input_synlist[id]
                if 'syn' in syn.hname():
                    it[self.cells[cn].Soma.nseg] += syn.pure_i
                else:
                    it[self.cells[cn].Soma.nseg] += syn.i
            for point in self.synlist[id].keys():
                for syninfo in self.synlist[id][point]:
                    it.append(syninfo.syn.pure_i)
            itlist.extend(it)
        it = torch.tensor(itlist, dtype=torch.float32)
        self.It[:, self.t_ptr] = it.to('cuda:0')

        if percise:
            ditdvlist = []
            for id in self.cells_id_sim:
                cn = self.cells_name_dic[id]
                ditdv = [0.] * len(self.pwmask[id])
                for id_seg, seg in enumerate(self.cells[cn].Soma):
                    seg_area = seg.area()
                    for ch_name in self.mech_list:
                        ditdv[id_seg] += getattr(seg, ch_name).didv * seg_area * 1e-2
                if id in self.input_ids:
                    syn = self.input_synlist[id]
                    if 'syn' in syn.hname():
                        ditdv[self.cells[cn].Soma.nseg] += syn.didv
                for point in self.synlist[id].keys():
                    for syninfo in self.synlist[id][point]:
                        ditdv.append(syninfo.syn.didv)
                ditdvlist.extend(ditdv)
            ditdv = torch.tensor(ditdvlist, dtype=torch.float32)
            if self.ngpu == 1:
                self.dItdv[:, self.t_ptr] = ditdv.to('cuda:0')
            else:
                for i in range(self.ngpu):
                    self.dItdvlist[i][:, self.t_ptr] = ditdv.to(f'cuda:{i}')

            ditdvprelist = []
            for id in self.cells_id_sim:
                ditdvpre = [0.] * len(self.pwmask[id])
                for point in self.synlist[id].keys():
                    for syninfo in self.synlist[id][point]:
                        ditdvpre.append(syninfo.syn.didvpre)
                ditdvprelist.extend(ditdvpre)
            ditdvpre = torch.tensor(ditdvprelist, dtype=torch.float32)
            if self.ngpu == 1:
                self.dItdvpre[:, self.t_ptr] = ditdvpre.to('cuda:0')
            else:
                for i in range(self.ngpu):
                    self.dItdvprelist[i][:, self.t_ptr] = ditdvpre.to(f'cuda:{i}')
        
        view_idx = torch.flip(self.t_ptr - self.indices, [0,])

        It_split = torch.vsplit(self.It, self.K_block_n_start[1:])
        dvtdw = torch.zeros((self.N, self.N), dtype=torch.float32, device='cuda:0')
        for i, (ki, Iti) in enumerate(zip(self.K0, It_split)):
            start = self.K_block_n_start[i]
            end = start + self.K_block_n[i]
            dvtdw[start: end, start: end] = torch.einsum('ijt,it->ij', ki, Iti[:, view_idx]) * h.dt * self.grad_scale

        if percise:
            if self.ngpu == 1:
                tmp_dItdw_conv_0tot_1 = self.dItdv[torch.newaxis, :, :] * self.dVtdw + self.dItdvpre[torch.newaxis, :, :] * self.dVpretdw
                tmp_dItdw_conv_0tot_1 = torch.hsplit(tmp_dItdw_conv_0tot_1, self.K_block_n_start[1:])
                for j, (ai, ki) in enumerate(zip(tmp_dItdw_conv_0tot_1, self.K0)):
                    start = self.K_block_n_start[j]
                    end = start + self.K_block_n[j]
                    dvtdw[:, start: end] += torch.einsum('ikt,jkt->ij', ai[:, :, view_idx], ki) * h.dt
                del tmp_dItdw_conv_0tot_1
            else:
                for i in range(self.ngpu):
                    with torch.cuda.stream(self.streams[i]):
                        tmp_dItdw_conv_0tot_1 = self.dItdvlist[i][torch.newaxis, :, :] * self.dVtdwlist[i] + self.dItdvprelist[i][torch.newaxis, :, :] * self.dVpretdwlist[i]
                        tmp_dItdw_conv_0tot_1 = torch.hsplit(tmp_dItdw_conv_0tot_1, self.K_block_n_start[1:])
                        for j, (ai, ki) in enumerate(zip(tmp_dItdw_conv_0tot_1, getattr(self, f'K{i}'))):
                            start = self.K_block_n_start[j]
                            end = start + self.K_block_n[j]
                            self.tmp_dvtdwlist[i][:, start: end] = torch.einsum('ikt,jkt->ij', ai[:, :, view_idx], ki) * h.dt
                        del tmp_dItdw_conv_0tot_1
                for i in range(self.ngpu):
                    torch.cuda.synchronize(f'cuda:{i}')
                for i in range(self.ngpu):
                    start = self.K_gpu_n_start[i]
                    end = start + self.K_gpu_n[i]
                    dvtdw[start: end, :] += self.tmp_dvtdwlist[i].to('cuda:0')

        if torch.any(torch.isnan(dvtdw)):
            print('\n')
            print(f"nan detected, t: {h.t}")
            print(f"it max: {torch.max(it):.5g}, min: {torch.min(it):.5g}")
            if percise:
                print(f"ditdv max: {torch.max(ditdv):.5g}, min: {torch.min(ditdv):.5g}")
                print(f"ditdvpre max: {torch.max(ditdvpre):.5g}, min: {torch.min(ditdvpre):.5g}")
            print(f"dvtdw max: {torch.max(dvtdw):.5g}, min: {torch.min(dvtdw):.5g}")
            assert 0

        dvtdw_l2norm = float(torch.linalg.norm(dvtdw, ord='fro'))
        if dvtdw_l2norm > self.grad_l2norm_thresold:
            scaler = self.grad_l2norm_thresold / dvtdw_l2norm
            dvtdw *= scaler
            self.dVoutputtdw *= scaler
            self.grad_scale *= scaler
            if self.ngpu == 1:
                self.dVtdw *= scaler
                self.dVpretdw *= scaler
            else:
                for i in range(self.ngpu):
                    self.dVtdwlist[i] *= scaler
                    self.dVpretdwlist[i] *= scaler

        dvpretdw = torch.zeros_like(dvtdw, dtype=torch.float32, device='cuda:0')
        dvpretdw[:, self.postplist] = dvtdw[:, self.preplist]
        
        if percise:
            if self.ngpu == 1:
                self.dVtdw[:, :, self.t_ptr] = dvtdw
                self.dVpretdw[:, :, self.t_ptr] = dvpretdw
            else:
                dvtdw_split = torch.vsplit(dvtdw, self.K_gpu_n_start[1:])
                dvpretdw_split = torch.vsplit(dvpretdw, self.K_gpu_n_start[1:])
                for i in range(self.ngpu):
                    self.dVtdwlist[i][:, :, self.t_ptr] = dvtdw_split[i].to(f'cuda:{i}')
                    self.dVpretdwlist[i][:, :, self.t_ptr] = dvpretdw_split[i].to(f'cuda:{i}')

        self.dVoutputtdw = torch.dstack([self.dVoutputtdw, dvtdw[:, self.poutput, torch.newaxis].to('cpu')])
        self.dVinputtdw = torch.dstack([self.dVinputtdw, dvtdw[:, self.pinput, torch.newaxis].to('cpu')])

        self.t_ptr = (self.t_ptr + 1) % self.K_len
        

    def get_dw_dx(self, dLtdv, lr_start, lr_end):
        # called after each run
        assert dLtdv.shape == (self.N_output, lr_end - lr_start)
        dLtdv = torch.asarray(dLtdv[:, ::self.K_mul], dtype=torch.float32)
        min_t_len = min(dLtdv.shape[-1], self.dVoutputtdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul].shape[-1])
        dw = torch.sum(dLtdv[np.newaxis, :, :min_t_len] * self.dVoutputtdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul][:, :, :min_t_len], dim=(1, 2)) / min_t_len
        dVoutputtdVinputt = torch.einsum("ikt,ijt->jkt", self.dVoutputtdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul][:, :, :min_t_len], 1. / (self.dVinputtdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul][:, :, :min_t_len] + 1e-6))    # shape (N_input, N_output, lr_end // K_mul - lr_start // K_mul)
        
        if torch.any(torch.isnan(dVoutputtdVinputt)):
            print('NaN in dVoutputtdVinputt')
        dVoutputtdVinputt[torch.isnan(dVoutputtdVinputt)] = 0.
        if torch.any(torch.isinf(dVoutputtdVinputt)):
            print('INF in dVoutputtdVinputt')
        dVoutputtdVinputt[torch.isinf(dVoutputtdVinputt)] = 0.

        if torch.any(torch.isnan(self.dVinputtdw)):
            print('NaN in dVinputtdw')
        self.dVinputtdw[torch.isnan(self.dVinputtdw)] = 0.
        if torch.any(torch.isinf(self.dVinputtdw)):
            print('INF in dVinputtdw')
        self.dVinputtdw[torch.isinf(self.dVinputtdw)] = 0.

        dx = torch.einsum("ot,iot,nit->it", dLtdv[:, :min_t_len], dVoutputtdVinputt, self.dVinputtdw[:, :, lr_start // self.K_mul: lr_end // self.K_mul][:, :, :min_t_len])
        if torch.any(torch.isnan(dx)):
            print('NaN in dx')
        dx[torch.isnan(dx)] = 0.

        # mask
        dw[self.pwmaskall] = 0.

        # return dw, dx
        return dw.numpy(), dx.numpy()


    def update_weights(self, dw):
        assert dw.shape == (self.N,)
        dw = torch.asarray(dw, dtype=torch.float32)

        # mask
        dw[self.pwmaskall] = 0.

        self.w += dw

        # clip w
        w_gap = self.w[self.pgap]
        w_syn = self.w[self.psyn]
        w_gap = torch.clip(w_gap, min=self.w_gap_min, max=self.w_gap_max)
        w_syn = torch.clip(w_syn, min=self.w_syn_min, max=self.w_syn_max)
        self.w[self.pgap] = w_gap
        self.w[self.psyn] = w_syn

        self.set_weights()
    

    def set_weights(self, w=None):
        # update weights to gap & syns
        if w is None:
            w = self.w
        else:
            assert w.shape == (self.N,)
        w = torch.asarray(w, dtype=torch.float32)
        w_gap = w[self.pgap]
        w_syn = w[self.psyn]
        w_gap = torch.clip(w_gap, min=self.w_gap_min, max=self.w_gap_max)
        w_syn = torch.clip(w_syn, min=self.w_syn_min, max=self.w_syn_max)
        self.w[self.pgap] = w_gap
        self.w[self.psyn] = w_syn
        for id in self.cells_id_sim:
            for point in self.synlist[id].keys():
                for syninfo in self.synlist[id][point]:
                    syn = syninfo.syn
                    pw = syninfo.p
                    syn.w = self.w[pw]
    

    def set_gap_weights(self, w):
        # update weights to gaps
        assert w.shape == (self.N,)
        w = torch.asarray(w, dtype=torch.float32)
        w_gap = w[self.pgap]
        w_gap = torch.clip(w_gap, min=self.w_gap_min, max=self.w_gap_max)
        self.w[self.pgap] = w_gap
        for id in self.cells_id_sim:
            for point in self.synlist[id].keys():
                for syninfo in self.synlist[id][point]:
                    syn = syninfo.syn
                    if 'gapjunction' in syn.hname():
                        pw = syninfo.p
                        syn.w = self.w[pw]
    

    def set_syn_weights(self, w=None):
        # update weights to syns
        assert w.shape == (self.N,)
        w = torch.asarray(w, dtype=torch.float32)
        w_syn = w[self.psyn]
        w_syn = torch.clip(w_syn, min=self.w_syn_min, max=self.w_syn_max)
        self.w[self.psyn] = w_syn
        for id in self.cells_id_sim:
            for point in self.synlist[id].keys():
                for syninfo in self.synlist[id][point]:
                    syn = syninfo.syn
                    if 'syn' in syn.hname():
                        pw = syninfo.p
                        syn.w = self.w[pw]
    

    def save_weights(self, path):
        np.save(path, self.w.numpy())
            
