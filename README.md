# Dendritic Surrogate
Dendritic voltage surrogate for training synaptic weights of biophysically detailed multi-compartment models.

Code associated with the paper "Gan He, Mengdi Zhao, Tiejun Huang and Kai Du, (2026). A Dendritic Voltage Surrogate-Based Synaptic Learning Framework for Biophysically Detailed Neurons and Networks" for reproducing Figures 3-5, including voltage reconstruction and training for single neuron models and correlation matrix training for the [BAAIWorm](https://github.com/Jessie940611/BAAIWorm) *C. elegans* network.

## Requirements
Python==3.12 (tested version)\
numpy\
matplotlib\
tqdm\
torch==2.8.0 (tested version, also requires compatible GPU)\
NEURON==8.2.7 (tested version)

## Usage
First compile mod files with 
```
cd 1_single_cell && nrnivmodl mod
cd ../2_c_elegans_network && nrnivmodl components/mechanism/modfile
```
Usage for the perisomatic model script in 1_single_cell:
```
usage: run_hh.py [-h] [--mode MODE] [--type TYPE] [--device DEVICE] [--percise] [--k_mul K_MUL] [--adam] [--seed SEED] [--init_seed INIT_SEED]

options:
  -h, --help            show this help message and exit
  --mode MODE           'num' for numerical validation, 'train' for training
  --type TYPE           output voltage type, 'pas' for subthreshold, 
                        'single' for single burst, 'multi' for multiple bursts
  --device DEVICE       PyTorch device
  --percise             whether to additionally use global gradients
  --k_mul K_MUL         temporal downsampling rate for gradient steps
  --adam                whether to use Adam optimizer
  --seed SEED           global random seed
  --init_seed INIT_SEED seed to reinitialize weights
```
Usage for the all-active model script in 1_single_cell:
```
usage: run_L5PC.py [-h] [--mode MODE] [--device DEVICE] [--k_mul K_MUL] [--adam] [--seed SEED] [--init_seed INIT_SEED]

options:
  -h, --help            show this help message and exit
  --mode MODE           'num' for numerical validation, 'train' for training
  --device DEVICE       PyTorch device
  --k_mul K_MUL         temporal downsampling rate for gradient steps
  --adam                whether to use Adam optimizer
  --seed SEED           global random seed
  --init_seed INIT_SEED seed to reinitialize weights
```
Usage for the *C. elegans* network script in 2_c_elegans_network:
```
usage: run_eworm.py [-h] [--mode MODE] [--ngpu NGPU] [--percise] [--k_mul K_MUL] [--adam]

options:
  -h, --help     show this help message and exit
  --mode MODE    'train' or 'test'
  --ngpu NGPU    number of gpus to use
  --percise      whether to additionally use global gradients
  --k_mul K_MUL  temporal downsampling rate for gradient steps
  --adam         whether to use Adam optimizer
```
### 1. Validate numerical accuracy for single neurons (Figure 3)
In the 1_single_cell directory:
#### Perisomatic model
- Subthreshold (Figure 3B)
```
python3 run_hh.py --mode num --type pas --device 'cuda:0'
```
- Bursting (Figure 3C)
```
python3 run_hh.py --mode num --type multi --device 'cuda:0'
```
#### All-active model 
- Dendritic calcium plateau & somatic bursting (Figures 3D-E)
```
python3 run_L5PC.py --mode num --device 'cuda:0'
```

### 2. Train synaptic weights of single neurons (Figure 4)
In the 1_single_cell directory:
#### Perisomatic model with full learning rule
- Subthreshold (Figure 4A)
```
python3 run_hh.py --mode train --type pas --percise --k_mul 5 --adam --device 'cuda:0'
```
- Single burst (Figure 4B)
```
python3 run_hh.py --mode train --type single --percise --k_mul 5 --adam --device 'cuda:0'
```
- Multiple bursts (Figure 4C)
```
python3 run_hh.py --mode train --type multi --percise --k_mul 5 --device 'cuda:0'
```
#### All-active model with local learning rule
- Dendritic calcium plateau & somatic bursting (Figures 4D-E)
```
python3 run_L5PC.py --mode train --k_mul 10 --device 'cuda:0'
```

Note that full-gradient training is memory-consuming with k_mul=1

### 3. Train BAAIWorm *C. elegans* network (Figure 5)
In the 2_c_elegans_network directory:
```
python3 run_eworm.py --mode train --ngpu 1 --percise --k_mul 5 --adam
```
The original implementation for BAAIWorm training is at [BAAIWorm](https://github.com/Jessie940611/BAAIWorm/tree/main/eworm_learn). The optimized training in this repo can run on a single GPU (> 6GB memory) with ~5 min/iteration. 
## License
This project is covered under the Apache License 2.0.

## Contact
For any questions please contact Gan He via email (hegan@pku.edu.cn).