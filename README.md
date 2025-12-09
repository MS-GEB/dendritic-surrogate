# Dendritic Surrogate
Dendritic voltage surrogate for training synaptic weights of biophysically detailed multi-compartment models.

Code associated with the paper "Gan He, Mengdi Zhao, Tiejun Huang and Kai Du, (2025). A Dendritic Voltage Surrogate-Based Synaptic Learning Framework for Biophysically Detailed Neurons and Networks" for reproducing Figures 3 & 4.

## Requirements
Python==3.12 (tested version)\
numpy\
matplotlib\
tqdm\
torch==2.8.0 (tested version, also best to have compatible GPU)\
NEURON==8.2.7 (tested version)

## Usage
First compile mod files with 
```
nrnivmodl ./mod
```

### 1. Validate numerical accuracy
#### Perisomatic model
- Subthreshold (Figure 3B)
```
python3 run_num.py --mode pas
```
- Bursting (Figure 3C)
```
python3 run_num.py --mode multi
```
#### All-active model 
- Dendritic calcium plateau & somatic bursting (Figures 3D-E)
```
python3 run_num_L5PC.py
```

### 2. Train synaptic weights
#### Perisomatic model with full learning rule
- Subthreshold (Figure 4A)
```
python3 run_syn.py --mode pas --adam
```
- Single burst (Figure 4B)
```
python3 run_syn.py --mode single --adam
```
- Multiple bursts (Figure 4C)
```
python3 run_syn.py --mode multi
```
#### All-active model with local learning rule
- Dendritic calcium plateau & somatic bursting (Figures 4D-E)
```
python3 run_syn_L5PC.py
```

Note that full-gradient training is memory-consuming with the default 'K_max_t' in run.py. Try reducing this value when out of memory.

## License
This project is covered under the Apache License 2.0.

## Contact
For any questions please contact Gan He via email (hegan@pku.edu.cn).