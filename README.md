# Dendritic Surrogate
Dendritic voltage surrogate for training synaptic weights of biophysically detailed multi-compartment models.

Code associated with the paper "Gan He, Mengdi Zhao, Tiejun Huang and Kai Du, (2025). Training biophysically detailed neurons and networks with a dendritic voltage surrogate" (soon on biorxiv when ready) for reproducing Figure 2.

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
#### Subthreshold (Figure 2b)
```
python3 run_num.py --mode pas
```
#### Bursting (Figure 2c)
```
python3 run_num.py --mode multi
```

### 2. Train synaptic weights
#### Subthreshold (Figure 2d)
```
python3 run_syn.py --mode pas --adam
```
#### Single burst (Figure 2e)
```
python3 run_syn.py --mode single --adam
```
#### Multiple bursts (Figure 2f)
```
python3 run_syn.py --mode multi
```

Note that training is memory-intensive with the default 'K_max_t' in run.py. Try reducing this value when out of memory.

## License
This project is covered under the Apache License 2.0.

## Contact
For any questions please contact Gan He via email (hegan@pku.edu.cn).