# Dendritic Surrogate

Code associated with the paper "Gan He, Mengdi Zhao, Tiejun Huang and Kai Du, (2025). Training biophysically detailed neurons and networks with a dendritic voltage surrogate" (soon on biorxiv when ready) for reproducing Figure 2.

Requirements
-----------------
Python==3.12 (tested version)\
numpy\
matplotlib\
tqdm\
torch==2.8.0 (tested version, also best to have compatible GPU)\
NEURON==8.2.7 (tested version)

Usage
---------------
First compile mod files with 
```
nrnivmodl ./mod
```
Note to add '--device' as your PyTorch computing device in the following commands (default is 'cuda:0')

### Numerical accuracy validation for subthreshold (Figure 2b)
```
python3 run.py --setup acc --mode pas
```
### Numerical accuracy validation for bursting (Figure 2c)
```
python3 run.py --setup acc --mode multi
```
### Somatic curve fitting for subthreshold (Figure 2d)
```
python3 run.py --setup cf --mode pas --adam
```
### Somatic curve fitting for single burst (Figure 2e)
```
python3 run.py --setup cf --mode single --adam
```
### Somatic curve fitting for multiple bursts (Figure 2f)
```
python3 run.py --setup cf --mode multi
```

Note that training is memory-intensive with the default 'K_max_t' in run.py. Try reducing this value when out of memory.

License
-------
This project is covered under the Apache License 2.0.

Contact
-------
For any questions please contact Gan He via email (hegan@mail.tsinghua.edu.cn).