import numpy as np
from LoadFunc_train import *
from CA import *
from generate_train_data import *
from Loadmodel_T_train import *

nx = ny = 128 # domain size
nz = 1
cell_len = 1e-6 #mesh size
rec_step = 300 # total time step in the training data
delta_t = 8e-6 # time increment for CA , unit is s
ea_type = '2D' # 2D or quasi-3D
sample_n = 40 # sample number for EA setting
T_type = 'noniso' # type of temperature, iso is isothermal, noniso is nonisothermal
T_len = 1 # if >1, rotated temperature is included
T_min = 15 # minimum undercooling for the temperatrure gradient
T_max = 25 # maximum undercooling for the temperatrure gradient
T_iso = 20 # if isothermal case, the undercooling


nca_train_data, nca_train_time = gen_data(nx, ny, nz, cell_len, rec_step, delta_t, sample_n,
                                          T_type, T_len, T_min, T_max, T_iso, ea_type)

np.save('./train_data_multi_1525k.npy',nca_train_data)

# if load data from file
'''
nca_train_time = np.array(range(rec_step))
nca_train_data = np.load('./train_data_multi_'+str(rec_step)+'step.npy',allow_pickle=True)
'''


