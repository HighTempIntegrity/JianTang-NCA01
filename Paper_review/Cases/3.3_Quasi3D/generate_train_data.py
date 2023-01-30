from LoadFunc_train import *
from CA import *

def gen_data(nx, ny, nz, cell_len, rec_step, delta_t, sample_n, T_type, T_len=1, T_min=15, T_max=25, T_iso=20,ea_type='2D'):
    if T_type == 'noniso':
        for i in range(T_len):
            T_tmp = np.repeat(np.linspace(T_min, T_max, nx)[..., None, None], ny, 1)
            if i == 0:
                T_list = T_tmp[None, ...]
            else:
                rot_gap = 360//T_len
                rot = T.RandomRotation(degrees=(rot_gap * (i - 1), rot_gap * i))
                T_tmp = torch.tensor(T_tmp).type(torch.float32).permute(2,0,1)
                T_tmp = rot(T_tmp).permute(1,2,0).detach().numpy()
                T_list = np.concatenate([T_list,T_tmp[None,...]],axis=0)
    elif T_type == 'iso':
        T_list = np.array(T_iso)[None, ...]

    for Ti in range(len(T_list)):
        # only change the first Euler angle
        if ea_type == '2D':
            ealist = np.array(list(itertools.product(np.array(range(18))*5.0, repeat=1)))
            ealist = np.concatenate((ealist,ealist*0.0,ealist*0.0),axis=-1)
        elif ea_type == '3D':
            ealist = np.array(list(itertools.product(np.array(range(9)) * 10.0, repeat=3)))
        '''
        # single-grain
        ea_loc_x = nx // 2
        ea_loc_y = ny // 2
        T_ran = T_list[Ti]
        for ea_i in range(len(ealist)):
            nca_train_time_tmp, nca_train_data_tmp = CA_routine(nx, ny, nz,
             cell_len, ealist, ea_i, ea_loc_x, ea_loc_y, delta_t, rec_step, T_ran)
            if len(nca_train_data_tmp) != rec_step:
                for tt in range(rec_step - len(nca_train_data_tmp)):
                    nca_train_data_tmp = np.concatenate((nca_train_data_tmp, nca_train_data_tmp[-1, ...][None, ...]),
                                                        axis=0)

            if (ea_i == 0)& (Ti == 0):
                #print(nca_train_data_tmp.shape)
                nca_train_data = np.array(nca_train_data_tmp)[None, ...]
            else:
                #print(nca_train_data_tmp.shape)
                nca_train_data = np.concatenate((nca_train_data, nca_train_data_tmp[None, ...]), axis=0)
        '''
        # multi-grain
        for ea_i in range(-1,-sample_n-1,-1):
          ini_nuc_num = np.random.randint(6,10)
          ea_loc_x = np.random.randint(1,nx-1,size=ini_nuc_num)
          ea_loc_y = np.random.randint(1,ny-1,size=ini_nuc_num)
          ea_nuc = np.random.randint(len(ealist),size=ini_nuc_num)
          ealist_mul = ealist[ea_nuc]
          T_ran =  T_list[np.random.randint(0, len(T_list))]
          nca_train_time_tmp, nca_train_data_tmp = CA_routine(nx, ny, nz,
             cell_len, ealist_mul, ea_i, ea_loc_x, ea_loc_y, delta_t, rec_step, T_ran)
          #print(nca_train_data.shape,nca_train_data_tmp.shape)
          if len(nca_train_data_tmp)!= rec_step:
            for ttt in range(rec_step-len(nca_train_data_tmp)):
              nca_train_data_tmp = np.concatenate((nca_train_data_tmp,nca_train_data_tmp[-1, ...][None,...]),axis=0)
          if (ea_i == -1) & (Ti == 0):
            nca_train_data = np.array(nca_train_data_tmp)[None, ...]
          else:
            nca_train_data = np.concatenate((nca_train_data,nca_train_data_tmp[None, ...]),axis=0)

    nca_train_time = np.array(range(rec_step))
    return nca_train_data, nca_train_time

