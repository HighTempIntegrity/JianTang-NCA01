import numpy as np
import torch.distributed

from LoadFunc_train import *
from CA import *
from generate_train_data import *
from Loadmodel_T_train import *


from torch.nn.parallel import DistributedDataParallel as DDP
def train(rank, world_size, nca_train_time, nca_train_data, parameterization, fold_path):
    # setup the process groups
    setup(rank,world_size)

    loss_step = []  # store the training loss
    train_loss_step = []
    valid_loss_step = []

    rand_seed = parameterization.get("rand_seed", -1)
    if rand_seed != -1:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
    ca = NCA(parameterization)
    ca.initialize_weights()
    retrain = parameterization.get("retrain", False)
    if retrain:
        ca = load_model()#torch.load('./model.pkl')

    epoch_num = parameterization.get("epoch", 4000)

    optimizer = torch.optim.Adam(ca.parameters(), lr=parameterization.get("lr", 0.001),
                                 weight_decay=parameterization.get("l2_reg", 0.0))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameterization.get("step_size", 3000)),
        gamma=parameterization.get("gamma", 1.0),  # default is no learning rate decay
    )
    CHANNEL_N = int(parameterization.get("in_dim", 16))
    time_fac = int(parameterization.get("time_fac", 1.0))
    sp_rate = int(parameterization.get("speedup_rate", 5.0))
    regularization_param = parameterization.get("reg_para", 0.0)
    regularization_exp = parameterization.get("reg_exp", 1.0)
    path = fold_path + '/model.pkl'
    num_t = int(parameterization.get("tot_t", 8))
    echo_step = int(parameterization.get("echo_step", 100))


    ini_t = [0, 27]
    # size for validation set
    val_size = int(nca_train_data.shape[0] // 10)
    x_train, y_train, x_val, y_val = red_t(ini_t, num_t, sp_rate, CHANNEL_N, val_size, nca_train_data)
    '''
    x_train2, y_train2, x_val2, y_val2 = add_sample(num_t, sp_rate, CHANNEL_N, val_size)
    x_train = torch.concat([x_train, x_train2],dim=0)
    y_train = torch.concat([y_train, y_train2], dim=0)
    x_val = torch.concat([x_val, x_val2], dim=0)
    y_val = torch.concat([y_val, y_val2], dim=0)
    '''

    batch_size = int(parameterization.get("batch_size", 100))


    training_set = prepare(x_train, y_train, rank,world_size,batch_size)
    validation_set = prepare(x_val, y_val, rank, world_size, batch_size)

    # move model to rank
    #process_group = torch.distributed.new_group()
    #ca = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ca, process_group) # normal norm layer cannot work with DDP
    ca = ca.to(rank)
    ca = DDP(ca, device_ids=[rank],output_device=rank)#,find_unused_parameters=True

    for epoch in range(epoch_num):
        training_set.sampler.set_epoch(epoch)
        acc_train = []
        for j, (x_batch, xt_batch) in enumerate(training_set):
            l_valid = 0.0
            l_time_sum = 0.0

            for nca_step in range(num_t):
                for time_i in range(time_fac):
                    try:
                        x_batch = ca(x_batch)
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise exception

                l_time_sum += weigt_loss(x_batch, xt_batch[:, nca_step])

            # l_time_sum += regularization_param * regularization(ca,regularization_exp)
            with torch.no_grad():
                l_time_sum.backward()
                for p in ca.parameters():
                    p.grad /= (p.grad.norm() + 1e-8)
                optimizer.step()
                optimizer.zero_grad()
                if ((epoch % echo_step == 0) or (epoch == (epoch_num-1))) & (rank == 0):
                    ca.eval()
                    acc_train.append(cal_acc(x_batch, xt_batch[:, nca_step]))
                    ca.train()
                    torch.cuda.empty_cache()
        with torch.no_grad():
            scheduler.step()

            if ((epoch % echo_step == 0) & (rank==0)):
                torch.save(ca.module.state_dict(), path)
                ca.eval()
                acc_val=[]
                for j, (x_valid, target_valid) in enumerate(validation_set):
                    for nca_step in range(num_t):
                        for time_i in range(time_fac):
                            try:
                                x_valid = ca(x_valid)  # _input)
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                else:
                                    raise exception
                        l_valid += weigt_loss(x_valid, target_valid[:,nca_step]).item()


                    # l_valid += regularization_param * ca.regularization().item()
                    acc_val.append(cal_acc(x_valid, target_valid[:,nca_step]))
                acc_train = np.mean(np.array(acc_train))
                acc_val = np.mean(np.array(acc_val))
                print(epoch, "training loss: ", l_time_sum.item(), "valid loss: ", l_valid,
                      "training acc: ", acc_train, "%   valid accuracy: ", acc_val, "%")
                loss_step.append(
                    str(epoch) + " training loss: " + str(l_time_sum.item()) + "  valid loss: " + str(l_valid) +
                    "  training acc: " + str(acc_train) + " %   valid accuracy: " + str(acc_val) + " %")
                train_loss_step.append(l_time_sum.item())
                valid_loss_step.append(l_valid)
                ca.train()
                torch.cuda.empty_cache()

    with torch.no_grad():
        torch.distributed.barrier()
        if (rank==0):
            #
            torch.save(ca.module.state_dict(), path)
            ca.eval()
            acc_val = []
            for j, (x_valid, target_valid) in enumerate(validation_set):
                for nca_step in range(num_t):
                    for time_i in range(time_fac):
                        try:
                            x_valid = ca(x_valid)  # _input)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception
                    l_valid += weigt_loss(x_valid, target_valid[:, nca_step]).item()

                # l_valid += regularization_param * ca.regularization().item()
                acc_val.append(cal_acc(x_valid, target_valid[:, nca_step]))
            acc_train = np.mean(np.array(acc_train))
            acc_val = np.mean(np.array(acc_val))
            print(epoch, "training loss: ", l_time_sum.item(), "valid loss: ", l_valid,
                  "training acc: ", acc_train, "%   valid accuracy: ", acc_val, "%")
            loss_step.append(
                str(epoch) + " training loss: " + str(l_time_sum.item()) + "  valid loss: " + str(l_valid) +
                "  training acc: " + str(acc_train) + " %   valid accuracy: " + str(acc_val) + " %")


            # plot 3 validation samples
            for ea_i in range(np.min([val_size, 3])):
                if ea_i == 0:
                    show_img = (x_valid.permute(0, 2, 3, 1))[ea_i, ...,
                               :].cpu().detach().numpy()
                else:
                    show_img = np.vstack((show_img, (x_valid.permute(0, 2, 3, 1))[ea_i, ...,
                                                    :].cpu().detach().numpy()))

            plt.subplot(1, 2, 1)
            plt.imshow(zoom(show_img[..., :3]))
            plt.subplot(1, 2, 2)
            plt.plot(np.array(range(len(train_loss_step)))*echo_step, train_loss_step, 'b.')
            plt.plot(np.array(range(len(valid_loss_step)))*echo_step, valid_loss_step, 'r.')
            plt.title('fold_path')
            # plt.yscale('log')
            plt.savefig(path + '.jpg')
            with open(fold_path +"/loss_history.txt", "w") as outfile:
                outfile.write("\n".join(loss_step))
            print("######### training end ##########")
    clean_up()
        # torch.distributed.barrier()

def clean_up():
    dist.destroy_process_group()


def cal_acc(x_in,y_in):
    acc = []
    x = (x_in.permute(0, 2, 3, 1)).detach().cpu().numpy()
    nx = x.shape[1]
    ny = x.shape[2]

    x = x * (x[..., 4:5] > 1e-10)

    x_true = (y_in.permute(0, 2, 3, 1)).detach().cpu().numpy()
    # calculate the difference between NCA and CA
    for i in range(len(x)):
        #for j in range(0, nz, nz//5):
            mis_ori = cal_misori(np.clip(x[i, ..., :3],0.0,1.0) * 0.5 * np.pi,
                                 x_true[i, ..., :3] * 0.5 * np.pi)  # misorientation angle
            filter = mis_ori > 15.0  # diff>10.0
            # store the rsme and accuracy in the middle part of the domain
            acc.append(1.0 - np.sum(filter)/(nx * ny))
    acc = np.array(acc)*100
    acc_ave = np.average(acc)
    return acc_ave

#############################################################################
# function for calculate misorientation and Euler angle distribution

#calculate the misorientation angle between two given groups of Euler angles
def cal_misori(pred,true):
      p1 = pred[:,:,0]
      p = pred[:,:,1]
      p2 = pred[:,:,2]
      q1 = true[:,:,0]
      q = true[:,:,1]
      q2 = true[:,:,2]

      nx=p.shape[0]
      ny=p.shape[1]

      t1=np.zeros((nx,ny,24))
      t2=np.zeros((nx,ny,24))
      t3=np.zeros((nx,ny,24))
      theta=np.zeros((nx,ny,24))
      g1=np.zeros((nx,ny,3,3))
      g2=np.zeros((nx,ny,3,3))
      gp=np.zeros((nx,ny,3,3))
      gp1=np.zeros((nx,ny,3,3))
      gp2=np.zeros((nx,ny,3,3))
      gq=np.zeros((nx,ny,3,3))
      gq1=np.zeros((nx,ny,3,3))
      gq2=np.zeros((nx,ny,3,3))
      m=np.zeros((nx,ny,24,3,3))

      #converting in the form of matrices for both grains
      gp1[:,:,0,0]=np.cos(p1)
      gp1[:,:,1,0]=-np.sin(p1)
      gp1[:,:,0,1]=np.sin(p1)
      gp1[:,:,1,1]=np.cos(p1)
      gp1[:,:,2,2]=1
      gp2[:,:,0,0]=np.cos(p2)
      gp2[:,:,1,0]=-np.sin(p2)
      gp2[:,:,0,1]=np.sin(p2)
      gp2[:,:,1,1]=np.cos(p2)
      gp2[:,:,2,2]=1
      gp[:,:,0,0]=1
      gp[:,:,1,1]=np.cos(p)
      gp[:,:,1,2]=np.sin(p)
      gp[:,:,2,1]=-np.sin(p)
      gp[:,:,2,2]=np.cos(p)
      gq1[:,:,0,0]=np.cos(q1)
      gq1[:,:,1,0]=-np.sin(q1)
      gq1[:,:,0,1]=np.sin(q1)
      gq1[:,:,1,1]=np.cos(q1)
      gq1[:,:,2,2]=1
      gq2[:,:,0,0]=np.cos(q2)
      gq2[:,:,1,0]=-np.sin(q2)
      gq2[:,:,0,1]=np.sin(q2)
      gq2[:,:,1,1]=np.cos(q2)
      gq2[:,:,2,2]=1
      gq[:,:,0,0]=1
      gq[:,:,1,1]=np.cos(q)
      gq[:,:,1,2]=np.sin(q)
      gq[:,:,2,1]=-np.sin(q)
      gq[:,:,2,2]=np.cos(q)
      g1=np.matmul(np.matmul(gp2,gp),gp1)
      g2=np.matmul(np.matmul(gq2,gq),gq1)

      #symmetry matrices considering the 24 symmteries for cubic system
      T=np.zeros((24,3,3));
      T[0,:,:]=[[1,0,0],[0,1,0],[0, 0 ,1]]
      T[1,:,:]=[[0,0,-1],  [0 ,-1 ,0], [-1, 0 ,0]]
      T[2,:,:]=[[0, 0 ,-1],  [ 0 ,1, 0],  [ 1 ,0 ,0]]
      T[3,:,:]=[[-1 ,0 ,0],  [ 0 ,1, 0],  [ 0 ,0 ,-1]]
      T[4,:,:]=[[0, 0 ,1],  [ 0 ,1 ,0],  [ -1, 0 ,0]]
      T[5,:,:]=[[1, 0 ,0],  [ 0 ,0 ,-1],  [ 0 ,1 ,0]]
      T[6,:,:]=[[1 ,0 ,0],  [ 0 ,-1 ,0],  [ 0 ,0 ,-1]]
      T[7,:,:]=[[1, 0 ,0],  [ 0 ,0, 1],  [ 0 ,-1 ,0]]
      T[8,:,:]=[[0 ,-1, 0],  [ 1 ,0 ,0],  [ 0 ,0 ,1]]
      T[9,:,:]=[[-1, 0 ,0],  [ 0 ,-1, 0],  [ 0 ,0 ,1]]
      T[10,:,:]=[[0, 1 ,0],  [ -1 ,0, 0],  [ 0 ,0 ,1]]
      T[11,:,:]=[[0, 0 ,1],  [ 1 ,0 ,0],  [ 0 ,1 ,0]]
      T[12,:,:]=[[0, 1 ,0],  [ 0, 0 ,1],  [ 1 ,0 ,0]]
      T[13,:,:]=[[0 ,0 ,-1],  [ -1 ,0 ,0],  [ 0, 1 ,0]]
      T[14,:,:]=[[0 ,-1 ,0],  [ 0 ,0 ,1],  [ -1 ,0 ,0]]
      T[15,:,:]=[[0, 1 ,0],  [ 0, 0 ,-1],  [ -1,0 ,0]]
      T[16,:,:]=[[0 ,0 ,-1],  [ 1 ,0 ,0],  [ 0 ,-1 ,0]]
      T[17,:,:]=[[0 ,0 ,1],  [ -1, 0 ,0],  [ 0, -1, 0]]
      T[18,:,:]=[[0 ,-1 ,0],  [ 0 ,0 ,-1],  [ 1 ,0 ,0]]
      T[19,:,:]=[[0 ,1 ,0],  [ 1 ,0 ,0],  [ 0 ,0 ,-1]]
      T[20,:,:]=[[-1 ,0 ,0],  [ 0 ,0 ,1],  [ 0 ,1, 0]]
      T[21,:,:]=[[0, 0 ,1],  [ 0 ,-1 ,0],  [ 1 ,0 ,0]]
      T[22,:,:]=[[0 ,-1 ,0],  [ -1, 0, 0],  [ 0 ,0 ,-1]]
      T[23,:,:]=[[-1, 0 ,0],  [ 0, 0 ,-1],  [ 0 ,-1 ,0]]

      T = np.array(T[None,None,...])

      #finding the 24 misorientation matrices(also can be calculated for 576 matrices)
      for i in range(24):
        m[:,:,i,:,:]=np.matmul(np.linalg.inv(np.matmul(T[:,:,i,:,:],g1)),g2)
        t1[:,:,i]=m[:,:,i,0,0]
        t2[:,:,i]=m[:,:,i,1,1]
        t3[:,:,i]=m[:,:,i,2,2]
        theta[:,:,i]=np.arccos(0.5*(t1[:,:,i]+t2[:,:,i]+t3[:,:,i]-1))




      #minimum of 24 angles is taken as miorientation angle
      ansRad=np.nanmin(theta,axis=-1)
      ansTheta=ansRad*180.0/np.pi
      return ansTheta


def takesolid(ea_cs):
    solid_pos = np.where(ea_cs[..., 3:4] > 0.99)
    return ea_cs[solid_pos[0], solid_pos[1], :3]

# load the CRNN Model from file
def load_model(model_file='./Setup_0'):
    model_para = np.load(model_file+'/model_setting.npy',allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location='cpu'))
    #ca = torch.load(model_file, map_location='cpu')
    return ca

def red_t(ini_t, num_t, sp_rate, CHANNEL_N, val_size, data):
    for i in range(len(ini_t)):
        if i == 0:
            x_initial = data[:, ini_t[i], ...].numpy()
            y = data[:, (ini_t[i]+sp_rate):data.shape[1]:sp_rate].numpy()
            if y.shape[1] > num_t:
                y = y[:, :num_t]
            elif y.shape[1] < num_t:
                while y.shape[1] < num_t:
                    y = np.concatenate((y, y[:, -1:]), axis=1)
        else:
            x_initial = np.concatenate([x_initial, data[:, ini_t[i], ...].numpy()], axis=0)
            y_tmp = data[:, (ini_t[i]+sp_rate):data.shape[1]:sp_rate].numpy()
            if y_tmp.shape[1] > num_t:
                y_tmp = y_tmp[:, :num_t]
            elif y_tmp.shape[1] < num_t:
                while y_tmp.shape[1] < num_t:
                    y_tmp = np.concatenate((y_tmp, y_tmp[:, -1:]), axis=1)
            y = np.concatenate((y, y_tmp), axis=0)

    seed = np.zeros([x_initial.shape[0], CHANNEL_N, x_initial.shape[2], x_initial.shape[3]],
                        np.float32)
    seed[:, :5, ...] = x_initial.astype(np.float32)
    x = torch.from_numpy(seed)
    y = torch.from_numpy(y)

    # split train/valid set
    ord = np.array(range(x.shape[0]))
    np.random.shuffle(ord)
    x = x[ord]
    y = y[ord]
    return x[:-val_size], y[:-val_size], x[-val_size:], y[-val_size:]

def add_sample(num_t, sp_rate, CHANNEL_N, val_size):
    nca_train_data = np.load('./train_data_multi_1525k.npy', allow_pickle=True)


    # reorder the data and normalized it, first three channels are Euler angles, third is phase state and last is undercooling
    nca_train_data = np.concatenate([nca_train_data[..., 1:4], nca_train_data[..., 0:1], nca_train_data[..., 4:5]],
                                    axis=-1)
    nca_train_data = np.array(nca_train_data)
    # normalized phase state: solid 1, liquid 0
    nca_train_data[nca_train_data[..., 3] != -1.0, 3] = 0.0
    nca_train_data[..., 3] += 1.0
    nca_train_data[nca_train_data == 3.0 * np.pi] = 0.0
    nca_train_data[..., :3] = nca_train_data[..., :3] / (
            0.5 * np.pi)  # normalize Euler angle
    nca_train_data[..., 4] = nca_train_data[..., 4] / 100.0  # normalized undercooling

    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 4, 2, 3]).type(torch.FloatTensor)

    ini_t = [i for i in range(0,250,20)]
    x_train, y_train, x_val, y_val = red_t(ini_t, num_t, sp_rate, CHANNEL_N, val_size, nca_train_data)
    return x_train, y_train, x_val, y_val


import torch.distributed as dist
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

from torch.utils.data.distributed import DistributedSampler
def prepare(x,y,rank,world_size, batch_size=20, pin_memory=False,num_workers=0):
    dataset = torch.utils.data.TensorDataset(x.to(rank), y.to(rank))
    sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=False,drop_last=False)
    dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=pin_memory,
                            num_workers=num_workers,drop_last=False,shuffle=False,sampler=sampler)

    return dataloader

import torch.multiprocessing as mp
if __name__ == '__main__':
    nx = ny = 128  # domain size
    nz = 1
    cell_len = 1e-6  # mesh size
    rec_step = 200  # total time step in the training data
    delta_t = 4e-6  # time increment for CA , unit is s
    ea_type = '2D'  # 2D or quasi-3D
    sample_n = 80  # sample number for EA setting
    T_type = 'non-iso'  # type of temperature, iso is isothermal, noniso is nonisothermal
    T_len = 10  # if >1, rotated temperature is included
    T_min = 15  # minimum undercooling for the temperatrure gradient
    T_max = 45  # maximum undercooling for the temperatrure gradient
    T_iso = 20  # if isothermal case, the undercooling

    '''
    nca_train_data, nca_train_time = gen_data(nx, ny, nz, cell_len, rec_step, delta_t, sample_n,
                                              T_type, T_len, T_min, T_max, T_iso, ea_type)

    np.save('./train_data_multi_'+str(rec_step)+'step.npy',nca_train_data)

    # if load data from file
    '''
    nca_train_time = np.array(range(rec_step))
    nca_train_data = np.load('../train_data_multi_1545rot.npy', allow_pickle=True)[:400]
    #nca_train_data = np.load('../train_data_multi_'+str(rec_step)+'step.npy', allow_pickle=True)
    #nca_train_data = np.load('../train_data_multi_'+str(rec_step)+'step_z10.npy',allow_pickle=True)

    # reorder the data and normalized it, first three channels are Euler angles, third is phase state and last is undercooling
    nca_train_data = np.concatenate([nca_train_data[..., 1:4], nca_train_data[..., 0:1], nca_train_data[..., 4:5]],
                                    axis=-1)
    nca_train_data = np.array(nca_train_data)
    # normalized phase state: solid 1, liquid 0
    nca_train_data[nca_train_data[..., 3] != -1.0, 3] = 0.0
    nca_train_data[..., 3] += 1.0
    nca_train_data[nca_train_data == 3.0 * np.pi] = 0.0
    nca_train_data[..., :3] = nca_train_data[..., :3] / (
            0.5 * np.pi)  # normalize Euler angle
    nca_train_data[..., 4] = nca_train_data[..., 4] / 100.0  # normalized undercooling
    print(nca_train_data.shape)

    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 4, 2, 3]).type(torch.FloatTensor)

    parameters = {
        "lr": [1e-3],
        "step_size": [150],
        "gamma": [0.3],
        "in_dim": [11],
        "neu_num": [96],
        "hid_lay_num": [5],
        "kernel_size": [3],
        "epoch": [400],
        "cs_pena": [1.0],
        "time_fac": [1.0],
        "rand_seed": [3024],
        # "padding_mode": ["replicate"],
        "speedup_rate": [2.0],
        "batch_size": [2],
        "tot_t": [50],
       # "prob_neuron": [32,64],
       # "prob_hidlay": [3,7],
       # "retrain": [True],
        # "reg_para": [1e-7],
        # "reg_exp": [1.0,2.0],
    }

    world_size = 3

    settings = list(itertools.product(*parameters.values()))
    i = 0
    folder_name = str(os.getcwd())
    for setup in settings:
        print("###################################")
        print('setup:  No.', i + 1)
        folder_path = folder_name + "/Setup_" + str(i)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        setup_properties = parameters
        j = 0
        for key in parameters:
            setup_properties[key] = setup[j]
            j = j + 1
        print(setup_properties)
        print('data stored at: ', folder_path)
        print("###################################")
        setup_path = folder_path + '/model_setting.npy'
        np.save(setup_path, setup_properties)
        mp.spawn(train, args=(world_size, nca_train_time, nca_train_data, setup_properties, folder_path), nprocs=world_size)
        i = i + 1
    print("ending training")



