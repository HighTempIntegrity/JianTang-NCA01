import numpy as np
from LoadFunc_train import *
from CA import *
from generate_train_data import *
from Loadmodel_T_train import *

nx = ny = 55 # domain size
nz = 1
cell_len = 1e-6 #mesh size
rec_step = 200 # total time step in the training data
delta_t = 8e-6 # time increment for CA , unit is s
ea_type = '3D' # 2D or quasi-3D
sample_n = 529 # sample number for EA setting
T_type = 'iso' # type of temperature, iso is isothermal, noniso is nonisothermal
T_len = 1 # if >1, rotated temperature is included
T_min = 20 # minimum undercooling for the temperatrure gradient
T_max = 25 # maximum undercooling for the temperatrure gradient
T_iso = 20 # if isothermal case, the undercooling
'''
nca_train_data, nca_train_time = gen_data(nx, ny, nz, cell_len, rec_step, delta_t, sample_n,
                                          T_type, T_len, T_min, T_max, T_iso, ea_type)

np.save('./train_data_multi_'+str(rec_step)+'step.npy',nca_train_data)
'''
# if load data from file
nca_train_time = np.array(range(rec_step))


nca_train_data = np.load('./train_data_multi_529.npy',allow_pickle=True)
#nca_train_data_tmp = np.load('./train_data_multi_400.npy',allow_pickle=True)

#nca_train_data = np.concatenate((nca_train_data, nca_train_data_tmp), axis=0)
nca_train_data_tmp = np.load('./train_data_multi_400_lg.npy',allow_pickle=True)
nca_train_data = np.concatenate((nca_train_data, nca_train_data_tmp), axis=0)
nca_train_data_tmp = np.load('./train_data_multi_400_lg2.npy',allow_pickle=True)
nca_train_data = np.concatenate((nca_train_data, nca_train_data_tmp), axis=0)

'''
for i in range(13):
    nca_train_data_tmp = np.load('E:\JianTang/rnn/test/nca_train_data'+str(i)+'.npy')
    if i == 0:
         nca_train_data = np.array(nca_train_data_tmp)
    else:
        #print(nca_train_data_tmp.shape)
        nca_train_data = np.concatenate((nca_train_data, nca_train_data_tmp), axis=0)
nca_train_data = np.concatenate((nca_train_data, nca_train_data[...,0:1]*0.0+20.0), axis=-1)

print(nca_train_data.shape)
'''


# reorder the data and normalized it, first three channels are Euler angles, third is phase state and last is undercooling
nca_train_data = np.concatenate([nca_train_data[..., 1:4], nca_train_data[..., 0:1], nca_train_data[..., 4:5]], axis=-1)
nca_train_data = np.array(nca_train_data)
# normalized phase state: solid 1, liquid 0
nca_train_data[nca_train_data[..., 3] != -1.0, 3] = 0.0
nca_train_data[..., 3] += 1.0
nca_train_data[nca_train_data == 3.0 * np.pi] = 0.0
nca_train_data[..., :3] = nca_train_data[..., :3] / (
            0.5 * np.pi) # normalize Euler angle
nca_train_data[..., 4] = nca_train_data[..., 4] / 100.0  # normalized undercooling
print(nca_train_data.shape)

nca_train_data = torch.from_numpy(nca_train_data)
nca_train_data = nca_train_data.permute([0, 1, 4, 2, 3]).type(torch.FloatTensor)

'''
for i in range(0,nca_train_data.shape[1],60):
    plt.imshow(zoom((np.array(nca_train_data[0,i]))))
    plt.pause(1)
    if i == 0:
        show_img = np.array(nca_train_data[0,i])
        #print(show_img[show_img[...,0]!=20.0/80.0,0])
    else:
        show_img = np.vstack((show_img, nca_train_data[0,i]))
show_img = torch.from_numpy(show_img)
plt.imshow(zoom((show_img)))


for i in range(nca_train_data.shape[0]):
    #plt.imshow(zoom((np.array(nca_train_data[i, -1]))))
    #plt.pause(1)
    if i == 0:
        show_img = np.array(nca_train_data[i, -1])
        print(show_img[show_img[...,0]!=20.0/80.0,0])
    else:
        show_img = np.vstack((show_img, nca_train_data[i, -1]))
show_img = torch.from_numpy(show_img)
plt.imshow(zoom((show_img)))
'''


def train(nca_train_time, nca_train_data, device, parameterization, fold_path):
    loss_step = []  # store the training loss
    rand_seed = parameterization.get("rand_seed", -1)
    if rand_seed != -1:
        #random.seed(rand_seed)
        #np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        #torch.cuda.manual_seed_all(rand_seed)
    ca = NCA(parameterization).to(device)
    ca.initialize_weights()
    retrain = parameterization.get("retrain", False)
    if retrain:
        ca = torch.load('./model.pkl').to(device)

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

    for epoch in range(epoch_num):

        for ini_t in [0]:

            x_initial = nca_train_data[:, ini_t, ...].numpy()
            seed = np.zeros([x_initial.shape[0], CHANNEL_N, x_initial.shape[2], x_initial.shape[3]], np.float32)
            seed[:, :5, ...] = x_initial.astype(np.float32)
            x = torch.from_numpy(seed)
            with torch.no_grad():
                nca_train_time_tmp = (np.array(nca_train_time) - nca_train_time[ini_t]).astype(int)
                total_step_num = nca_train_time_tmp[-50]

            batch_size = int(parameterization.get("batch_size", 60))

            #size for validation set
            val_size = 80#int(x.shape[0]//10)

            train_dataset = torch.utils.data.TensorDataset(x[:-val_size], nca_train_data[:-val_size])

            training_set = DataLoader(train_dataset, batch_size=batch_size)

            for j, (x_batch, xt_batch) in enumerate(training_set):
                l_valid = 0.0
                l_time_sum = 0.0
                x_batch = x_batch.to(device)
                for nca_step in range(ini_t, total_step_num + 1, sp_rate):

                    if ((nca_step in nca_train_time_tmp) & (nca_step != ini_t)):
                        target_img = xt_batch[:, np.where(nca_train_time_tmp == (nca_step - ini_t))[0][0]].to(device)
                        l_time_sum += weigt_loss(x_batch, target_img)

                    for time_i in range(time_fac):
                        try:
                            x_batch = ca(x_batch)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception
                # l_time_sum += regularization_param * regularization(ca,regularization_exp)
                with torch.no_grad():
                    l_time_sum.backward()
                    loss_step.append(l_time_sum.item())
                    for p in ca.parameters():
                        p.grad /= (p.grad.norm() + 1e-8)
                    optimizer.step()
                    optimizer.zero_grad()
                    if torch.cuda.is_available() & (epoch%20!=0):
                        del x_batch
                        del target_img
                        torch.cuda.empty_cache()
        with torch.no_grad():
            scheduler.step()

            if epoch % 20 == 0:

                torch.save(ca, path)
                acc_train = cal_acc(x_batch, target_img)
                del x_batch
                del target_img
                x_valid = x[-val_size:].to(device)
                target_valid = nca_train_data[-val_size:]
                for nca_step in range(ini_t, total_step_num + 1, sp_rate):
                    if ((nca_step in nca_train_time_tmp) & (nca_step != ini_t)):
                        target_img = target_valid[:, np.where(nca_train_time_tmp == (nca_step - ini_t))[0][0]].to(
                            device)
                        l_valid += weigt_loss(x_valid, target_img).item()


                    for time_i in range(time_fac):
                        try:
                            x_valid = ca(x_valid)  # _input)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception
                # l_valid += regularization_param * ca.regularization().item()
                acc_val = cal_acc(x_valid, target_img)
                print(epoch, "training loss: ", l_time_sum.item(), "valid loss: ", l_valid,
                      "training acc: ", acc_train, "%   valid accuracy: ", acc_val, "%")

                del x_valid
                del target_valid
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

    with torch.no_grad():
        torch.save(ca, path)
        x_valid = x[-val_size:].to(device)
        target_valid = nca_train_data[-val_size:]
        acc_train = cal_acc(x_batch, target_img)
        del x_batch
        del target_img
        for nca_step in range(ini_t, total_step_num + 1, sp_rate):
            if ((nca_step in nca_train_time_tmp) & (nca_step != ini_t)):
                target_img = target_valid[:, np.where(nca_train_time_tmp == (nca_step - ini_t))[0][0]].to(device)
                l_valid += weigt_loss(x_valid, target_img).item()

            for time_i in range(time_fac):
                try:
                    x_valid = ca(x_valid)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
        # l_valid += regularization_param * ca.regularization().item()

        acc_val = cal_acc(x_valid, target_img)
        print(epoch, "training loss: ", l_time_sum.item(), "valid loss: ", l_valid,
              "training acc: ", acc_train, "%   valid accuracy: ", acc_val, "%")




        # plot 3 validation samples
        for ea_i in range(3):
            if ea_i == 0:
                show_img = (x_valid.permute(0, 2, 3, 1))[ea_i].cpu().detach().numpy()
            else:
                show_img = np.vstack((show_img, (x_valid.permute(0, 2, 3, 1))[ea_i].cpu().detach().numpy()))
        del x_valid
        del target_valid
        torch.cuda.empty_cache()
        plt.subplot(1, 2, 1)
        plt.imshow(zoom(show_img[..., :3]))
        plt.savefig('prediction.jpg')
        plt.subplot(1, 2, 2)
        plt.plot(loss_step, '.')
        plt.title('fold_path')
        # plt.yscale('log')
        plt.savefig(path + '.jpg')
        print("######### training end ##########")
    # torch.distributed.barrier()

def cal_acc(x_in,y_in):
    acc = []
    x = (x_in.permute(0, 2, 3, 1)).detach().cpu().numpy()
    nx = x.shape[1]
    ny = x.shape[2]
    x = x * (x[..., 4:5] > 1e-10)

    x_true = (y_in.permute(0, 2, 3, 1)).detach().cpu().numpy()
    # calculate the difference between NCA and CA
    for i in range(len(x)):
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


def ensambletraning(parameters, nca_train_time, nca_train_data):
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        train(nca_train_time, nca_train_data, device, setup_properties, folder_path)
        i = i + 1


parameters = {
    "lr": [1e-3],
    "step_size": [2000],
    "gamma": [0.3],
    "in_dim": [11],
    "neu_num": [96],
    "hid_lay_num": [5],
    "kernel_size": [3],
    "epoch": [5000],
    "cs_pena": [1.0],
    "time_fac": [10.0],
    "rand_seed": [1024],
    "batch_size": [150],
    # "padding_mode": ["replicate"],
    "speedup_rate": [10.0],
    #"retrain": [True],
    # "reg_para": [1e-7],
    # "reg_exp": [1.0,2.0],
}
ensambletraning(parameters, nca_train_time, nca_train_data)
print("ending training")



