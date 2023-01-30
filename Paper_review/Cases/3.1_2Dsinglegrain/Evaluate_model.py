import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision.transforms as T
import numpy as np
from LoadFunc_train import *
from CA_test_run import *
from Loadmodel_T_train import *

# function for run individual test run and  calculate average accuracy and rsme
def evaluate_model(nx, ny, nz, cell_len, CHANNEL_N, ca, sp_rate, delta_t, run_n=10, max_frame=120, ini_seed=40, ini_mode='random', ini_pos='random', ea_mode='single', ea_normalize=[85.0 * np.pi / 180.0, 100.0], include_T=False, T_ini=20.0, cooling=False, cr=0.0):
    for run_i in range(run_n):
        rsme_tmp, acc_tmp, rsme_misori_tmp, com_sp_tmp = CA_routine(nx, ny, nz, cell_len, CHANNEL_N, ca, run_i, sp_rate, delta_t, ini_seed, ini_mode, ini_pos, ea_mode, ea_normalize, include_T, T_ini, cooling, cr)
        rsme_tmp = np.array(rsme_tmp)
        acc_tmp = np.array(acc_tmp)*100.0
        com_sp_tmp = np.array(com_sp_tmp)
        rsme_misori_tmp = np.array(rsme_misori_tmp)*100.0
        if len(rsme_tmp) != max_frame:
            for ttt in range(max_frame - len(rsme_tmp)):
                rsme_tmp = np.append(rsme_tmp, rsme_tmp[-1])
                acc_tmp = np.append(acc_tmp, acc_tmp[-1])
                rsme_misori_tmp = np.append(rsme_misori_tmp, rsme_tmp[-1])
        if run_i == 0:
            rsme = rsme_tmp[None, ...]
            acc = acc_tmp[None, ...]
            rsme_misori = rsme_misori_tmp[None,...]
            com_sp = com_sp_tmp
        else:
            rsme = np.vstack((rsme, rsme_tmp[None, ...]))
            acc = np.vstack((acc, acc_tmp[None, ...]))
            rsme_misori = np.vstack((rsme_misori, rsme_misori_tmp[None, ...]))
            com_sp = np.append(com_sp,com_sp_tmp)
    print("The computation time for NCA single step is "+str(np.mean(com_sp))+ str(u" \u00B1 ")+str(np.std(com_sp))+" s ")

    # save the model rsme and accuracy at different run
    np.savetxt('model_rsme.csv', rsme, delimiter=',')
    np.savetxt('model_acc.csv', acc, delimiter=',')
    np.savetxt('model_rsme_misori.csv', rsme_misori, delimiter=',')
    rsme_m = np.mean(rsme, axis=0)
    rsme_std = np.std(rsme, axis=0)
    acc_m = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    rsme_misori_m = np.mean(rsme_misori, axis=0)
    rsme_misori_std = np.std(rsme_misori, axis=0)
    print("The RSME of prediction in last frame is "+str(rsme_m[-1])+ str(u" \u00B1 ") + str(rsme_std[-1]))
    print("The accuracy of prediction in last frame is " + str(acc_m[-1])+str(u" \u00B1 ")+str(acc_std[-1])+" % ")
    print("The RSME of misorientation in last frame is " + str(rsme_misori_m[-1]) + str(u" \u00B1 ")+ str(rsme_misori_std[-1])+" % " )

    # plot the rsme and accuracy
    plt.figure(4,figsize=(35,10))
    kwargs = {'color': 'b', 'alpha': 0.2}
    plt.subplot(1,3,1)
    plt.plot(rsme_m)
    plt.fill_between(np.array(range(max_frame)), rsme_m - rsme_std, rsme_m + rsme_std, **kwargs)
    plt.title("RSME")
    plt.xlabel("Step")
    plt.ylabel("RSME")
    plt.subplot(1,3,2)
    plt.plot(acc_m)
    plt.fill_between(np.array(range(max_frame)), acc_m - acc_std, acc_m + acc_std, **kwargs)
    plt.title("Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.subplot(1,3,3)
    plt.plot(rsme_misori_m)
    plt.fill_between(np.array(range(max_frame)), rsme_misori_m - rsme_misori_std, rsme_misori_m + rsme_misori_std, **kwargs)
    plt.title("RSME of misorientation")
    plt.xlabel("Step")
    plt.ylabel("RSME of misorientation (%)")
    plt.savefig('evaluate.png',dpi=900)
    plt.show()


# load the CRNN Model from file
model_file = './model.pkl'
ca = load_model(model_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#ca = ca.to(device)

# evaluate the trained model
nx = ny = 55
nz = 1
cell_len = 1e-6
delta_t = 8e-6
CHANNEL_N = 11
sp_rate = 1
run_n = 10
max_frame = 1100
ini_seed = 1
ini_mode ='random'
ini_pos = 'mid'#random'
ea_mode = 'single'
ea_normalize = [90.0 * np.pi / 180.0, 100.0]
include_T = True
cooling = False#True
cr=0.0
T_ini = 20.0#np.repeat(np.linspace(20, 25, nx)[..., None, None], ny, 1)# 22.5#20.0 #
#rot = T.RandomRotation(degrees=(0,360))
#T_tmp = torch.tensor(T_ini).type(torch.float32).permute(2, 0, 1)
#T_ini = rot(T_tmp).permute(1, 2, 0).detach().numpy()

evaluate_model(nx, ny, nz, cell_len, CHANNEL_N, ca, sp_rate, delta_t, run_n, max_frame, ini_seed, ini_mode, ini_pos, ea_mode, ea_normalize, include_T, T_ini)