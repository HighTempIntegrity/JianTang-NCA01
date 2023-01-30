# Cellular Automata for solidification
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import time
from LoadFunc_train import *
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap = matplotlib.cm.get_cmap("turbo") # define a colormap


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


def EulerDistribution(pred, true):
    pred = takesolid(pred) * 180.0 / np.pi
    true = takesolid(true) * 180.0 / np.pi

    kwargs = dict(histtype='stepfilled', alpha=0.5, bins=10, range=(-5, 95))
    plt.subplot(3, 1, 1)
    ea1_p = plt.hist(pred[..., 0], **kwargs, label='pred')
    ea1_t = plt.hist(true[..., 0], **kwargs, label='true')
    plt.xlabel('EA1')
    plt.ylabel('Frequency')
    plt.subplot(3, 1, 2)
    ea2_p = plt.hist(pred[..., 1], **kwargs, label='pred')
    ea2_t = plt.hist(true[..., 1], **kwargs, label='true')
    plt.xlabel('EA2')
    plt.ylabel('Frequency')
    plt.subplot(3, 1, 3)
    ea3_p = plt.hist(pred[..., 2], **kwargs, label='pred')
    ea3_t = plt.hist(true[..., 2], **kwargs, label='true')
    plt.xlabel('EA3')
    plt.ylabel('Frequency')
    plt.subplots_adjust(hspace=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig('EA_dis.png', bbox_inches='tight')
    plt.show()
    return pd.DataFrame({'ea1_p_upper': ea1_p[1][1:], 'ea_1p_count': ea1_p[0], 'ea_1t_count': ea1_t[0],
                         'ea2_p_upper': ea2_p[1][1:], 'ea_2p_count': ea2_p[0], 'ea_2t_count': ea2_t[0],
                         'ea3_p_upper': ea3_p[1][1:], 'ea_3p_count': ea3_p[0], 'ea_3t_count': ea3_t[0]})


#############################################################################
# CA framework for solidification
def create_block(cell_block, nx, ny, nz, nei_tmp):
    for zi in range(nz):
        for yi in range(ny):
            for xi in range(nx):
                # initialize the cell number for each cell
                cell_block[xi, yi, zi, 1] = zi * nx * ny + yi * nx + xi + 1
                # initialize the octahedron center as the cell center
                cell_block[xi, yi, zi, 8] = xi + 1
                cell_block[xi, yi, zi, 9] = yi + 1
                cell_block[xi, yi, zi, 10] = zi + 1

                for ni in range(len(nei_tmp)):
                    if nei_tmp[ni][0] == 0 and nei_tmp[ni][1] == 0 and nei_tmp[ni][2] == 0:  # we skip the middle cell
                        continue
                    xi_tmp = xi + nei_tmp[ni][0]
                    yi_tmp = yi + nei_tmp[ni][1]
                    zi_tmp = zi + nei_tmp[ni][2]

                    if xi_tmp < 0 or xi_tmp > nx - 1 or yi_tmp < 0 or yi_tmp > ny - 1 or zi_tmp < 0 or zi_tmp > nz - 1:
                        continue
                    if cell_block[xi_tmp, yi_tmp, zi_tmp, 0] == 1.0:
                        cell_block[xi, yi, zi, 13 + ni] = zi_tmp * nx * ny + yi_tmp * nx + xi_tmp + 1
    return cell_block


def initialization(nx, ny, nz, file=None):
    # initialize the CA cell block with 38 channels as following:
    # Shape number, Cell number, Grain ID, Cell state, three Euler angle
    # Temperature, Octahedron center coordinate XYZ
    # Octahedron length, critical nucleation, Cell number of 27 neighbors
    # If there are less neighbors, the empty one will be filled with 0
    # shape number =1, the cell exist; otherwise it is not; this is design for complicated shape
    cell_block = np.zeros([nx, ny, nz, 40])

    # initilizalize the critical undercooling as -100.0; potential nucleaus is with positive critical undercooling
    cell_block[..., 12] = -100.0
    cell_block[..., 0] = 1.0  # initialize the shape number as 1, shape number = 0 will not be considered as a neighbor

    # initialize the domain as fully liquid
    cell_block[..., 3] = -1.0
    cell_block[..., 4:7] = 3.0 * np.pi

    nei_tmp = np.stack(np.meshgrid(range(-1, 2), range(-1, 2), range(-1, 2)),
                       -1).reshape(-1, 3)  # define moore neighborhood
    cell_block = create_block(cell_block, nx, ny, nz, nei_tmp)

    return cell_block


## Melt
def melt(cell_block):
    # find the undercooling above zero, change their CS=-1 and EA= 3*pi
    melt_cell = np.where(cell_block[..., 7] <= 0.0)
    cell_block[melt_cell[0], melt_cell[1], melt_cell[2], 3] = -1
    cell_block[melt_cell[0], melt_cell[1], melt_cell[2], 4:7] = 3.0 * np.pi

    # find the edge of melt pool and assign them into interface cell CS=0
    m = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    max_cs = m(torch.from_numpy(-1.0 * cell_block[..., 3])[None, ...])[
        0, ...].detach().numpy()  # find the maximum cell state number in a 3*3*3 neighborhood
    inter_loc = np.where((max_cs == 1.0) & (cell_block[..., 3] > 0.0))
    cell_block[inter_loc[0], inter_loc[1], inter_loc[2], 3] = 0.0
    return cell_block


## Grain growth
def growth(cell_block, cell_len, delta_t):
    # cell_len is the length of the cell
    # delta_t is the time interval
    # Be careful that all the length value in cell_block are normalized by cell_len

    # update all the octahedron length in interface cells
    inter_pos = np.where(cell_block[..., 3] == 0)
    cell_block[inter_pos[0], inter_pos[1], inter_pos[2], 11] = \
        (np.array(cell_block[inter_pos[0], inter_pos[1], inter_pos[2], 11])
         * cell_len + den_grow_v(
                    np.array(cell_block[inter_pos[0], inter_pos[1], inter_pos[2], 7])) * delta_t) / cell_len

    # find the edge of melt pool and assign them into interface cell CS=0
    m = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    max_cs = m(torch.from_numpy(cell_block[..., 3])[None, ...])[0, ...].detach().numpy()
    liq_int = np.where(
        (max_cs == 0) & (cell_block[..., 3] == -1.0))  # find the liquid cells (CS =-1) near the interface cell (cs=0)
    # search all the liquid cell near the interface cell and judge whether it will capture by the interface cell
    for i in range(len(liq_int[0])):

        liq_x = liq_int[0][i]
        liq_y = liq_int[1][i]
        liq_z = liq_int[2][i]
        liq_cell_tmp = np.array(cell_block[liq_x, liq_y, liq_z, :])
        if liq_cell_tmp[0] != 1.0:
            continue

        nei_liq_d = np.zeros(
            27) + 0.001  # a list to store the distance between the liquid cell center and the nearest octahedron surface
        nei_rule = np.array([[-1, -1, -1, 0],
                             [-1, -1, 0, 1],
                             [-1, -1, 1, 2],
                             [0, -1, -1, 3],
                             [0, -1, 0, 4],
                             [0, -1, 1, 5],
                             [1, -1, -1, 6],
                             [1, -1, 0, 7],
                             [1, -1, 1, 8],
                             [-1, 0, -1, 9],
                             [-1, 0, 0, 10],
                             [-1, 0, 1, 11],
                             [0, 0, -1, 12],
                             [0, 0, 0, 13],
                             [0, 0, 1, 14],
                             [1, 0, -1, 15],
                             [1, 0, 0, 16],
                             [1, 0, 1, 17],
                             [-1, 1, -1, 18],
                             [-1, 1, 0, 19],
                             [-1, 1, 1, 20],
                             [0, 1, -1, 21],
                             [0, 1, 0, 22],
                             [0, 1, 1, 23],
                             [1, 1, -1, 24],
                             [1, 1, 0, 25],
                             [1, 1, 1, 26]])
        # define moore neighborhood

        nei_tmp = nei_rule[cell_block[liq_x, liq_y, liq_z, 13:13 + 27] != 0]
        nei_tmp[..., 0] = nei_tmp[..., 0] + liq_x
        nei_tmp[..., 1] = nei_tmp[..., 1] + liq_y
        nei_tmp[..., 2] = nei_tmp[..., 2] + liq_z
        inter_pos = np.where(
            cell_block[nei_tmp.transpose()[0], nei_tmp.transpose()[1], nei_tmp.transpose()[2], 3] == 0.0)
        for j in range(len(inter_pos[0])):
            nei_pos_num = inter_pos[0][j]
            nei_cell_tmp = np.array(cell_block[nei_tmp.transpose()[0][nei_pos_num],
                                               nei_tmp.transpose()[1][nei_pos_num], nei_tmp.transpose()[2][
                                                   nei_pos_num], ...])
            nei_liq_d[nei_tmp.transpose()[3][nei_pos_num]] = cap_judge(liq_cell_tmp, nei_cell_tmp, cell_len)

        # judge whether the liquid cell is captured
        if min(nei_liq_d) <= 0:  # if so the liquid cell is captured by at least one neighbor interface cell
            min_nei_int = cell_block[
                liq_x, liq_y, liq_z, 13 + np.argmin(
                    nei_liq_d)]  # find the neighbor interface cell has octahedron center closer to the liquid cell center
            nei_pos = np.where(cell_block[..., 1] == min_nei_int)
            nei_cell_x = nei_pos[0][0]
            nei_cell_y = nei_pos[1][0]
            nei_cell_z = nei_pos[2][0]
            nei_cell_tmp = np.array(cell_block[nei_cell_x, nei_cell_y, nei_cell_z, :])
            # turn the captured cell into interface state and copy the grain information from the parent interface cell
            cell_block[liq_x, liq_y, liq_z, 3] = 0.0
            cell_block[liq_x, liq_y, liq_z, 2] = cell_block[nei_cell_x, nei_cell_y, nei_cell_z, 2]
            cell_block[liq_x, liq_y, liq_z, 4:7] = cell_block[nei_cell_x, nei_cell_y, nei_cell_z, 4:7]

            # calculate the octahedron center and length for the new captured interface cell
            oct_len_new, oct_cen_x, oct_cen_y, oct_cen_z = cal_oct(liq_cell_tmp, nei_cell_tmp, cell_len)
            cell_block[liq_x, liq_y, liq_z, 8] = oct_cen_x
            cell_block[liq_x, liq_y, liq_z, 9] = oct_cen_y
            cell_block[liq_x, liq_y, liq_z, 10] = oct_cen_z
            cell_block[liq_x, liq_y, liq_z, 11] = oct_len_new

    # for all the interface cell without liquid neighbor, turn it into fully solid (CS=1)
    m = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    max_cs = m(torch.from_numpy(-1.0 * cell_block[..., 3])[None, ...])[0, ...].detach().numpy()
    cell_to_soild = np.where(
        (max_cs <= 0.0) & (cell_block[..., 3] == 0.0))  # find the interface cell without liquid neighbor
    cell_block[cell_to_soild[0], cell_to_soild[1], cell_to_soild[2], 3] = 1

    return cell_block


# @jit(nopython=True)
def cap_judge(liq_cell_tmp, nei_cell_tmp, cell_len):
    # Euler angle in interface cell
    iu = nei_cell_tmp[4:7]
    # define the Rotation matrix of Euler angle which can transform into the crystallographic coordination system
    xk = (liq_cell_tmp[8:11] - nei_cell_tmp[8:11]) * cell_len
    xk = xk.reshape(3, 1)
    ETM = np.array([[np.cos(iu[0]) * np.cos(iu[2]) - np.cos(iu[1]) * np.sin(iu[0]) * np.sin(iu[2]),
                     -np.sin(iu[2]) * np.cos(iu[0]) - np.cos(iu[2]) * np.cos(iu[1]) * np.sin(iu[0]),
                     np.sin(iu[1]) * np.sin(iu[0])],
                    [np.cos(iu[2]) * np.sin(iu[0]) + np.sin(iu[2]) * np.cos(iu[0]) * np.cos(iu[1]),
                     -np.sin(iu[0]) * np.sin(iu[2]) + np.cos(iu[0]) * np.cos(iu[1]) * np.cos(iu[2]),
                     -np.cos(iu[0]) * np.sin(iu[1])],
                    [np.sin(iu[2]) * np.sin(iu[1]), np.cos(iu[2]) * np.sin(iu[1]), np.cos(iu[1])]])
    xk = ETM.transpose().dot(xk)
    d = (np.sum(np.abs(xk)) - nei_cell_tmp[11] * cell_len) / (3 ** 0.5)
    return d


def cal_oct(liq_cell_tmp, nei_cell_tmp, cell_len):
    # Euler angle in interface cell
    iu = nei_cell_tmp[4:7]
    # define the Rotation matrix of Euler angle which can transform into the crystallographic coordination system
    xk = (liq_cell_tmp[8:11] - nei_cell_tmp[8:11]) * cell_len
    xk = xk.reshape([3, 1])
    ETM = np.array([[np.cos(iu[0]) * np.cos(iu[2]) - np.cos(iu[1]) * np.sin(iu[0]) * np.sin(iu[2]),
                     -np.sin(iu[2]) * np.cos(iu[0]) - np.cos(iu[2]) * np.cos(iu[1]) * np.sin(iu[0]),
                     np.sin(iu[1]) * np.sin(iu[0])],
                    [np.cos(iu[2]) * np.sin(iu[0]) + np.sin(iu[2]) * np.cos(iu[0]) * np.cos(iu[1]),
                     -np.sin(iu[0]) * np.sin(iu[2]) + np.cos(iu[0]) * np.cos(iu[1]) * np.cos(iu[2]),
                     -np.cos(iu[0]) * np.sin(iu[1])],
                    [np.sin(iu[2]) * np.sin(iu[1]), np.cos(iu[2]) * np.sin(iu[1]), np.cos(iu[1])]])
    xk = ETM.transpose().dot(xk)
    d = (np.sum(np.abs(xk)) - nei_cell_tmp[11] * cell_len) / (
                3 ** 0.5)  # calculate the distance between the liq center to its nearest octahedron surface

    # find the index of the nearest octaherdon surface
    hkl = np.array(xk)
    xk0 = np.where(xk == 0)
    if len(xk0[0]) != 0:
        for i in range(len(xk0[0])):
            hkl[xk0[0][i], xk0[1][i]] = 1.0  # replace the 0 by 1
    hkl = hkl / np.abs(hkl)  # get the index of nearest octahedron surface

    # project the liquid center into nearest edge of octahedron
    xa = xk + abs(d) * hkl / (3 ** 0.5)

    # find the nearest corner s1, and the other two corner of the nearest surface s2, s3
    corn = np.array([[hkl[0][0], 0, 0], [0, hkl[1][0], 0], [0, 0, hkl[2][0]]]).transpose() * nei_cell_tmp[11] * cell_len
    s1_mark = np.where(abs(xa) == np.max(abs(xa)))[0][0]
    s1 = corn[:, s1_mark]
    # delete s1 from corn so the other two are s2,s3
    s1_mark = np.where(s1 != 0)[0][0]
    corn = np.delete(corn, s1_mark, axis=1)

    # project xa into the edge of s1s2 and s2s3, calculate the distance of projected points to the s1
    is1 = (np.array(corn[:, 0] - s1).transpose().dot(xa - s1) / (np.linalg.norm(corn[:, 0] - s1) ** 2)) * (
                corn[:, 0] - s1)
    js1 = (np.array(corn[:, 1] - s1).transpose().dot(xa - s1) / (np.linalg.norm(corn[:, 1] - s1) ** 2)) * (
            corn[:, 1] - s1)

    # calculate the length of s1s2 and s3s1
    slen = corn
    slen[:, 0] = slen[:, 0] - s1
    slen[:, 1] = slen[:, 1] - s1

    # calulate the edge of new octahedron
    L12 = (min(np.linalg.norm(is1), cell_len * (3.0 ** 0.5)) +
           min((np.linalg.norm(slen[:, 0]) - np.linalg.norm(is1)), cell_len * (3.0 ** 0.5))) / 2.0
    L13 = (min(np.linalg.norm(js1), cell_len * (3.0 ** 0.5)) +
           min((np.linalg.norm(slen[:, 1]) - np.linalg.norm(js1)), cell_len * (3.0 ** 0.5))) / 2.0
    Lk = max(L12, L13) * ((2.0 / 3.0) ** 0.5)

    # calculate the center of new octahedron
    xck = (nei_cell_tmp[11] * cell_len - Lk * (3 ** 0.5)) * s1 / np.linalg.norm(s1)
    ck = np.array(nei_cell_tmp[8:11]).transpose() * cell_len + ETM.dot(xck)
    liq_cell_oct = ck / cell_len

    return Lk * (3 ** 0.5) / cell_len, liq_cell_oct[0], liq_cell_oct[1], liq_cell_oct[2]


def den_grow_v(T):
    # dendrite growth velocity function
    # T is undercooling
    a1 = 4.6818 * (10 ** (-10))
    a2 = 3.1456 * (10 ** (-6))
    devt = a1 * (T ** 2) + a2 * (T ** 3)
    return devt



def interface_acc(cell_block_nca, cell_block_ca):
    inter_pos = np.where((cell_block_ca[..., 3] == 0) | (cell_block_nca[..., 3] == 0))
    diff_cell = np.where(cell_block_ca[inter_pos[0], inter_pos[1], inter_pos[2], 2] ==
                         cell_block_nca[inter_pos[0], inter_pos[1], inter_pos[2], 2])
    acc = len(diff_cell[0]) * 100.0 / len(inter_pos[0])
    return acc


def CA_routine(nx, ny, nz, cell_len, CHANNEL_N, ca, run_i, sp_rate, delta_t, ini_seed=40, ini_mode='random', ini_pos='random', ea_mode='single', ea_normalize=[85.0 * np.pi / 180.0, 100.0], include_T=False, T_ini=20.0,  cooling=False, cr=0.0):
    # intialize the ca block
    cell_block = initialization(nx, ny, nz)

    # time_step_num
    t_step_ca = 0.0

    t_step = 0.0

    # time
    t_ca = 0.0

    cell_block[..., 7] = T_ini  # setting constant undercooling = 20 K

    cell_block_ca = np.array(cell_block)

    rsme = []
    acc = []
    rsme_misori = []
    NCA_comp_speed = []

    with VideoWriter('ca_nca_compare.mp4') as vid:
        with VideoWriter('nca_hid_channel.mp4') as vid_hc:
            with VideoWriter('ca_speedup.mp4') as vid_ca:
                while t_step >= 0:

                    # setting initial nucleus
                    if t_step_ca == 0:
                        seed_num = ini_seed
                        if ini_pos == 'random':
                            ran_x = np.random.randint(cell_block_ca.shape[0], size=seed_num)
                            ran_y = np.random.randint(cell_block_ca.shape[1], size=seed_num)
                        elif ini_pos == 'mid':
                            ran_x = [nx//2]
                            ran_y = [ny//2]
                        # the range of Euler angle for initial nucluei
                        if ea_mode == 'single':
                            seed_ea_size = 1
                        elif ea_mode == 'full':
                            seed_ea_size = 3
                        if ini_mode == 'random':
                            ran_ea_i = np.random.randint(90, size=[seed_num, seed_ea_size])
                        elif ini_mode == 'each5':
                            ran_ea_i = np.random.randint(17, size=[seed_num, seed_ea_size]) * 5.0
                        elif ini_mode == 'each10':
                            ran_ea_i = np.random.randint(9, size=[seed_num, seed_ea_size]) * 10.0
                        elif ini_mode == '205080':
                            ran_ea_tmp = np.array([20.0, 50.0, 80.0])
                            ran_ea_i = ran_ea_tmp[np.random.randint(3, size=[seed_num, seed_ea_size])]

                        # ran_ea = np.random.randint(85,size=seed_num)
                        np.savetxt('ini_state'+str(run_i)+'.csv',[ran_x,ran_y,ran_ea_i[...,0]],delimiter=',')
                        #ran_ea_i = np.array([80.0,20.0,50.0,80.0,20.0])
                        #ran_ea_i = ran_ea_i[...,None]
                        #ran_y = np.array([nx//6,nx//3,nx//2,nx//2+nx//6,nx//2+nx//3])
                        #ran_x = np.array([ny//2,ny//2,ny//2,ny//2,ny//2])*0+10
                        for ran_i in range(seed_num):
                            cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 3] = 0.0
                            cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 4] = ran_ea_i[ran_i, 0] * np.pi / 180.0
                            cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 2] = 1.0
                            # test for single EA or full euler space
                            if ea_mode == 'single':
                                cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 5] = 0.0
                                cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 6] = 0.0
                            elif ea_mode == 'full':
                                cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 5] = ran_ea_i[ran_i,1]*np.pi/180.0
                                cell_block_ca[ran_x[ran_i], ran_y[ran_i], 0, 6] = ran_ea_i[ran_i,2]*np.pi/180.0

                        cell_block_ca2 = np.array(cell_block_ca)
                        t_step_ca2 = t_step_ca
                        t_ca2 = t_ca
                    if cooling:
                        cell_block_ca[..., 7] = cell_block_ca[..., 7] + cr
                    nca_train_time = []
                    # see whether there are still liquid cell
                    liq_cell_ca = len(np.where((cell_block_ca[..., 3] == -1.0) & (cell_block_ca[..., 7] != 0.0))[0])

                    nca_train_data = np.array(cell_block_ca[..., 0, 3:7])[None, ...]

                    # compare nca with ca
                    if t_step_ca == 0:
                        # initial setting of NCA
                        if not include_T:
                            x_initial = np.array(cell_block_ca[..., 0, 3:7])
                            x_initial = np.append(x_initial[..., 1:], x_initial[..., 0:1], axis=-1)
                            x_initial = np.array(x_initial)
                            x_initial[x_initial[..., -1] != -1.0, -1] = 0.0
                            x_initial[..., -1] += 1.0
                            x_initial[x_initial == 3.0 * np.pi] = 0.0
                            x_initial[..., :3] = x_initial[..., :3] / ea_normalize[0]
                        else:
                            x_initial = np.array(cell_block_ca[..., 0, 3:8])
                            x_initial = np.concatenate([x_initial[..., 1:4], x_initial[..., 0:1], x_initial[..., 4:5]], axis=-1)
                            x_initial = np.array(x_initial)
                            x_initial[x_initial[..., 3] != -1.0, 3] = 0.0
                            x_initial[..., 3] += 1.0
                            x_initial[x_initial == 3.0 * np.pi] = 0.0
                            x_initial[..., :3] = x_initial[..., :3] / ea_normalize[0]
                            x_initial[..., 4] = x_initial[..., 4] / ea_normalize[1] # if consider temperature effect, the fifth channel will be normalized temperature

                        x_initial = torch.from_numpy(x_initial).permute(2, 0, 1).detach().numpy()

                        seed = np.zeros([1, CHANNEL_N, x_initial.shape[1], x_initial.shape[2]], np.float32)
                        if not include_T:
                            seed[:, :4, ...] = x_initial.astype(np.float32)
                        else:
                            seed[:, :5, ...] = x_initial.astype(np.float32)
                        x = torch.from_numpy(seed)

                        device =  torch.device('cpu')# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        x = x.to(device)
                        ca = ca.to(device)
                        start = time.time()
                        x = ca(x)
                        end = time.time()
                        NCA_comp_speed.append(end - start)
                        if liq_cell_ca != 0.0:
                            cell_block_ca2, t_ca2, t_step_ca2 = ca_sp_step(cell_block_ca2, t_ca2, t_step_ca2, cell_len, 'ca', sp_rate, delta_t)

                    elif ((t_step_ca > 0) & (t_step_ca % sp_rate == 0)):

                        x = x * (x[:,4:5,...]>1e-10)
                        if device.type=='cuda':
                            x_train = (x.permute(0, 2, 3, 1)).detach().cpu().numpy()
                        else:
                            x_train = (x.permute(0, 2, 3, 1)).detach().numpy()

                        x_true = np.array(nca_train_data)
                        x_true = np.append(x_true[..., 1:], x_true[..., 0:1], axis=-1)
                        x_true = np.array(x_true)
                        x_true[x_true[..., -1] != -1.0, -1] = 0.0
                        x_true[..., -1] += 1.0
                        x_true[x_true == 3.0 * np.pi] = 0.0

                        # calculate the difference between NCA and CA
                        diff = np.abs(
                            (x_train[0, ..., :3]* ea_normalize[0] - x_true[-1, ..., :3])) # absolute difference
                        mis_ori = cal_misori(x_train[0, ..., :3] * ea_normalize[0],
                                            x_true[-1, ..., :3]) # misorientation angle
                        filter = mis_ori > 15.0  # diff>10.0

                        # store the rsme and accuracy in the middle part of the domain
                        rsme.append((np.sum(
                            np.square(diff) / (3.0 * nx  * ny ))) ** 0.5)
                        acc.append(1.0 - np.sum(filter)/(nx*ny))
                        rsme_misori.append((np.sum(np.square(mis_ori)/(nx * ny))**0.5)/90.0)
                        '''
                        # draw the distribution of three Euler angle
                        ea_dis = EulerDistribution(np.concatenate((x_train[0,
                                                                   nx // 2 - nx // 3:nx // 2 + nx // 3,
                                                                   ny // 2 - ny // 3:ny // 2 + ny // 3,
                                                                   :3] * ea_normalize[0],
                                                                   x_train[0,
                                                                   nx // 2 - nx // 3:nx // 2 + nx // 3,
                                                                   ny // 2 - ny // 3:ny // 2 + ny // 3, 3:4]),
                                                                  axis=-1), x_true[0, nx // 2 - nx // 3:nx // 2 + nx // 3,
                                                                            ny // 2 - ny // 3:ny // 2 + ny // 3])
                        '''

                        # show the difference between CA with speed-up rate of 1 and NCA with the selected speed-up rate
                        filter = filter[..., None]
                        x_true = x_true / ea_normalize[0]
                        if ea_mode == 'single':
                            img1 = cmap(x_true[-1, ..., 0])[...,:3]  # true growth from CA
                            img2 = cmap(x_train[0, ..., 0])[...,:3]   # prediction from NCA
                        else:
                            img1 = x_true[-1, ..., 0:3] # true growth from CA
                            img2 = x_train[0, ..., 0:3]# prediction from NCA
                        show_img = np.hstack((img1, img2, cmap(mis_ori* (np.array(filter)[...,0]) / 90.0)[...,:3])) # show the CA, NCA, difference > 10, and difference
                        plt.imshow(zoom(show_img))
                        vid.add(zoom(show_img))

                        # show the hidden channel
                        hid_ch = x_train[0, ..., 4:]
                        # sort the hidden channel based on range of channel values
                        hid_ch_m = np.max(np.max(hid_ch,axis=0),axis=0)-np.min(np.min(hid_ch,axis=0),axis=0)
                        sort_ch = np.argsort(hid_ch_m)

                        for cha_si in range(len(sort_ch)):
                          max_pos = sort_ch[int(len(sort_ch)-cha_si-1)]
                          if cha_si == 0:
                            hid_ch_tmp = hid_ch[...,int(max_pos):int(max_pos+1)]
                          else:
                            hid_ch_tmp = np.concatenate([hid_ch_tmp,hid_ch[...,int(max_pos):int(max_pos+1)]],-1)

                        hid_ch = hid_ch_tmp
                        #hid_ch = (hid_ch- hid_ch.min()[...,None]) / (hid_ch.max()-hid_ch.min())[...,None]

                        hid_ch = (hid_ch - hid_ch.min(axis=0, keepdims=True).min(axis=1, keepdims=True)) / \
                                 (hid_ch.max(axis=0, keepdims=True).max(axis=1, keepdims=True) - hid_ch.min(axis=0, keepdims=True).min(axis=1, keepdims=True))

                        if hid_ch.shape[-1] < 3:
                            for hid_ch_i in range(hid_ch.shape[-1],3):
                                hid_ch = np.concatenate([hid_ch, hid_ch[...,0:1]*0.0], -1)
                            show_img2 = hid_ch
                        else:
                            for hid_i in range(hid_ch.shape[-1] // 3):
                                if hid_i == 0:
                                    show_img2 = hid_ch[..., 0:3]
                                else:
                                    show_img2 = np.hstack((show_img2, hid_ch[..., 3 * hid_i:3 * (
                                                hid_i + 1)]))
                        plt.imshow(show_img2)
                        vid_hc.add(zoom(show_img2))

                        # if the cooling rate os not zero
                        if cooling:
                            x[:, 4, ...] = torch.from_numpy(np.squeeze(cell_block_ca[...,7].astype(np.float32)/ ea_normalize[1]))#[None,...]


                        # update the grain distribution by NCA for this time step
                        start = time.time()
                        x = ca(x)
                        end = time.time()
                        NCA_comp_speed.append(end - start)

                        # compare the CA with speed-up rate 1 and selected speed-up rate
                        x_true2 = np.array(cell_block_ca2[..., 0, 3:7])[None, ...]
                        x_true2 = np.append(x_true2[..., 1:], x_true2[..., 0:1], axis=-1)
                        x_true2 = np.array(x_true2)
                        x_true2[x_true2[..., -1] != -1.0, -1] = 0.0
                        x_true2[..., -1] += 1.0
                        x_true2[x_true2 == 3.0 * np.pi] = 0.0
                        x_true2 = x_true2 / ea_normalize[0]
                        if ea_mode == 'single':
                            img3 = cmap(x_true2[-1, ..., 0])[...,:3]
                        else:
                            img3 = x_true2[-1, ..., :3]
                        diff = np.abs((x_true2[-1, ..., :3] - x_true[-1, ..., :3]))
                        mis_ori = cal_misori(x_true2[-1, ..., :3]* ea_normalize[0], x_true[-1, ..., :3]* ea_normalize[0])
                        filter = mis_ori > 15.0
                        show_img3 = np.hstack((img1, img3, cmap(mis_ori* (np.array(filter))/90.0)[...,:3]))
                        plt.imshow(zoom(show_img3))
                        # add image into video
                        vid_ca.add(zoom(show_img3))

                        if liq_cell_ca != 0.0:
                            cell_block_ca2, t_ca2, t_step_ca2 = ca_sp_step(cell_block_ca2, t_ca2, t_step_ca2, cell_len, 'ca',sp_rate, delta_t)


                    # judge whether quit the time loop
                    if (liq_cell_ca == 0.0) or t_step == 1000000:  # if there is no liquid or the time step is too much
                        print("############ run "+str(run_i)+ " End ################")
                        if device.type=='cuda':
                            x_train = (x.permute(0, 2, 3, 1)).detach().cpu().numpy()
                        else:
                            x_train = (x.permute(0, 2, 3, 1)).detach().numpy()
                        # save plot for the last frame
                        plt.figure(1)
                        ax = plt.subplot()
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
                        plt.imshow(zoom(show_img))
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        plt.savefig("./CA_NCA_Compare.png",dpi=900)
                        plt.show()

                        plt.figure(2)
                        ax2 = plt.subplot()
                        ax2.imshow(zoom(show_img2))
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        plt.savefig("./NCA_hid_channel.png",dpi=900)
                        plt.show()

                        plt.figure(1)
                        ax3 = plt.subplot()
                        plt.imshow(zoom(show_img3))
                        divider = make_axes_locatable(ax3)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        plt.savefig("./CA_speedup_Compare.png",dpi=900)
                        plt.show()

                        T_img = cmap(x_train[0, ..., 4] * ea_normalize[1])[..., :3]
                        plt.imshow(T_img)
                        plt.savefig("./T.png", dpi=900)

                        # save true and prediction data
                        np.save('ca_true' + str(run_i) + '.npy', x_true[0] * ea_normalize[0])
                        np.save('nca_pred' + str(run_i) + '.npy', np.concatenate((x_train[0, ...,
                                                                                  :3] * ea_normalize[0],
                                                                                  x_train[0, ...,
                                                                                  3:4],
                                                                                  x_train[0, ...,
                                                                                  4:5] * ea_normalize[1]), axis=-1))
                        #ea_dis.to_csv('EA_distribution' + str(run_i) + '.csv')

                        # plot the distribution of three Euler Angles
                        p_ea = (x_train * ea_normalize[0] * 180.0 / np.pi) #[0, nx // 2 - nx // 3:nx // 2 + nx // 3,
                               #ny // 2 - ny // 3:ny // 2 + ny // 3, :3]
                        t_ea = (x_true * ea_normalize[0] * 180.0 / np.pi) #[0, nx // 2 - nx // 3:nx // 2 + nx // 3,
                               #ny // 2 - ny // 3:ny // 2 + ny // 3]
                        plt.figure(3,figsize=(50,15))
                        plt.rcParams['xtick.direction'] = 'in'
                        plt.rcParams['ytick.direction'] = 'in'
                        plt.rcParams.update({'font.size': 40})
                        plt.subplot(1, 3, 1)
                        plt.scatter(p_ea[..., 0:1].reshape(-1), t_ea[..., 0:1].reshape(-1), alpha=0.003)
                        plt.plot([-10,100],[-10,100])
                        plt.xlabel('Pred ($^\circ$)',fontsize='large')
                        plt.ylabel('True ($^\circ$)',fontsize='large')
                        plt.xlim([-5, 95])
                        plt.ylim([-5, 95])
                        plt.title(r'$\phi$1',fontsize='large')

                        plt.rcParams['xtick.direction'] = 'in'
                        plt.rcParams['ytick.direction'] = 'in'
                        plt.rcParams.update({'font.size': 40})
                        plt.subplot(1, 3, 2)
                        plt.scatter(p_ea[..., 1:2].reshape(-1), t_ea[..., 1:2].reshape(-1), alpha=0.003)
                        plt.plot([-10, 100], [-10, 100])
                        plt.xlabel('Pred ($^\circ$)',fontsize='large')
                        plt.ylabel('True ($^\circ$)',fontsize='large')
                        plt.xlim([-5, 95])
                        plt.ylim([-5, 95])
                        plt.title(r'$\phi$2',fontsize='large')

                        plt.rcParams['xtick.direction'] = 'in'
                        plt.rcParams['ytick.direction'] = 'in'
                        plt.rcParams.update({'font.size': 40})
                        plt.subplot(1, 3, 3)
                        plt.rcParams['xtick.direction'] = 'in'
                        plt.rcParams['ytick.direction'] = 'in'
                        plt.scatter(p_ea[..., 2:3].reshape(-1), t_ea[..., 2:3].reshape(-1), alpha=0.003)
                        plt.plot([-10, 100], [-10, 100])
                        plt.xlabel('Pred ($^\circ$)',fontsize='large')
                        plt.ylabel('True ($^\circ$)',fontsize='large')
                        plt.xlim([-5, 95])
                        plt.ylim([-5, 95])
                        plt.title(r"$\phi$3",fontsize='large')
                        plt.savefig("./EA_dis.png",dpi=900)
                        plt.show()
                        print(t_step_ca)
                        return rsme, acc, rsme_misori, NCA_comp_speed


                    # single ca step
                    if liq_cell_ca != 0.0:
                        cell_block_ca, t_ca, t_step_ca = ca_sp_step(cell_block_ca, t_ca, t_step_ca, cell_len, 'ca', 1.0, delta_t)


def ca_sp_step(cell_block_tmp, t, t_step, cell_len, mode, sp_rate, delta_t):
    delta_t = delta_t * sp_rate
    # melt
    cell_block_tmp = melt(cell_block_tmp)

    # growth
    if mode == 'ca':
        cell_block_tmp = growth(cell_block_tmp, cell_len, delta_t)

    # update the time
    t = t + delta_t
    t_step = t_step + 1.0
    return cell_block_tmp, t, t_step


def saveinfo(x, ti):
    ea1 = x[..., 0]
    ea2 = x[..., 1]
    ea3 = x[..., 2]
    phase = x[..., 3]
    T = x[..., 4]
    plt.figure(1)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(ea1)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./ea1_" + str(ti) + ".png", dpi=900)
    plt.show()

    plt.figure(2)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(ea2)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./ea2_" + str(ti) + ".png", dpi=900)
    plt.show()

    plt.figure(3)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(ea3)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./ea3_" + str(ti) + ".png", dpi=900)
    plt.show()

    plt.figure(4)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(phase)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./phase_" + str(ti) + ".png", dpi=900)
    plt.show()

    plt.figure(5)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(T)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./T_" + str(ti) + ".png", dpi=900)
    plt.show()

    # show the hidden channel
    hid_ch = x[..., 4:].detach().numpy()
    # sort the hidden channel based on range of channel values
    hid_ch_m = np.max(np.max(hid_ch, axis=0), axis=0) - np.min(np.min(hid_ch, axis=0), axis=0)
    sort_ch = np.argsort(hid_ch_m)
    cha_si = 0
    max_pos = sort_ch[int(len(sort_ch) - cha_si - 1)]
    hid_ch_tmp = hid_ch[..., int(max_pos):int(max_pos + 1)]
    hid_ch_tmp = (hid_ch_tmp + np.min(np.min(hid_ch_tmp, axis=0), axis=0))/(np.max(np.max(hid_ch_tmp, axis=0), axis=0) - np.min(np.min(hid_ch_tmp, axis=0), axis=0))
    plt.figure(6)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(hid_ch_tmp)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./h1_" + str(ti) + ".png", dpi=900)
    plt.show()

    cha_si = 1
    max_pos = sort_ch[int(len(sort_ch) - cha_si - 1)]
    hid_ch_tmp = hid_ch[..., int(max_pos):int(max_pos + 1)]
    hid_ch_tmp = (hid_ch_tmp + np.min(np.min(hid_ch_tmp, axis=0), axis=0))/(np.max(np.max(hid_ch_tmp, axis=0), axis=0) - np.min(np.min(hid_ch_tmp, axis=0), axis=0))
    plt.figure(7)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(hid_ch_tmp)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./h2_" + str(ti) + ".png", dpi=900)
    plt.show()

    cha_si = 2
    max_pos = sort_ch[int(len(sort_ch) - cha_si - 1)]
    hid_ch_tmp = hid_ch[..., int(max_pos):int(max_pos + 1)]
    hid_ch_tmp = (hid_ch_tmp + np.min(np.min(hid_ch_tmp, axis=0), axis=0))/(np.max(np.max(hid_ch_tmp, axis=0), axis=0) - np.min(np.min(hid_ch_tmp, axis=0), axis=0))
    plt.figure(8)
    ax = plt.subplot()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    show_img = cmap(hid_ch_tmp)[..., :3]
    plt.imshow(zoom(show_img))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plt.savefig("./h3_" + str(ti) + ".png", dpi=900)
    plt.show()

