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



def CA_routine(nx, ny, nz, cell_len, ealist, ea_i, ea_loc_x, ea_loc_y, delta_t, rec_step, T_ini=20.0):
    # intialize the ca block
    cell_block = initialization(nx, ny, nz)

    # time_step_num
    t_step_ca = 0.0

    t_step = 0.0

    # time
    t_ca = 0.0

    cell_block[..., 7] = T_ini # this is just for debug!!!   setting constant undercooling = 20 K

    cell_block_ca = np.array(cell_block)

    nca_train_time = []

    while t_step >= 0:

        # assign initial nuclei
        if t_step_ca == 0:
            if ea_i >= 0:
                nuc_x = ea_loc_x
                nuc_y = ea_loc_y
                cell_block_ca[nuc_x, nuc_y, 0, 3] = 0.0
                cell_block_ca[nuc_x, nuc_y, 0, 4] = ealist[ea_i,0] * np.pi / 180.0
                cell_block_ca[nuc_x, nuc_y, 0, 2] = 1.0
                cell_block_ca[nuc_x, nuc_y, 0, 5] = ealist[ea_i,1] * np.pi / 180.0
                cell_block_ca[nuc_x, nuc_y, 0, 6] = ealist[ea_i,2] * np.pi / 180.0
            else:
                nuc_x = ea_loc_x
                nuc_y = ea_loc_y
                cell_block_ca[nuc_x, nuc_y, 0, 3] = 0.0
                cell_block_ca[nuc_x, nuc_y, 0, 4] = ealist[:,0] * np.pi / 180.0
                cell_block_ca[nuc_x, nuc_y, 0, 2] = 1.0
                cell_block_ca[nuc_x, nuc_y, 0, 5] = ealist[:,1] * np.pi / 180.0
                cell_block_ca[nuc_x, nuc_y, 0, 6] = ealist[:,2] * np.pi / 180.0

                # see whether there are still liquid cell
        liq_cell_ca =  len(np.where((cell_block_ca[..., 3] == -1.0) & (cell_block_ca[..., 7] != 0.0))[0])

        '''
        if t_step_ca%20 == 0:
            plt.imshow(cell_block_ca[...,0,4:7])
            plt.title("time step: "+str(t_step_ca)+"  T(K): "+str(T_ini))
            plt.show()
        '''

        # record Euler angle and cell state for NCA training
        if t_step_ca % 1 == 0:
            if len(nca_train_time) == rec_step:
                return nca_train_time, nca_train_data
            if len(nca_train_time) == 0:
                nca_train_time.append(t_step_ca)
                nca_train_data = np.array(cell_block_ca[..., 0, 3:8])[None, ...]
            else:
                nca_train_time.append(t_step_ca)
                nca_train_data = np.vstack((nca_train_data, np.array(cell_block_ca[..., 0, 3:8])[None, ...]))

        # judge whether quit the time loop
        if (liq_cell_ca == 0.0) or t_step == 100000:  # if there is no liquid or the time step is too much
            # nca_train_time.append(t_step_ca)
            if len(nca_train_time) != rec_step:
                nca_train_data = np.vstack((nca_train_data, cell_block_ca[..., 0, 3:8][None, ...]))
            return nca_train_time, nca_train_data

        # single ca step
        if liq_cell_ca != 0.0:
            cell_block_ca, t_ca, t_step_ca = ca_single_step(cell_block_ca, t_ca, t_step_ca, cell_len, 'ca', delta_t)


def ca_single_step(cell_block_tmp, t, t_step, cell_len, mode, delta_t):
    # melt
    cell_block_tmp = melt(cell_block_tmp)

    # growth
    if mode == 'ca':
        cell_block_tmp = growth(cell_block_tmp, cell_len, delta_t)

    # update the time
    t = t + delta_t
    t_step = t_step + 1.0
    return cell_block_tmp, t, t_step