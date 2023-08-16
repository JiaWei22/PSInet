import numpy as np
import cv2 as cv
import torch
import random
import scipy
from torch import nn as nn
from GeodisTK import geodesic2d_fast_marching
from skimage import morphology
from patchSim import RelevantMap
import math
from smoothLine import smooth_lines

def dist(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def gen_point(centroids,sorted_id,img_point_list,dis=20):

    if len(img_point_list)==0 and len(sorted_id)>0:
        img_point_list.append(centroids[sorted_id[0]])
        return centroids[sorted_id[0]]
    elif len(img_point_list)==0 and len(sorted_id)==0:
        img_point_list.append(centroids[0])
        return centroids[0]

    for i in range(len(sorted_id)):
        point_1 = centroids[sorted_id[i]]
        distances = [dist(point_1, prev) for prev in img_point_list]

        if all(distance > dis for distance in distances):
            img_point_list.append(point_1)
            return point_1
    return centroids[0]


def point_loc_with_noise(point,shape):
    x = int(point[0] + random.uniform(-2, 2))
    y = int(point[1] + random.uniform(-2, 2))
    x_limit = shape[1]
    y_limit = shape[2]
    if x <0:
        x = 0
    elif x >= x_limit:
        x = x_limit-1
    if y<0:
        y = 0
    elif y>=y_limit:
        y = y_limit-1
    return x,y


def find_max_region(img, k, connectivity=4):
    # img = img.cpu().numpy().astype(np.uint8)
    img = img.cpu().numpy().astype(np.uint8)
    """求解最大联通区域"""
    nums, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=connectivity)
    # 移除背景
    background = 0
    stats_no_bg = np.delete(stats, background, axis=0)
    # 选择出前k大
    arr1 = []
    for inx, item in enumerate(stats_no_bg):
        arr1.append((item[4], inx + 1))
    arr1.sort(reverse=True)
    k = min(k, len(arr1))
    ans = np.zeros(img.shape, dtype=np.float32)
    # 修改
    for i in range(k):
        loc = np.where(labels == arr1[i][1])
        ans[loc] = 1

    # 选择出前k大
    arr = []
    for inx, item in enumerate(stats_no_bg):
        arr.append(item[4])
    sorted_id = sorted(range(len(arr)), key=lambda k: arr[k], reverse=True)
    k = min(k, len(sorted_id))
    # 修改

    return k,sorted_id,centroids,ans

def make_hint(img ,result,pathSim,img_point_list,epoch,connectivity=4, pos = True,method ="scribbles",diameter = 15,use_sim=False):
    num = 10
    if method == "lines":
        num = 1

    k,sorted_id,centroids, ans = find_max_region(result,  num, connectivity=4)

    hint = np.zeros(result.shape, dtype=np.float32)
    point = gen_point(centroids, sorted_id, img_point_list)

    if method =="click":
        x, y = point_loc_with_noise(point, img.shape)
        hint[x, y] = 5000
        hint = cv.GaussianBlur(hint, (diameter, diameter), 0)
        if np.max(hint)!= 0:
            hint = (hint - np.min(hint)) / (np.max(hint) - np.min(hint))

    elif method =="disk":
        x, y = point_loc_with_noise(point, img.shape)
        hint[x, y] = 5000
        hint = cv.GaussianBlur(hint, (diameter, diameter), 0)
        hint[hint!=0] =1

    elif method == "scribbles":
        hint = ans

    elif method == "lines":
        temp = morphology.skeletonize(ans)

        if epoch >70:
            hint[temp] = 255
            hint_aft_smooth = smooth_lines(hint.astype(np.uint8))
            location = np.where(hint_aft_smooth == 255)
            hint = np.zeros(result.shape, dtype=np.float32)
            hint[location] = 1
        else:
            hint[temp] = 1

    elif method =="eGT":
        img = img.cpu().numpy().astype(np.float32).transpose(1,2,0)
        centroids= centroids.astype(np.uint8)
        S = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        x, y = point_loc_with_noise(point, img.shape)
        S[x][y] = 1
        hint = geodesic2d_fast_marching(img,S)
        hint =np.exp(-hint)

    elif method == "localeGT":
        radius = int(diameter/2)
        img = img.cpu().numpy().astype(np.float32).transpose(1,2,0)
        centroids= centroids.astype(np.uint8)
        localGTmap = np.zeros((img.shape[0], img.shape[1]), np.float32)
        h,w,c = img.shape
        x,y = point_loc_with_noise(point, img.shape)

        dis_x = min(x, radius)
        dis_y = min(y, radius)
        left_bound = max(x - radius, 0)
        up_bound = max(y - radius, 0)
        right_bound = min(x + radius, w)
        down_bound = min(y + radius, h)

        partimg = img[left_bound:right_bound,up_bound:down_bound,:]
        S=np.zeros((partimg.shape[0], partimg.shape[1]), np.uint8)
        S[dis_x,dis_y] = 1
        localhint = geodesic2d_fast_marching(partimg,S)
        localhint = np.exp(-localhint)
        localGTmap[left_bound:right_bound,up_bound:down_bound] = localhint[:,:]
        hint = localGTmap


    elif method == "localGT":
        radius = int(diameter / 2)
        img = img.cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        centroids = centroids.astype(np.uint8)
        localGTmap = np.zeros((img.shape[0], img.shape[1]), np.float32)
        h, w, c = img.shape
        x, y = point_loc_with_noise(point, img.shape)

        dis_x = min(x, radius)
        dis_y = min(y, radius)
        left_bound = max(x - radius, 0)
        up_bound = max(y - radius, 0)
        right_bound = min(x + radius, w)
        down_bound = min(y + radius, h)

        partimg = img[left_bound:right_bound, up_bound:down_bound, :]
        S = np.zeros((partimg.shape[0], partimg.shape[1]), np.uint8)
        S[dis_x, dis_y] = 1
        localhint = geodesic2d_fast_marching(partimg, S)
        # localhint = np.exp(-localhint)
        localGTmap[left_bound:right_bound, up_bound:down_bound] = localhint[:, :]
        hint = localGTmap


    hint = torch.from_numpy(hint)


    if len(sorted_id)!=0 and use_sim:
        patch_x,patch_y  = centroids[sorted_id[0]]
        patch_x, patch_y = int(patch_x),int(patch_y)
        pathSim,_ = pathSim.calc_relevantMap(patch_x, patch_y)
        pathSim = torch.from_numpy(pathSim)
    else:
        pathSim = 0

    return hint, pathSim



def user_interact_sim(img, cur, gt, real_batch, img_pathSim_list, img_point_list, patchsize, diameter,epoch, method = "lines", use_sim=False,
                      ):
    """
    模拟下一次迭代过程的用户输入
    :param groundTruth: 真实值
    :param cur: 上一轮迭代所输出分割结果
    :param k: 选择出的联通分支个数
    :return:
    """

    shape = gt.shape
    bs = shape[0]
    cur = cur.cuda()

    pos_domain = torch.zeros(shape).cuda()
    location = torch.where((cur - gt) <0)
    pos_domain[location] = 1

    neg_domain = torch.zeros(shape).cuda()
    location = torch.where((cur - gt) == 1)
    neg_domain[location] = 1

    pos = torch.zeros(shape).cuda()
    neg = torch.zeros(shape).cuda()

    if use_sim:

        x = int(shape[1]/patchsize)
        y = int(shape[1]/patchsize)
        pos_simmap = torch.zeros(bs, x, y).cuda()
        neg_simmap = torch.zeros(bs, x, y).cuda()


        for i in range(bs):
            pos[i], pos_simmap[i] = make_hint(img[i], pos_domain[i],img_pathSim_list[i]
                                               ,img_point_list[i][0],pos=True, method=method, diameter=diameter,
                                              use_sim=use_sim,epoch=epoch)
        for i in range(bs):
            neg[i], neg_simmap[i] = make_hint(img[i], neg_domain[i], img_pathSim_list[i], img_point_list[i][1],
                                                                     pos=False, method=method, diameter=diameter,use_sim=use_sim,epoch=epoch)
        return pos, neg, pos_simmap, neg_simmap
    else:
        temp = 0
        for i in range(bs):
            pos[i], _ = make_hint(img[i], pos_domain[i],temp,img_point_list[i][0],
                                              pos=True, method=method, diameter=diameter, use_sim=use_sim,epoch=epoch)
        for i in range(bs):
            neg[i], _ = make_hint(img[i], neg_domain[i],temp,img_point_list[i][1],
                                              pos=False, method=method, diameter=diameter, use_sim=use_sim,epoch=epoch)
        return pos, neg, _, _




if __name__ == '__main__':
    gt = cv.imread('1.png')
    cur = cv.imread(r'D:\iinet\MoNuSeg_split\test\img\TCGA-2Z-A9J9-01A-01-TS1_1_1.jpg')
    img =  cv.imread(r'D:\iinet\MoNuSeg_split\train\img\TCGA-2Z-A9J9-01A-01-TS1_1_1.jpg')
    k,sorted_id,centroids,ans = find_max_region(img, 3, connectivity=4)
    s = geodesic2d_fast_marching(img, centroids)
    cv.imwrite('img_pos', s)