import numpy as np
import cv2 as cv
import sys
from math import ceil

# 设置最大递归次数，默认值是1000
sys.setrecursionlimit(10000)

vis = None
max_length = 0
max_list = []
cur_list = []


def get_m_nei_path(img):
    """ 解析出一条m-邻域的路径 """
    kernel1 = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    kernel2 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8)
    kernel3 = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    kernel4 = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)

    mask = cv.morphologyEx(img, cv.MORPH_ERODE, kernel1, iterations=1, borderValue=0)
    img = img - mask
    mask = cv.morphologyEx(img, cv.MORPH_ERODE, kernel2, iterations=1, borderValue=0)
    img = img - mask
    mask = cv.morphologyEx(img, cv.MORPH_ERODE, kernel3, iterations=1, borderValue=0)
    img = img - mask
    mask = cv.morphologyEx(img, cv.MORPH_ERODE, kernel4, iterations=1, borderValue=0)
    img = img - mask
    return img


def get_bounding(x, y, height, width):
    """ 求出边界框 """
    left = y - 1 if y > 1 else 0
    top = x - 1 if x > 1 else 0
    right = y + 1 if y < width - 1 else width - 1
    bottom = x + 1 if x < height - 1 else height - 1

    return left, top, right, bottom


def get_start_points(img):
    """计算出所有的起点，要求8-邻域内只有一个点"""
    height, width = img.shape

    location = np.where(img == 255)
    x_inx = location[0]
    y_inx = location[1]
    length = len(x_inx)

    res = []

    for k in range(length):
        x = x_inx[k]
        y = y_inx[k]

        left, top, right, bottom = get_bounding(x, y, height, width)

        # 计算出邻居个数
        nei = -1
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                if img[i, j] == 255:
                    nei += 1

        # 如果邻居个数是1，加入起点数组
        if nei == 1:
            res.append([x, y])

    return np.array(res)


def dfs(img, x, y):
    global vis, max_list

    if len(cur_list) > len(max_list):
        max_list = cur_list.copy()

    vis[x, y] = 1
    height, width = img.shape
    left, top, right, bottom = get_bounding(x, y, height, width)

    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if img[i, j] == 255 and vis[i, j] == 0:
                cur_list.append((i, j))
                dfs(img, i, j)
                cur_list.pop()


def smooth_one_connective(img):
    """ 平滑一个联通分支 """
    global vis, max_list, cur_list
    vis = None
    max_list = []
    cur_list = []

    height, width = img.shape

    # 计算出一条唯一的m-邻接的路径
    # img = get_m_nei_path(img)
    # cv.imwrite('img.png', img)

    # 获得所有的起点
    location = get_start_points(img)

    # 进行dfs，求出最长的路径
    for x, y in location:
        vis = np.zeros((height, width))
        cur_list.append((x, y))
        dfs(img, x, y)
        cur_list.pop()

    # 标记结果
    res = np.zeros((height, width), dtype=np.uint8)
    for x, y in max_list:
        res[x, y] = 255

    return res


def init():
    """ 初始化全局变量 """
    global vis, max_list, max_length, cur_list
    vis = None
    max_length = 0
    max_list = []
    cur_list = []


def smooth_lines(img):
    """ 平滑所有的联通分支 """
    init()

    # 计算出每一个联通分支
    nums, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    d = np.max(labels)

    # 将每一个联通分支的结果进行加和
    res = np.zeros(img.shape, dtype=np.uint8)
    for i in range(1, d+1):
        con = np.zeros(img.shape, dtype=np.uint8)
        location = np.where(labels == i)
        con[location] = 255
        line = smooth_one_connective(con)
        res += line

    return res


def visualization(img, line):
    """ 可视化选择的线 """
    height, width = img.shape
    visual = np.zeros((height, width, 3), dtype=np.uint8)
    visual[:, :, 0] = img
    visual[:, :, 1] = img
    visual[:, :, 2] = img

    location = np.where(line == 255)
    visual[location] = [100, 200, 150]

    return visual


if __name__ == '__main__':
    image = cv.imread('line2.png')
    image = image[:, :, 0]
    # img = np.zeros(image.shape, dtype=np.uint8)
    # img[:, 1:] = image[:, 0:-1]
    # image = img
    lines = smooth_lines(image)
    visual = visualization(image, lines)


    # cv.imshow('visual', visual)
    cv.imwrite('visual.png', visual)
    cv.waitKey(0)
