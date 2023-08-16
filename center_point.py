import numpy as np
import cv2 as cv

vis = None
point_list = []


def get_start_points(img):
    """计算出所有的起点，要求8-邻域内只有一个点"""
    height, width = img.shape

    location = np.where(img == 255)
    x_inx = location[0]
    y_inx = location[1]
    length = len(x_inx)


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
            return x, y

    return None


def get_bounding(x, y, height, width):
    """ 求出边界框 """
    left = y - 1 if y > 1 else 0
    top = x - 1 if x > 1 else 0
    right = y + 1 if y < width - 1 else width - 1
    bottom = x + 1 if x < height - 1 else height - 1

    return left, top, right, bottom


def dfs(img, x, y):
    global vis, point_list
    # 将当前遍历到的顶点加入栈
    point_list.append((x, y))

    vis[x, y] = 1
    height, width = img.shape
    left, top, right, bottom = get_bounding(x, y, height, width)

    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if img[i, j] == 255 and vis[i, j] == 0:
                dfs(img, i, j)


def get_center_point(img):
    """ 计算线段中点，每次传入的图像只能包含一个线段 """
    global vis, point_list
    height, width = img.shape

    vis = np.zeros((height, width))
    point_list = []

    location = np.where(img == 255)
    x_inx = location[0]
    y_inx = location[1]

    # 找到一个边界点，从边界点开始遍历
    starts = get_start_points(img)
    if starts is None:
        raise Exception('cannot find start point')
    dfs(img, starts[0], starts[1])

    return point_list[int(len(point_list) / 2)]


def visualization(img, center):
    """ 可视化选择的线 """
    height, width = img.shape
    visual = np.zeros((height, width, 3), dtype=np.uint8)
    visual[:, :, 0] = img
    visual[:, :, 1] = img
    visual[:, :, 2] = img

    for i in range(-2, 3):
        for j in range(-2, 3):
            x = center[0] + i
            y = center[1] + j
            if 0 <= x < height and 0 <= y < width:
                visual[x, y] = [0, 255, 0]

    return visual


if __name__ == '__main__':
    image = cv.imread('line2.png')
    image = image[:, :, 0]
    center = get_center_point(image)
    print(center)

    visual = visualization(image, center)
    cv.imshow('visual', visual)
    cv.waitKey(0)
