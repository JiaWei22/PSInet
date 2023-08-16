import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from time import time
from math import ceil
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops


class RelevantMap:
    def __init__(self, img, patch_height=16, patch_width=16, alpha=0.8):

        self.scaleX = 1.0
        self.scaleY = 1.0
        self.img = img
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.nx = None
        self.ny = None
        self.width = None
        self.height = None
        self.patchesColorHist = None
        self.patchesTexture = None
        self.grayCoM_offset = 1
        self.alpha = alpha
        self.map_w =int(img.shape[0]/self.patch_width)
        self.map_h = int(img.shape[1] / self.patch_height)


        self.img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        self.channel = self.img.shape[2]
        self.patches, self.patches_gray = self.get_patches()
        self.heatmap = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.get_feature()

    def get_patches(self):

        heightPre = self.img.shape[0]
        widthPre = self.img.shape[1]


        self.nx = ceil(heightPre / self.patch_height)
        self.ny = ceil(widthPre / self.patch_width)

        self.height = self.nx * self.patch_height
        self.width = self.ny * self.patch_width
        self.img = cv.resize(self.img, dsize=(self.width, self.height), fx=1, fy=1, interpolation=cv.INTER_LINEAR)
        self.img_gray = cv.resize(self.img_gray, dsize=(self.width, self.height), fx=1, fy=1, interpolation=cv.INTER_LINEAR)

        self.scaleX = self.height / heightPre
        self.scaleY = self.width / widthPre

        patches = np.zeros((self.nx, self.ny, self.patch_height, self.patch_width, self.channel), dtype=np.uint8)
        patches_gray = np.zeros((self.nx, self.ny, self.patch_height, self.patch_width), dtype=np.uint8)
        for i in range(self.nx):
            xt = self.patch_height * i
            xd = self.patch_height * (i + 1)
            for j in range(self.ny):
                patches[i, j] = self.img[xt:xd, self.patch_width * j:self.patch_width * (j + 1), :]
                patches_gray[i, j] = self.img_gray[xt:xd, self.patch_width * j:self.patch_width * (j + 1)]

        return patches, patches_gray

    def clip_patch(self, x, y, mode='HSV'):

        x = int(self.scaleX * x)
        y = int(self.scaleY * y)

        xd = int(self.patch_height / 2)
        yd = int(self.patch_width / 2)

        xt = max(x - xd, 0)
        xd = min(self.height, xt + self.patch_height)
        yl = max(y - yd, 0)
        yr = min(self.width, yl + self.patch_width)

        res = None
        if mode == 'HSV':
            res = self.img[xt:xd, yl:yr, :]
        elif mode == 'GRAY':
            res = self.img_gray[xt:xd, yl:yr]
        res = cv.resize(res, dsize=(self.patch_width, self.patch_height), fx=1, fy=1, interpolation=cv.INTER_LINEAR)
        return res

    def get_feature(self):


        # 1. 计算颜色分布直方图
        self.patchesColorHist = np.zeros((self.nx, self.ny, self.channel, 256), dtype=np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                for c in range(self.channel):
                    hist = cv.calcHist([self.patches[i, j, :, :, c]], [0], None, [256], [0.0, 255.0])
                    self.patchesColorHist[i, j, c, :] = hist[:, 0]

        # 2. 计算纹理特征
        self.patchesTexture = np.zeros((256, 256, 4), dtype=np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                mat = graycomatrix(self.patches_gray[i, j], [self.grayCoM_offset],
                               [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True, normed=True)
                correlation = graycoprops(mat, 'contrast')
                self.patchesTexture[i, j, :] = correlation

    def color_relevant(self, patches_x, patches_y, template):
        NCC = 0
        for c in range(self.channel):
            res = cv.compareHist(self.patchesColorHist[patches_x, patches_y, c, :], template[c, :], method=cv.HISTCMP_CORREL)
            NCC += res
        return NCC / 3.0

    def calc_relevantMap(self, x, y, mode='COLOR_TEXTURE_HARMONIC'):
        """
        所需要调用的接口，计算点击的邻域与每一个patch相关性的的heatmap
        :param x, y: 对应图片上像素的位置
        :param mode:
            COLOR_TEXTURE_HARMONIC: 同时使用颜色和纹理，并计算纹理特征
            COLOR_ONLY: 只使用颜色
            TEXTURE_ONLY: 只是用纹理
        :return:
            heatmap: 和patch的个数相同的一个矩阵，是我们要的结果
            template: 截出来的那片小区域，如果尺寸和patch不相同的话，会进行插值转化为相同大小的
        """

        template = self.clip_patch(x, y, mode='HSV')
        template_gray = self.clip_patch(x, y, mode='GRAY')

        # 计算template颜色分布直方图
        templateColorHist = np.zeros((3, 256), dtype=np.float32)
        for c in range(self.channel):
            templateColorHist[c, :] = cv.calcHist([template[:, :, c]], [0], None, [256], [0.0, 255.0])[:, 0]

        # 1. 利用颜色分布直方图计算相似性，耗时2ms
        heatmap1 = np.zeros((self.nx, self.ny), dtype=np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                heatmap1[i, j] = self.color_relevant(i, j, templateColorHist)
        location = np.where(heatmap1 < 0)
        heatmap1[location] = 0

        # 2. 利用SSIM计算，耗时长，效果及其不好
        # heatmap2 = np.zeros((self.nx, self.ny), dtype=np.float32)
        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         heatmap2[i, j] = ssim(template, self.patches[i, j], multichannel=True)

        # 3. 计算灰度共生矩阵
        heatmap3 = np.zeros((self.nx, self.ny), dtype=np.float32)
        mat = graycomatrix(template_gray, [self.grayCoM_offset], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
        correlation = graycoprops(mat, 'contrast')[0]
        for i in range(self.nx):
            for j in range(self.ny):
                heatmap3[i, j] = np.linalg.norm(self.patchesTexture[i, j] - correlation)
        # heatmap3 = 1.0 / np.square(heatmap3)
        # heatmap3 = (heatmap3 - np.average(heatmap3) + 0.0000001) / np.std(heatmap3)
        # heatmap3 = heatmap3 - np.min(heatmap3)
        # heatmap3 = heatmap3 / np.std(heatmap3)
        # heatmap3 = np.log(heatmap3)
        heatmap3 = np.exp(-heatmap3 / np.std(heatmap3))

        # 整合两个
        if mode == 'COLOR_TEXTURE_HARMONIC':
            heatmap = 2 * heatmap1 * heatmap3 / (heatmap1 + heatmap3)
        elif mode == 'TEXTURE_ONLY':
            heatmap = heatmap3
        elif mode == 'COLOR_ONLY':
            heatmap = heatmap1
        else:
            raise Exception('wrong mode in calc_relevantMap')

        # 通过指数加权平均进行更新
        if np.max(self.heatmap) == 0:
            self.heatmap = heatmap
        else:
            self.heatmap = self.alpha * heatmap + (1 - self.alpha) * self.heatmap

        return self.heatmap, cv.cvtColor(template, cv.COLOR_HSV2BGR)

    def visualization(self, heatmap, template):
        plt.figure()

        # 绘制出剪裁区域
        ax = plt.subplot(self.nx+1, 1, 1)
        ax.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        ax.axis('off')

        # 绘制出每一个patch，并显示匹配程度
        for i in range(self.nx):
            for j in range(self.ny):
                ax = plt.subplot(self.nx+1, self.ny, self.ny*(i+1)+j+1)
                img = cv.cvtColor(self.patches[i, j], cv.COLOR_HSV2RGB)
                # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

                color = (0, 255, 0)
                if heatmap[i, j] < 0.4:
                    color = (0, 0, 255)
                elif heatmap[i, j] > 0.8:
                    color = (255, 0, 0)
                img = cv.putText(img, str(round(heatmap[i, j], 5)), (0, self.patch_height),
                                 cv.FONT_HERSHEY_PLAIN, self.patch_height / 60, color, thickness=1)
                ax.imshow(img)
                ax.axis('off')
        # plt.savefig("save.png", dpi=300)
        plt.show()
        # plt.savefig("save.png", dpi=300)


def main():
    # 参数设置
    img = cv.imread(r'D:\iinet\MoNuSeg_split\train\img\TCGA-KB-A93J-01A-01-TS1_2_1.jpg')
    patch_height = 32
    patch_width = 32
    y1 = 64 + 4
    x1 = 64 + 4
    y2 = 190
    x2 = 70
    alpha = 0.9
    start = time()
    # 一个demo
    # 获取一个对象，传入图片，尺寸
    relevantMap = RelevantMap(img, patch_height, patch_width, alpha)
    # 传入一个点，计算出heatmap，template是剪裁出的区域

    heatmap, template = relevantMap.calc_relevantMap(x1, y1, mode='COLOR_TEXTURE_HARMONIC')
    # heatmap, template = relevantMap.calc_relevantMap(x2, y2)
    end = time()
    print('using time', f"{int((end-start) * 1000)}ms")
    # 可视化
    relevantMap.visualization(heatmap, template)


if __name__ == '__main__':
    main()
