import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from time import time
from math import ceil
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops


class RelevantMap:
    def __init__(self, location, patch_size=32, alpha=0.8):
        """
        :param patch_size * patch_size: 要求是8, 16, 32
        :param alpha:
            指数加权平均的参数
            heatmap(1) = heatmap(1)
            heatmap(i) = (1 - alpha) * heatmap(i - 1) + alpha * heatmap(i)
        """
        if patch_size != 8 and patch_size != 16 and patch_size != 32:
            raise Exception('not support patch_size, 8, 16, 32 only')

        self.location = location
        self.patch_height = patch_size
        self.patch_width = patch_size

        self.nx = None
        self.ny = None
        self.width = None
        self.height = None
        self.channel = None

        self.patchesColorHist = None
        self.patchesTexture = None
        self.grayCoM_offset = 1
        self.img = None
        self.img_gray = None

        self.alpha = alpha
        self.heatmap = None

        # 从npy文件中读出img, img_gray, patchesColorHist, patchesTexture
        self.read()

    def read(self):
        location = f'{self.location[:-4]}.npy'
        file = np.load(location, allow_pickle=True).item()

        self.img = file['HSV']
        self.img_gray = file['GRAY']
        self.patchesColorHist = file[f'patchesColorHist_{self.patch_width}']
        self.patchesTexture = file[f'patchesTexture_{self.patch_width}']

        self.width, self.height, self.channel = self.img.shape

        # 计算出heatMap的尺寸，即应该有patch的个数
        self.nx = int(self.height / self.patch_height)
        self.ny = int(self.width / self.patch_width)

        self.heatmap = np.zeros((self.nx, self.ny), dtype=np.float32)


    def get_patches(self):
        """ 对原图进行划分，得到若干的patches """

        # 求解patches
        patches = np.zeros((self.nx, self.ny, self.patch_height, self.patch_width, self.channel), dtype=np.uint8)
        patches_gray = np.zeros((self.nx, self.ny, self.patch_height, self.patch_width), dtype=np.uint8)

        for i in range(self.ny):
            xt = self.patch_height * i
            xd = self.patch_height * (i + 1)
            for j in range(self.nx):
                patches[i, j] = self.img[xt:xd, self.patch_width * j:self.patch_width * (j + 1), :]
                patches_gray[i, j] = self.img_gray[xt:xd, self.patch_width * j:self.patch_width * (j + 1)]
        return patches, patches_gray

    def clip_patch(self, x, y, mode='HSV'):
        """ 识别用户点击的点，并从中采取出来一个邻域 """

        xd = int(self.patch_height / 2)
        yd = int(self.patch_width / 2)

        xt = max(x - xd, 0)
        xd = min(self.height, xt + self.patch_height)
        yl = max(y - yd, 0)
        yr = min(self.width, yl + self.patch_width)

        # 剪裁出点击邻域的一个patch并进行归一化到(patch_height, patch_width)
        res = None
        if mode == 'HSV':
            res = self.img[xt:xd, yl:yr, :]
        elif mode == 'GRAY':
            res = self.img_gray[xt:xd, yl:yr]
        res = cv.resize(res, dsize=(self.patch_width, self.patch_height), fx=1, fy=1, interpolation=cv.INTER_LINEAR)
        return res

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
        templateColorHist = np.empty((3, 256), dtype=np.float32)
        for c in range(self.channel):
            templateColorHist[c, :] = cv.calcHist([template[:, :, c]], [0], None, [256], [0.0, 255.0])[:, 0]

        # 1. 利用颜色分布直方图计算相似性，耗时2ms
        # heatmap1 = np.empty((self.nx, self.ny), dtype=np.float32)

        avg1 = np.average(self.patchesColorHist, axis=3)
        avg1 = np.reshape(avg1, (avg1.shape[0], avg1.shape[1], avg1.shape[2], 1))
        d1 = self.patchesColorHist - avg1
        avg2 = np.average(templateColorHist, axis=1)
        avg2 = np.reshape(avg2, (avg2.shape[0], 1))
        d2 = templateColorHist - avg2
        up = d1 * d2
        up = np.sum(up, axis=3)
        down = np.sum(np.square(d1), axis=3) * np.sum(np.square(d2), axis=1)
        down = np.sqrt(down)
        heatmap1 = np.sum(up / down, axis=2) / 3.0

        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         NCC = 0
        #         for c in range(self.channel):
        #             res = cv.compareHist(self.patchesColorHist[i, j, c, :], templateColorHist[c, :], method=cv.HISTCMP_CORREL)
        #             NCC += res
        #         NCC /= 3.0
        #         heatmap1[i, j] = NCC

        location = np.where(heatmap1 < 0)
        heatmap1[location] = 0

        # 2. 利用SSIM计算，耗时长，效果及其不好
        # heatmap2 = np.zeros((self.nx, self.ny), dtype=np.float32)
        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         heatmap2[i, j] = ssim(template, self.patches[i, j], multichannel=True)

        # 3. 计算灰度共生矩阵
        mat = graycomatrix(template_gray, [self.grayCoM_offset], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
        correlation = graycoprops(mat, 'contrast')[0]
        heatmap3 = np.linalg.norm(self.patchesTexture - correlation, axis=2)
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

        return self.heatmap, template

    def visualization(self, heatmap, template):
        patches, patches_gray = self.get_patches()
        template = cv.cvtColor(template, cv.COLOR_HSV2BGR)

        plt.figure()

        # 绘制出剪裁区域
        ax = plt.subplot(self.nx+1, 1, 1)
        ax.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        ax.axis('off')

        # 绘制出每一个patch，并显示匹配程度
        for i in range(self.nx):
            for j in range(self.ny):
                ax = plt.subplot(self.nx+1, self.ny, self.ny*(i+1)+j+1)
                img = cv.cvtColor(patches[i, j], cv.COLOR_HSV2RGB)
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

        plt.show()
        # plt.savefig("after3.pdf", dpi=300)


def main():
    # 参数设置

    y1 = 96 + 16
    x1 = 32 + 16
    y2 = 190
    x2 = 70

    location = './MoNuSeg_split/train/img/TCGA-18-5592-01Z-00-DX1_2_1.jpg'
    alpha = 0.9
    patch_size = 32

    # 一个demo
    # 获取一个对象，传入图片，尺寸
    start = time()
    relevantMap = RelevantMap(location, patch_size, alpha)
    end = time()
    print('create class using time', f"{int((end-start) * 1000)}ms")

    start = time()
    # 传入一个点，计算出heatmap，template是剪裁出的区域
    heatmap, template = relevantMap.calc_relevantMap(x1, y1, mode='COLOR_TEXTURE_HARMONIC')
    # heatmap, template = relevantMap.calc_relevantMap(x2, y2)
    end = time()
    print('calc using time', f"{int((end-start) * 1000)}ms")

    # 可视化
    relevantMap.visualization(heatmap, template)


if __name__ == '__main__':
    main()
