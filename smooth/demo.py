import cv2 as cv
from smoothLine import smooth_lines, visualization


# 传入一个(width, height)的二值图像
image = cv.imread('line1.png')
image = image[:, :, 0]
lines = smooth_lines(image)
visual = visualization(image, lines)
cv.imwrite('visual1.png', visual)


image = cv.imread('line2.png')
image = image[:, :, 0]
lines = smooth_lines(image)
visual = visualization(image, lines)
cv.imwrite('visual2.png', visual)

