import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data, img_as_ubyte, img_as_float, measure, img_as_bool
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import random_walker
# %matplotlib inline
import skimage.segmentation
from skimage.measure import label, regionprops
from skimage import morphology
from skimage import io
# from skimage.data import binary_blobs


# img = data.astronaut()
# img = rgb2gray(img)
#
# s = np.linspace(0, 2*np.pi, 400)
# x = 220 + 100*np.cos(s)
# y = 100 + 100*np.sin(s)
# init = np.array([x, y]).T
#
# snake = active_contour(gaussian(img, 3),
#                        init, alpha=0.015, beta=10, gamma=0.001)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.show()

# img = io.imread('D:\MyProject\pythonProject\BayesianInference\Img_As.jpg')
# 下面这个地方更改处理的文件名
strr = 'Images\Img_Cd.jpg'
strr1 = strr + '1.jpg'
img = img_as_float(io.imread(strr))

# idx = np.argwhere(img>200)
h, w = img.shape

# plt.figure(1)
# plt.imshow(img)
# 随机游走切分
makers = np.zeros_like(img)

makers[img > 0.9] = 1
makers[img < 0.2] = 2

SegResult = random_walker(img, makers, beta=10, mode='bf')


# plt.figure(2)
# plt.imshow(SegRes
# plt.imshow(SegResult)

# 区域处理，去掉最小区域
label_img = label(SegResult)
plt.figure(10)
plt.imshow(SegResult)
props = regionprops(label_img)

prop_rem = morphology.remove_small_objects(label_img, 100)

# plt.figure(3)
# plt.imshow(prop_rem)
# for reg_tmp in reversed(props):
#     if reg_tmp.area <= 100:
#         props.remove(reg_tmp)
#     else:
#         plt.figure(i)
#         plt.imshow(reg_tmp.image)
#         i = i + 1
prop_bw = prop_rem
prop_bw[prop_bw == 1] = 0
prop_bw[prop_bw >= 1] = 1

# plt.figure(4)
# plt.imshow(prop_bw)
sk = morphology.skeletonize(prop_bw)
# plt.figure(5)
# plt.imshow(sk)
# plt.figure(6)
# plt.imshow(img*0.8+np.double(sk)*0.2)

# 显示结果
# 下面这个地方是显示凸显的属性
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), dpi=200)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 15), dpi=250)
fig.suptitle(strr)

ax1.imshow(img)
ax1.axis('off')
ax1.set_title('original', fontsize=12)

ax2.imshow(img*0.8+np.double(sk)*0.2)
ax2.axis('off')
ax2.set_title('move', fontsize=12)

# 下面这个地方是保存文件的位置和属性
fig.savefig(strr1, dpi=350)
# 查看直方
# plt.hist(img.flat, bins=100)

# plt.imshow(img)
# plt.imshow(data)
# plt.imshow(b)
# plt.show()
