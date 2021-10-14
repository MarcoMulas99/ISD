from copy import deepcopy

from conv_1d_kernels import CConveKernelCombo, CConvKernelTriangle, CConvKernelMovingAverage
import numpy as np
from utils import load_mnist_data, plot_ten_digits
import matplotlib.pyplot as plt

x, y = load_mnist_data()

conv1 = CConvKernelTriangle()
conv2 = CConvKernelMovingAverage()
conv1.kernel_mask()
conv2.kernel_mask()

seq = [conv1, conv2, conv1]

convCombo = CConveKernelCombo(seq)

images = np.zeros(shape=(10, x.shape[1]))


for i in np.unique(y):
    images[i] = x[y == i, :][0]

images_average = deepcopy(images)
images_triangle = deepcopy(images)
images_combo = deepcopy(images)

plot_ten_digits(images, save=True, name="no_filter.pdf")

for i, img in enumerate(images_average):
    images_average[i] = conv2.kernel(images_average[i])

plot_ten_digits(images_average, save=True, name="average_filter.pdf")

for i, img in enumerate(images_triangle):
    images_triangle[i] = conv1.kernel(images_triangle[i])

plot_ten_digits(images_triangle, save=True, name="triangle_filter.pdf")

for i, img in enumerate(images_combo):
    images_combo[i] = convCombo.kernel(images_combo[i])

plot_ten_digits(images_combo, save=True, name="combo_filter.pdf")

print(images[0])
print(images_average[0])
print(images_triangle[0])
print(images_combo[0])



#plt.savefig("no_filter.pdf")

