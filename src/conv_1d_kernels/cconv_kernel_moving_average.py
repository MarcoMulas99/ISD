from .cconv_kernel import CConvKernel
import numpy as np

class CConvKernelMovingAverage(CConvKernel):

    def kernel_mask(self):
        self._mask = np.ones(self.kernel_size) / self.kernel_size