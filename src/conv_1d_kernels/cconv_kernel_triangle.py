from .cconv_kernel import CConvKernel
import numpy as np

class CConvKernelTriangle(CConvKernel):

    def kernel_mask(self):
        a = np.array(range(1, self.kernel_size+1))
        b = np.array(range(1, self.kernel_size))
        b = np.flip(b)

        ab = np.concatenate((a, b), axis=None)

        self._mask = ab/np.sum(ab)