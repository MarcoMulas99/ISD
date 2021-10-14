from abc import ABC, abstractmethod
import numpy as np

class CConvKernel(ABC):

    def __init__(self):
        self._kernel_size = 3
        self._mask = None

    @property
    def kernel_size(self):
        return self._kernel_size
    @property
    def mask(self):
        return self._mask
    @kernel_size.setter
    def kernel_size(self, value):
        if(value % 2 == 0): raise Exception ("Kernel size must be an odd value")
        self.kernel_mask()

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("kernel_mask not implemented!")

    def kernel(self, x):
        xp = x.copy().ravel()

        unchanged = (self.kernel_size-1)//2
        offset = self.kernel_size//2

        for i, value in enumerate(x):
            if i < unchanged: pass
            elif i > x.shape[0]-2: pass
            else:
                count = 0
                for j in range(-offset, offset+1):
                    count += self.mask[offset + j] * x[i+j]
                xp[i] = count

        return xp
