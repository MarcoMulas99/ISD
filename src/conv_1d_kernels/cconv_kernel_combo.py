
class CConveKernelCombo():

    def __init__(self, value):
        self._kernel_seq = value

    @property
    def kernel_seq(self):
        return self._kernel_seq

    @kernel_seq.setter
    def kernel_seq(self, value):
        self._kernel_seq = value

    def kernel(self, x):
        xp = x.copy().ravel()

        for k in self.kernel_seq:
            xp = k.kernel(xp)

        return xp
