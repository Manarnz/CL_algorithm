import numpy as np
import setting

class heat_kernel():
    def __init__(self, timestep, size):
        self.size = size
        self.timestep = timestep
        self.delta = setting.IMGSIZE/timestep

        self.data = np.zeros((size, size))
        center = self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                x = i - center
                y = j - center
                self.data[i, j] = np.exp(-(x**2 + y**2)/4)/(4*np.pi*self.delta**2)
        self.data /= np.sum(self.data)
