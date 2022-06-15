import numpy as np
from rlremedy.models.time_series.configuration import SPParams


class SeasonalProcess:

    def __init__(self, sp_params: SPParams):
        self.sp_params = sp_params


    def sample_paths(self):
        """Returns a sample of paths from the process."""

        length = np.pi * 2 * self.sp_params.cycles
        samples = np.reshape(np.sin(np.arange(0, length, length / self.sp_params.resolution)),[-1,1])

        return samples