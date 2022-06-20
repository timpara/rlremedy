from typing import Optional

import numpy as np
from datetime import datetime
from rlremedy.models.time_series.configuration import OUParams


class OuProcess:
    def __init__(self, OU_params: OUParams):
        """
        - Ou_params is an instance of OUParams dataclass.
        """
        self.OU_params = OU_params

    def get_W(self) -> np.ndarray:
        """
        Simulate a Brownian motion discretely samplet at unit time increments.
        Returns the cumulative sum
        """
        dW = self.get_dW()
        # cumulative sum and then make the first index 0.
        dW_cs = dW.cumsum()
        return np.insert(dW_cs, 0, 0)[:-1]
    def get_dW(self) -> np.ndarray:
        """
        Sample T times from a normal distribution,
        to simulate discrete increments (dW) of a Brownian Motion.
        Optional random_state to reproduce results.
        """
        np.random.seed(int(datetime.utcnow().timestamp()))
        return np.random.normal(0.0, 1.0, self.OU_params.sample_size)
    def sample_paths(self,
        X_0: Optional[float] = None,
    ) -> np.ndarray:
        """
        - T is the sample size.
        - X_0 the initial value for the process, if None, then X_0 is taken
            to be gamma (the asymptotic mean).
        Returns a 1D array.
        """
        t = np.arange(self.OU_params.sample_size, dtype=np.float128) # float to avoid np.exp overflow
        exp_alpha_t = np.exp(-self.OU_params.alpha * t)
        dW = self.get_dW()
        integral_W = self._get_integal_W(t, dW)
        _X_0 = self._select_X_0(X_0)
        return (
            _X_0 * exp_alpha_t
            + self.OU_params.gamma * (1 - exp_alpha_t)
            + self.OU_params.beta * exp_alpha_t * integral_W
        )


    def _select_X_0(self,X_0_in: Optional[float]) -> float:
        """Returns X_0 input if not none, else gamma (the long term mean)."""
        if X_0_in is not None:
            return X_0_in
        return self.OU_params.gamma


    def _get_integal_W(self,
                       t: np.ndarray,
                       dW: np.ndarray) -> np.ndarray:
        """Integral with respect to Brownian Motion (W), ∫...dW."""
        exp_alpha_s = np.exp(self.OU_params.alpha * t)
        integral_W = np.cumsum(exp_alpha_s * dW)
        return np.insert(integral_W, 0, 0)[:-1]
#----------------------------------------------------
if __name__=="__main__":
    OU_params = OUParams(alpha=0.6, gamma=30, beta=1, sample_size=1000)
    OU_proc = OuProcess(OU_params)
    OU_proc = OU_proc.sample_paths()

    # plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 7))

    title = "Ornstein-Uhlenbeck process, "
    title += fr"$\alpha={OU_params.alpha}$, $\gamma = {OU_params.gamma}$, $\beta = {OU_params.beta}$"
    plt.plot(OU_proc)
    plt.gca().set_title(title, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()