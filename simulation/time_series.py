import numpy as np
import matplotlib.pyplot as plt

class multi_asset_path_sim():

    def __init__(self,n_tickers=2,corr_matrix=None,vola=None,T=252,r = 0.001, S_0 = 100):
        '''

        :param n_tickers: Number of assets
        :param corr_matrix: Correlation matrix
        :param vola: Volatility (annual, 0.01=1%)
        :param T: Number of simulated days
        :param r: Risk-free rate (annual, 0.01=1%)
        :param S_0: Starting values of the assets
        '''
        self.n_tickers=n_tickers

        if corr_matrix is not None:
            self.corr_matrix=corr_matrix
            assert len(self.corr_matrix) != len(self.n_tickers)
        else:
            self.corr_matrix=np.identity(self.n_tickers)

        if vola is not None:
            self.vola=vola
            assert len(self.vola) == self.n_tickers
        else:
            self.vola=np.ones(self.n_tickers)



        self.T = int(T)

        self.simulated_paths = np.full((self.n_tickers, self.T),S_0)  # Stock price, first value is simulation input

        self.r = r
        self.dt = 1.0 / self.T  # Time increment (annualized)

        self.do_cholesky_dec()
    def do_cholesky_dec(self):
        # Perform Cholesky decomposition on coefficient matrix
        self.cholesky_matrix_dec  = np.linalg.cholesky(self.corr_matrix)



    def simulate_paths(self):
        for t in range(self.T):
            # Generate array of random standard normal draws
            random_array = np.random.standard_normal(self.n_tickers)

            # Multiply R (from factorization) with random_array to obtain correlated epsilons
            epsilon_array = np.inner(random_array, self.cholesky_matrix_dec)

            # Sample price path per stock
            for n in range(self.n_tickers):
                S = self.simulated_paths [n, t - 1]
                v = self.vola[n]
                epsilon = epsilon_array[n]

                # Generate new stock price
                self.simulated_paths[n, t] = S * np.exp((self.r - 0.5 * v ** 2) * self.dt + v * np.sqrt(self.dt) * epsilon)

    def plot_paths(self):
        # Plot simulated price paths

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        array_day_plot = [t for t in range(self.T)]

        for n in range(self.n_tickers):
            ax.plot(array_day_plot, self.simulated_paths[n], label=f'{n}')

        plt.grid()
        plt.xlabel('Day')
        plt.ylabel('Asset Price')
        plt.legend(loc='best')

        plt.show()
