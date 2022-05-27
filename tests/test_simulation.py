from simulation.time_series import multi_asset_path_sim
import numpy as np
import pytest

#%%
def test_multi_asset_path_sim():
    '''
    Instantiates the multi_asset_path_sim
    :return:
    '''
    T=1e5
    sim_env = multi_asset_path_sim(T=T,vola=[0.02,0.01])
    # It will check your custom environment and output additional warnings if needed
            # Reconstruct coefficient matrix from factorization (only for validation)
    np.testing.assert_array_almost_equal(np.dot( np.linalg.cholesky(sim_env.corr_matrix),sim_env.cholesky_matrix_dec.T.conj() ),np.identity(sim_env.n_tickers))

    sim_env.simulate_paths()
    assert np.shape(sim_env.simulated_paths)==(2,T)