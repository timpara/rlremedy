"""Tests for Geometric Brownian Motion."""

from absl.testing import parameterized
import numpy as np
from rlremedy.models.time_series import MultivariateGeometricBrownianMotion
import tensorflow.compat.v2 as tf
#import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


arrays_all_close = geometric_brownian_motion_test_utils.arrays_all_close
calculate_sample_paths_mean_and_variance = geometric_brownian_motion_test_utils.calculate_sample_paths_mean_and_variance

NUM_SAMPLES = 100000
NUM_STDERRS = 3.0  # Maximum size of the error in multiples of the standard
                   # error.


def _tolerance_by_dtype(dtype):
  """Returns the expected tolerance based on dtype."""
  return 1e-8 if dtype == np.float64 else 5e-3


@test_util.run_all_in_graph_and_eager_modes
class GeometricBrownianMotionTest(parameterized.TestCase, tf.test.TestCase):
  @parameterized.named_parameters(
    {
      'testcase_name': 'SinglePrecision',
      'dtype': np.float32,
    }, {
      'testcase_name': 'DoublePrecision',
      'dtype': np.float64,
    })
  def test_multivariate_drift_and_volatility(self, dtype):
    """Tests multivariate GBM drift and volatility functions."""
    means = [0.05, 0.02]
    volatilities = [0.1, 0.2]
    corr_matrix = [[1, 0.1], [0.1, 1]]
    process = MultivariateGeometricBrownianMotion(
      dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix,
      dtype=tf.float64)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    with self.subTest('Drift'):
      drift = drift_fn(0.2, state)
      expected_drift = np.array(means) * state
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest('Volatility'):
      vol = volatility_fn(0.2, state)
      expected_vol = np.expand_dims(
        np.array(volatilities) * state,
        axis=-1) * np.linalg.cholesky(corr_matrix)
      self.assertAllClose(vol, expected_vol, atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
    {
      'testcase_name': 'SinglePrecision',
      'dtype': np.float32,
    }, {
      'testcase_name': 'DoublePrecision',
      'dtype': np.float64,
    })
  def test_multivariate_drift_and_volatility_no_corr(self, dtype):
    """Tests multivariate GBM drift and volatility functions."""
    means = [0.05, 0.02]
    volatilities = [0.1, 0.2]
    corr_matrix = [[1, 0.0], [0.0, 1]]
    process = MultivariateGeometricBrownianMotion(
      dim=2, means=means, volatilities=volatilities,
      dtype=tf.float64)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    with self.subTest('Drift'):
      drift = drift_fn(0.2, state)
      expected_drift = np.array(means) * state
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest('Volatility'):
      vol = volatility_fn(0.2, state)
      expected_vol = np.expand_dims(
        np.array(volatilities) * state,
        axis=-1) * np.linalg.cholesky(corr_matrix)
      self.assertAllClose(vol, expected_vol, atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
    {
      'testcase_name': 'SinglePrecisionNoDraws',
      'supply_draws': False,
      'dtype': np.float32,
    }, {
      'testcase_name': 'DoublePrecisionNoDraws',
      'supply_draws': False,
      'dtype': np.float64,
    }, {
      'testcase_name': 'DoublePrecisionWithDraws',
      'supply_draws': True,
      'dtype': np.float64,
    })
  def test_univariate_sample_mean_and_variance_constant_parameters(
          self, supply_draws, dtype):
    """Tests the mean and vol of the univariate GBM sampled paths."""
    mu = 0.05
    sigma = 0.05
    times = np.array([0.1, 0.5, 1.0], dtype=dtype)
    initial_state = 2.0
    mean, var, se_mean, se_var = calculate_sample_paths_mean_and_variance(
      self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
    expected_mean = ((mu - sigma ** 2 / 2) * np.array(times)
                     + np.log(initial_state))
    expected_var = sigma ** 2 * times
    atol_mean = se_mean * NUM_STDERRS
    atol_var = se_var * NUM_STDERRS

    with self.subTest('Drift'):
      arrays_all_close(self, tf.squeeze(mean), expected_mean, atol_mean,
                       msg='comparing means')
    with self.subTest('Variance'):
      arrays_all_close(self, tf.squeeze(var), expected_var, atol_var,
                       msg='comparing variances')

