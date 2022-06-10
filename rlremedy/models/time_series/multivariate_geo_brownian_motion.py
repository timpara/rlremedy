import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.models.building import generate_mc_normal_draws
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device=device, enable=True)
"""Multivariate Geometric Brownian Motion."""
class MultivariateGeometricBrownianMotion():
  """Multivariate Geometric Brownian Motion.
  Represents a d-dimensional Ito process:
  ```None
    dX_i(t) = means_i * X_i(t) * dt + volatilities_i * X_i(t) * dW_i(t),
    1 <= i <= d
  ```
  where `W(t) = (W_1(t), .., W_d(t))` is a d-dimensional Brownian motion with
  a correlation matrix `corr_matrix`, `means` and `volatilities` are `Tensor`s
  that correspond to mean and volatility of a Geometric Brownian Motion `X_i`
  ## Example
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff
  corr_matrix = [[1, 0.1], [0.1, 1]]
  process = tff.models.MultivariateGeometricBrownianMotion(
      dim=2,
      means=1, volatilities=[0.1, 0.2],
      corr_matrix=corr_matrix,
      dtype=tf.float64)
  times = [0.1, 0.2, 1.0]
  initial_state=[1.0, 2.0]
  # Use SOBOL sequence to draw trajectories
  samples_sobol = process.sample_paths(
      times=times,
      initial_state=initial_state,
      random_type=tff.math.random.RandomType.SOBOL,
      num_samples=100000)
  # You can also supply the random normal draws directly to the sampler
  normal_draws = tf.random.stateless_normal(
      [100000, 3, 2], seed=[4, 2], dtype=tf.float64)
  samples_custom = process.sample_paths(
      times=times,
      initial_state=initial_state,
      normal_draws=normal_draws)
  ```
  """

  def __init__(self,
               dim,
               means=0.0,
               volatilities=1.0,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the Multivariate Geometric Brownian Motion.
    Args:
      dim: A Python scalar. The dimensionality of the process
      means:  A real `Tensor` of shape broadcastable to `[dim]`.
        Corresponds to the vector of means of the GBM components `X_i`.
      Default value: 0.0.
      volatilities: A `Tensor` of the same `dtype` as `means` and of shape
        broadcastable to `[dim]`. Corresponds to the volatilities of the GBM
        components `X_i`.
        Default value: 1.0.
      corr_matrix: An optional `Tensor` of the same `dtype` as `means` and of
        shape `[dim, dim]`. Correlation of the GBM components `W_i`.
        Default value: `None` which maps to a process with
        independent GBM components `X_i`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        'multivariate_geometric_brownian_motion'.
    Raises:
      ValueError: If `corr_matrix` is supplied and is not of shape `[dim, dim]`
    """
    self._name = name or "multivariate_geometric_brownian_motion"
    with tf.name_scope(self._name):
      self._means = tf.convert_to_tensor(means, dtype=dtype,
                                         name="means")
      self._dtype = self._means.dtype
      self._vols = tf.convert_to_tensor(volatilities, dtype=self._dtype,
                                        name="volatilities")
      self._dim = dim
      if corr_matrix is None:
        self._corr_matrix = None
      else:
        self._corr_matrix = tf.convert_to_tensor(corr_matrix, dtype=self._dtype,
                                                 name="corr_matrix")
        if self._corr_matrix.shape.as_list() != [dim, dim]:
          raise ValueError("`corr_matrix` must be of shape [{0}, {0}] but is "
                           "of shape {1}".format(
                               dim, self._corr_matrix.shape.as_list()))

  def dim(self):
    """The dimension of the process."""
    return self._dim

  def dtype(self):
    """The data type of process realizations."""
    return self._dtype

  def name(self):
    """The name to give to ops created by this class."""
    return self._name

  def drift_fn(self):
    """Python callable calculating instantaneous drift."""
    def _drift_fn(t, x):
      """Drift function of the GBM."""
      del t
      return self._means * x
    return _drift_fn

  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility."""
    def _vol_fn(t, x):
      """Volatility function of the GBM."""
      del t
      # Shape [num_samples, dim]
      vols = self._vols * x
      if self._corr_matrix is not None:
        vols = tf.expand_dims(vols, axis=-1)
        cholesky = tf.linalg.cholesky(self._corr_matrix)
        return vols * cholesky
      else:
        return tf.linalg.diag(vols)
    return _vol_fn

  def sample_paths(self,
                   times,
                   initial_state=None,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   skip=0,
                   normal_draws=None,
                   name=None):
    """Returns a sample of paths from the process.
    Args:
      times: Rank 1 `Tensor` of positive real values. The times at which the
        path points are to be evaluated.
      initial_state: A `Tensor` of the same `dtype` as `times` and of shape
        broadcastable with `[num_samples, dim]`. Represents the initial state of
        the Ito process.
      Default value: `None` which maps to a initial state of ones.
      num_samples: Positive scalar `int`. The number of paths to draw.
      random_type: Enum value of `RandomType`. The type of (quasi)-random
        number generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
        `HALTON_RANDOMIZED` the seed should be an Python integer. For
        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
        `Tensor` of shape `[2]`.
        Default value: `None` which means no seed is set.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
        Default value: 0.
      normal_draws: A `Tensor` of shape `[num_samples, num_time_points, dim]`
        and the same `dtype` as `times`. Represents random normal draws to
        compute increments `N(0, t_{n+1}) - N(0, t_n)`. When supplied,
        `num_samples` argument is ignored and the first dimensions of
        `normal_draws` is used instead. `num_time_points` should be equal to
        `tf.shape(times)[0]`.
        Default value: `None` which means that the draws are generated by the
        algorithm.
      name: Str. The name to give this op.
        Default value: `sample_paths`.
    Returns:
      A `Tensor`s of shape [num_samples, k, dim] where `k` is the size
      of the `times`.
    Raises:
      ValueError: If `normal_draws` is supplied and `dim` is mismatched.
    """
    name = name or (self._name + "_sample_path")
    with tf.name_scope(name):
      times = tf.convert_to_tensor(times, self._dtype, name="times")
      if normal_draws is not None:
        normal_draws = tf.convert_to_tensor(normal_draws, dtype=self._dtype,
                                            name="normal_draws")
      if initial_state is None:
        initial_state = tf.ones([num_samples, self._dim], dtype=self._dtype,
                                name="initial_state")
      else:
        initial_state = (
            tf.convert_to_tensor(initial_state, dtype=self._dtype,
                                 name="initial_state")
            + tf.zeros([num_samples, 1], dtype=self._dtype))
      # Shape [num_samples, 1, dim]
      initial_state = tf.expand_dims(initial_state, axis=1)
      num_requested_times = times.shape[0]
      return self._sample_paths(
          times=times, num_requested_times=num_requested_times,
          initial_state=initial_state,
          num_samples=num_samples, random_type=random_type,
          seed=seed, skip=skip,
          normal_draws=normal_draws)

  def _sample_paths(self,
                    times,
                    num_requested_times,
                    initial_state,
                    num_samples,
                    random_type,
                    seed,
                    skip,
                    normal_draws,):
    """Returns a sample of paths from the process."""
    if normal_draws is None:
      # Normal draws needed for sampling.
      # Shape [num_requested_times, num_samples, dim]
      normal_draws = generate_mc_normal_draws(
          num_normal_draws=self._dim, num_time_steps=num_requested_times,
          num_sample_paths=num_samples, random_type=random_type,
          seed=seed,
          dtype=self._dtype, skip=skip)
    else:
      # Shape [num_time_points, num_samples, dim]
      normal_draws = tf.transpose(normal_draws, [1, 0, 2])
      draws_dim = normal_draws.shape[2]
      if self._dim != draws_dim:
        raise ValueError(
            "`dim` should be equal to `normal_draws.shape[2]` but are "
            "{0} and {1} respectively".format(self._dim, draws_dim))
    times = tf.concat([[0], times], -1)
    # Time increments
    # Shape [num_requested_times, 1, 1]
    dt = tf.expand_dims(tf.expand_dims(times[1:] - times[:-1], axis=-1),
                        axis=-1)
    if self._corr_matrix is None:
      stochastic_increment = normal_draws
    else:
      cholesky = tf.linalg.cholesky(self._corr_matrix)
      stochastic_increment = tf.linalg.matvec(cholesky, normal_draws)

    # The logarithm of all the increments between the times.
    # Shape [num_requested_times, num_samples, dim]
    log_increments = ((self._means - self._vols**2 / 2) * dt
                      + tf.sqrt(dt) * self._vols
                      * stochastic_increment)

    # Since the implementation of tf.math.cumsum is single-threaded we
    # use lower-triangular matrix multiplication instead
    once = tf.ones([num_requested_times, num_requested_times],
                   dtype=self._dtype)
    lower_triangular = tf.linalg.band_part(once, -1, 0)
    cumsum = tf.linalg.matvec(lower_triangular,
                              tf.transpose(log_increments))
    cumsum = tf.transpose(cumsum, [1, 2, 0])
    samples = initial_state * tf.math.exp(cumsum)
    return samples

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
