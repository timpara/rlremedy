# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometric Brownian Motion model."""

import tensorflow as tf


from utils.models import building
from utils.math import piecewise as pw
class GeometricBrownianMotion():
  """Geometric Brownian Motion.

  Represents the 1-dimensional Ito process:

  ```None
    dX(t) = means(t) * X(t) * dt + volatilities(t) * X(t) * dW(t),
  ```

  where `W(t)` is a 1D Brownian motion, `mean(t)` and `volatility(t)` are either
  constant `Tensor`s or piecewise constant functions of time.

  Supports batching which enables modelling multiple univariate geometric
  brownian motions (GBMs) efficiently. No guarantee is made about the
  relationships between the batched univariate GMBs. To control the correlation
  between multiple GBMs use `MultivariateGeometricBrownianMotion`.

  ## Example

  ```python
  import tensorflow as tf
  import tf_quant_finance as tff
  process = tff.models.GeometricBrownianMotion(0.05, 1.0, dtype=tf.float64)
  times = [0.1, 0.2, 1.0]
  # Use SOBOL sequence to draw trajectories
  samples_sobol = process.sample_paths(
      times=times,
      initial_state=1.5,
      random_type=tff.math.random.RandomType.SOBOL,
      num_samples=100000)

  # You can also supply the random normal draws directly to the sampler
  normal_draws = tf.random.stateless_normal(
      [100000, 3, 1], seed=[4, 2], dtype=tf.float64)
  samples_custom = process.sample_paths(
      times=times,
      initial_state=1.5,
      normal_draws=normal_draws)
  ```
  """

  def __init__(self,
               mean,
               volatility,
               dtype=None,
               name=None):
    """Initializes the Geometric Brownian Motion.

    Args:
      mean: A real `Tensor` broadcastable to `batch_shape + [1]` or an instance
        of left-continuous `PiecewiseConstantFunc` with `batch_shape + [1]`
        dimensions. Here `batch_shape` represents a batch of independent
        GBMs. Corresponds to the mean drift of the Ito process.
      volatility: A real `Tensor` broadcastable to `batch_shape + [1]` or an
        instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`
        and `batch_shape` as set by `mean`. Corresponds to the volatility of the
        process and should be positive.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred from
          `mean` is used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        'geometric_brownian_motion'.
    """
    self._name = name or 'geometric_brownian_motion'
    with tf.name_scope(self._name):
      self._mean, self._mean_is_constant = pw.convert_to_tensor_or_func(
          mean, dtype=dtype, name='mean')
      self._dtype = dtype or self._mean.dtype
      (
          self._volatility,
          self._volatility_is_constant
      ) = pw.convert_to_tensor_or_func(volatility, dtype=self._dtype,
                                       name='volatility')
      self._volatility_squared = self._volatility_squared_from_volatility(
          self._volatility,
          self._volatility_is_constant,
          dtype=self._dtype,
          name='volatility_squared')
      self._dim = 1

  def dim(self):
    """The dimension of the process."""
    return self._dim

  def dtype(self):
    """The data type of process realizations."""
    return self._dtype

  def name(self):
    """The name to give to ops created by this class."""
    return self._name

  def drift_is_constant(self):
    """Returns True if the drift of the process is a constant."""
    return self._mean_is_constant

  def volatility_is_constant(self):
    """Returns True is the volatility of the process is a constant."""
    return self._volatility_is_constant

  def drift_fn(self):
    """Python callable calculating instantaneous drift."""
    def _constant_fn(t, x):
      """Drift function of the GBM with constant mean."""
      del t
      return self._mean * x

    def _piecewise_fn(t, x):
      """Drift function of the GBM with piecewise constant mean."""
      return self._mean(t) * x

    return _constant_fn if self.drift_is_constant() else _piecewise_fn

  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility."""
    def _constant_fn(t, x):
      """Volatility function of the GBM with constant volatility."""
      del t
      vol = self._volatility * tf.expand_dims(x, -1)
      return vol

    def _piecewise_fn(t, x):
      """Volatility function of the GBM with piecewise constant volatility."""
      vol = self._volatility(t) * tf.expand_dims(x, -1)
      return vol

    return _constant_fn if self.volatility_is_constant() else _piecewise_fn

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

    If `mean` and `volatility` were specified with batch dimensions the sample
    paths will be generated for all batch dimensions for the specified `times`
    using a single set of random draws.

    Args:
      times: A `Tensor` of positive real values of a shape `[T, k]`, where
        `T` is either empty or a shape which is broadcastable to `batch_shape`
        (as defined by the shape of `mean` or `volatility` which were set when
        this instance of GeometricBrownianMotion was initialised) and `k` is the
        number of time points. The times at which the path points are to be
        evaluated.
      initial_state: A `Tensor` of the same `dtype` as `times` and of shape
        broadcastable to `[batch_shape, num_samples]`. Represents the initial
        state of the Ito process.
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
      normal_draws: A `Tensor` of shape `[num_samples, num_time_points, 1]`
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
      A `Tensor`s of shape [batch_shape, num_samples, k, 1] where `k` is the
      the number of `time points`.

    Raises:
      ValueError: If `normal_draws` is supplied and does not have shape
      broadcastable to `[num_samples, num_time_points, 1]`.
    """
    name = name or (self._name + '_sample_path')

    with tf.name_scope(name):
      times = tf.convert_to_tensor(times, self._dtype)
      if normal_draws is not None:
        normal_draws = tf.convert_to_tensor(normal_draws, dtype=self._dtype,
                                            name='normal_draws')
      initial_state = building.convert_to_tensor_with_default(
          initial_state,
          tf.ones([1], dtype=self._dtype),
          dtype=self._dtype,
          name='initial_state')

      num_requested_times = times.shape[-1]
      return self._sample_paths(
          times=times,
          num_requested_times=num_requested_times,
          initial_state=initial_state,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
          skip=skip,
          normal_draws=normal_draws)

  def _integrate_parameter(self, x, x_is_constant, t0, t1, name=None):
    """Returns the integral of x(t).dt over the interval [t0, t1].

    Args:
      x: Scalar real `Tensor` of shape [`batch_shape`] or an instance of a
        left-continuous `PiecewiseConstantFunc`. The function to be integrated.
      x_is_constant: 'bool' which is True if x is a Scalar real `Tensor`.
      t0: A `Tensor` which is broadcastable to [`batch_shape`, `k`], where `k`
        is the number of intervals to evaluate the integral over. The start
        times of the `k` intervals.
      t1: A `Tensor` which is broadcastable to [`batch_shape`, `k`], where `k`
        is the number of intervals to evaluate the integral over. The end
        times of the `k` intervals.
      name: Str. The name to give this op.

    Returns:
      A `Tensor` of shape [`batch_shape`, `k`] with the integrals of x over the
      intervals [`t0`, `t1`].
    """
    return x * (t1 - t0) if x_is_constant else x.integrate(t0, t1, name)

  def _sample_paths(self,
                    times,
                    num_requested_times,
                    initial_state,
                    num_samples,
                    random_type,
                    seed,
                    skip,
                    normal_draws):
    """Returns a sample of paths from the process."""
    if normal_draws is None:
      # Normal draws needed for sampling
      normal_draws = building.generate_mc_normal_draws(
          num_normal_draws=1, num_time_steps=num_requested_times,
          num_sample_paths=num_samples, random_type=random_type,
          seed=seed,
          dtype=self._dtype, skip=skip)
    else:
      # Shape [num_time_points, num_samples, dim]
      normal_draws = tf.transpose(normal_draws, [1, 0, 2])
      num_samples = tf.shape(normal_draws)[1]
      draws_dim = normal_draws.shape[2]
      if draws_dim != 1:
        raise ValueError(
            '`dim` should be equal to `1` but is {0}'.format(draws_dim))
    # Create a set of zeros that is the right shape to add a '0' as the first
    # element for each series of times.
    zeros = tf.zeros(tf.concat([times.shape[:-1], [1]], 0), dtype=self._dtype)
    times = tf.concat([zeros, times], -1)
    mean_integral = self._integrate_parameter(
        self._mean, self._mean_is_constant, times[..., :-1], times[..., 1:])
    # mean_integral has shape [batch_shape, k-1], where self._mean has shape
    # [batch_shape, 1] and times has shape [k].
    mean_integral = tf.expand_dims(mean_integral, -2)
    volatility_sq_integral = self._integrate_parameter(
        self._volatility_squared, self._volatility_is_constant,
        times[..., :-1], times[..., 1:])
    volatility_sq_integral = tf.expand_dims(volatility_sq_integral, -2)
    # Giving mean_integral and volatility_sq_integral
    # shape = `batch_shape + [1, k-1]`,
    # where self._mean has shape `batch_shape + [1]` and times has shape `[k]`.

    # The logarithm of all the increments between the times.
    log_increments = ((mean_integral - volatility_sq_integral / 2)
                      # Ensure tf.math.sqrt is differentiable at 0.0
                      + _sqrt_no_nan(volatility_sq_integral)
                      * tf.transpose(tf.squeeze(normal_draws, -1)))
    # Since the implementation of tf.math.cumsum is single-threaded we
    # use lower-triangular matrix multiplication instead
    once = tf.ones([num_requested_times, num_requested_times],
                   dtype=self._dtype)
    lower_triangular = tf.linalg.band_part(once, -1, 0)
    cumsum = tf.linalg.matvec(lower_triangular,
                              log_increments)

    samples = tf.expand_dims(initial_state, [-1]) * tf.math.exp(cumsum)
    return tf.expand_dims(samples, -1)

  def _volatility_squared_from_volatility(
      self, volatility, volatility_is_constant, dtype=None, name=None):
    """Returns volatility squared as either a `PiecewiseConstantFunc` or a `Tensor`.

    Args:
      volatility: Either a 'Tensor' or 'PiecewiseConstantFunc'.
      volatility_is_constant: `bool` which is True if volatility is of type
        `Tensor`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        '_volatility_squared'.
    """
    name = name or (self._name + '_volatility_squared')
    if volatility_is_constant:
      return volatility ** 2
    else:
      return pw.PiecewiseConstantFunc(
          volatility.jump_locations(), volatility.values()**2,
          dtype=dtype, name=name)



def _backward_pde_coeffs(drift_fn, volatility_fn, discounting):
  """Returns coeffs of the backward PDE."""
  def second_order_coeff_fn(t, coord_grid):
    volatility = volatility_fn(t, _coord_grid_to_mesh_grid(coord_grid))
    volatility_times_volatility_t = tf.linalg.matmul(
        volatility, volatility, transpose_b=True)

    # We currently have [dim, dim] as innermost dimensions, but the returned
    # tensor must have [dim, dim] as outermost dimensions.
    rank = len(volatility.shape.as_list())
    perm = [rank - 2, rank - 1] + list(range(rank - 2))
    volatility_times_volatility_t = tf.transpose(
        volatility_times_volatility_t, perm)
    return volatility_times_volatility_t / 2

  def first_order_coeff_fn(t, coord_grid):
    mean = drift_fn(t, _coord_grid_to_mesh_grid(coord_grid))

    # We currently have [dim] as innermost dimension, but the returned
    # tensor must have [dim] as outermost dimension.
    rank = len(mean.shape.as_list())
    perm = [rank - 1] + list(range(rank - 1))
    mean = tf.transpose(mean, perm)
    return mean

  def zeroth_order_coeff_fn(t, coord_grid):
    if not discounting:
      return None
    return -discounting(t, _coord_grid_to_mesh_grid(coord_grid))

  return second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn


def _coord_grid_to_mesh_grid(coord_grid):
  if len(coord_grid) == 1:
    return tf.expand_dims(coord_grid[0], -1)
  return tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)


@tf.custom_gradient
def _sqrt_no_nan(x):
  """Returns square root with a gradient at 0 being 0."""
  root = tf.math.sqrt(x)
  def grad(upstream):
    return tf.math.divide_no_nan(upstream, root) / 2
  return root, grad
