# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Indexed entropy model for continuous random variables."""

import tensorflow as tf
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import round_ops
import pdb

__all__ = [
    "ContinuousIndexedEntropyModel",
    "LocationScaleIndexedEntropyModel",
]


class ContinuousIndexedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Indexed entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor (a shared "prior" between sender and receiver). It also pre-computes
  integer probability tables, which can then be used to compress and decompress
  bottleneck tensors reliably across different platforms.

  A typical workflow looks like this:

  - Train a model using an instance of this entropy model as a bottleneck,
    passing the bottleneck tensor through it. With `training=True`, the model
    computes a differentiable upper bound on the number of bits needed to
    compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  conditionally independent given some other random variable, possibly depending
  on data. All dependencies must be represented by the `indexes` tensor. For
  each bottleneck tensor element, it selects the appropriate scalar
  distribution.

  The `indexes` tensor must contain only integer values in a pre-specified range
  (but may have floating-point type for purposes of backpropagation). To make
  the distribution conditional on `n`-dimensional indexes, `index_ranges` must
  be specified as an iterable of `n` integers. `indexes` must have the same
  shape as the bottleneck tensor with an additional channel dimension of length
  `n`. The position of the channel dimension is given by `channel_axis`. The
  index values in the `k`th channel must be in the range `[0, index_ranges[k])`.
  If `index_ranges` has only one element (i.e. `n == 1`), `channel_axis` may be
  `None`. In that case, the additional channel dimension is omitted, and the
  `indexes` tensor must have the same shape as the bottleneck tensor.

  The implied distribution for the bottleneck tensor is determined as:
  ```
  prior_fn(**{k: f(indexes) for k, f in parameter_fns.items()})
  ```

  A more detailed description (and motivation) of this indexing scheme can be
  found in the following paper. Please cite the paper when using this code for
  derivative work.

  > "Integer Networks for Data Compression with Latent-Variable Models"<br />
  > J. Ballé, N. Johnston, D. Minnen<br />
  > https://openreview.net/forum?id=S1zz2i0cY7

  Examples:

  To make a parameterized zero-mean normal distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyNormal,
      index_ranges=(64,),
      parameter_fns=dict(
          loc=lambda _: 0.,
          scale=lambda i: tf.exp(i / 8 - 5),
      ),
      coding_rank=1,
      channel_axis=None,
  )
  ```
  Then, each element of `indexes` in the range `[0, 64)` would indicate that the
  corresponding element in `bottleneck` is normally distributed with zero mean
  and a standard deviation between `exp(-5)` and `exp(2.875)`, inclusive.

  To make a parameterized logistic mixture distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyLogisticMixture,
      index_ranges=(10, 10, 5),
      parameter_fns=dict(
          loc=lambda i: i[..., 0:2] - 5,
          scale=lambda _: 1,
          weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
      ),
      coding_rank=1,
      channel_axis=-1,
  )
  ```
  Then, the last dimension of `indexes` would consist of triples of elements in
  the ranges `[0, 10)`, `[0, 10)`, and `[0, 5)`, respectively. Each triple
  would indicate that the element in `bottleneck` corresponding to the other
  dimensions is distributed with a mixture of two logistic distributions, where
  the components each have one of 10 location parameters between `-5` and `+4`,
  inclusive, unit scale parameters, and one of five different mixture
  weightings.
  """

  def __init__(self,
               prior_fn,
               index_ranges,
               parameter_fns,
               coding_rank,
               channel_axis=-1,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               bottleneck_dtype=None,
               prior_dtype=tf.float32,
               decode_sanity_check=True,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      prior_fn: A callable returning a `tfp.distributions.Distribution` object,
        typically a `Distribution` class or factory function. This is a density
        model fitting the marginal distribution of the bottleneck data with
        additive uniform noise, which is shared a priori between the sender and
        the receiver. For best results, the distributions should be flexible
        enough to have a unit-width uniform distribution as a special case,
        since this is the marginal distribution for bottleneck dimensions that
        are constant. The callable will receive keyword arguments as determined
        by `parameter_fns`.
      index_ranges: Iterable of integers. `indexes` must have the same shape as
        the bottleneck tensor, with an additional dimension at position
        `channel_axis`. The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.
      parameter_fns: Dict of strings to callables. Functions mapping `indexes`
        to each distribution parameter. For each item, `indexes` is passed to
        the callable, and the string key and return value make up one keyword
        argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
      channel_axis: Integer or `None`. Determines the position of the channel
        axis in `indexes`. Defaults to the last dimension. If set to `None`,
        the index tensor is expected to have the same shape as the bottleneck
        tensor (only allowed when `index_ranges` has length 1).
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
      prior_dtype: `tf.dtypes.DType`. Data type of prior and probability
        computations. Defaults to `tf.float32`.
      decode_sanity_check: Boolean. If `True`, an raises an error if the binary
        strings passed into `decompress` are not completely decoded.
      laplace_tail_mass: Float, or a float-valued tf.Tensor. If positive,
        will augment the prior with a `NoisyLaplace` mixture component for
        training stability. (experimental)
    """
    if not callable(prior_fn):
      raise TypeError("`prior_fn` must be a class or factory function.")
    for name, fn in parameter_fns.items():
      if not isinstance(name, str):
        raise TypeError("`parameter_fns` must have string keys.")
      if not callable(fn):
        raise TypeError(f"`parameter_fns['{name}']` must be callable.")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        bottleneck_dtype=bottleneck_dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    self._index_ranges = tuple(int(r) for r in index_ranges)
    if not self.index_ranges:
      raise ValueError("`index_ranges` must have at least one element.")
    self._channel_axis = None if channel_axis is None else int(channel_axis)
    if self.channel_axis is None and len(self.index_ranges) > 1:
      raise ValueError(
          "`channel_axis` can't be `None` for `len(index_ranges) > 1`.")
    self._prior_fn = prior_fn
    self._parameter_fns = dict(parameter_fns)
    self._prior_dtype = tf.as_dtype(prior_dtype)
    self.decode_sanity_check = decode_sanity_check

    with self.name_scope:
      if self.compression:
        if self.channel_axis is None:
          index_range, = index_ranges
          indexes = tf.range(index_range, dtype=tf.int32)
        else:
          indexes = [tf.range(r, dtype=tf.int32) for r in index_ranges]
          indexes = tf.meshgrid(*indexes, indexing="ij")
          indexes = tf.stack(indexes, axis=self.channel_axis)
        self._prior = self._make_prior(indexes)
        cdf, cdf_offset = self._build_tables(self.prior, range_coder_precision)
        self._init_compression(cdf, cdf_offset, None)

  @property
  def index_ranges(self):
    """Upper bound(s) on values allowed in `indexes` tensor."""
    return self._index_ranges

  @property
  def parameter_fns(self):
    """Functions mapping `indexes` to each distribution parameter."""
    return self._parameter_fns

  @property
  def prior_dtype(self):
    """Data type of `prior`."""
    return self._prior_dtype

  @property
  def prior_fn(self):
    """Class or factory function returning a `Distribution` object."""
    return self._prior_fn

  @property
  def channel_axis(self):
    """Position of channel axis in `indexes` tensor."""
    return self._channel_axis

  def _make_prior(self, indexes):
    indexes = tf.cast(indexes, self.prior_dtype)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    prior = self.prior_fn(**parameters)
    assert prior.dtype == self.prior_dtype
    if prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")
    return prior

  def _normalize_indexes(self, indexes):
    indexes = math_ops.lower_bound(indexes, 0)
    if self.channel_axis is None:
      index_range, = self.index_ranges
      bounds = index_range - 1
    else:
      axes = [1] * indexes.shape.rank
      axes[self.channel_axis] = len(self.index_ranges)
      bounds = tf.reshape([s - 1 for s in self.index_ranges], axes)
    return math_ops.upper_bound(indexes, tf.cast(bounds, indexes.dtype))

  def _flatten_indexes(self, indexes):
    indexes = tf.cast(indexes, tf.int32)
    if self.channel_axis is None:
      return indexes
    else:
      strides = tf.math.cumprod(self.index_ranges, exclusive=True, reverse=True)
      return tf.linalg.tensordot(indexes, strides, [[self.channel_axis], [0]])

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, indexes, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    import pdb
    #pdb.set_trace()
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    print("getting index")
    indexes = self._normalize_indexes(indexes)
    if training:
      def log_prob_fn(bottleneck_perturbed, indexes):
        # When using expected_grads=True, we will use a tf.custom_gradient on
        # this function. In this case, all non-Variable tensors that determine
        # the result of this function need to be declared explicitly, i.e we
        # need `indexes` to be a declared argument and `prior` instantiated
        # here. If we would instantiate it outside this function declaration and
        # reference here via a closure, we would get a `None` gradient for
        # `indexes`.
        prior = self._make_prior(indexes)
        return self._log_prob(prior, bottleneck_perturbed)
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, indexes, expected_grads=self.expected_grads)
    else:
      prior = self._make_prior(indexes)
      bottleneck_perturbed = self.quantize(bottleneck)
      log_probs = self._log_prob(prior, bottleneck_perturbed)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return round_ops.round_st(bottleneck)

  @tf.Module.with_name_scope
  def compress(self, bottleneck, indexes):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    all_but_last_n_elems = lambda t, n: t[:-n] if n else t
    batch_shape = all_but_last_n_elems(tf.shape(flat_indexes), self.coding_rank)
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols -= tf.gather(self.cdf_offset, flat_indexes)
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    handle = gen_ops.entropy_encode_index(handle, flat_indexes, symbols)
    return gen_ops.entropy_encode_finalize(handle)

  @tf.Module.with_name_scope
  def decompress(self, strings, indexes):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      indexes: `tf.Tensor` specifying the scalar distribution for each output
        element. See class docstring for examples.

    Returns:
      A `tf.Tensor` of the same shape as `indexes` (without the optional channel
      dimension).
    """
    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    last_n_elems = lambda t, n: t[-n:] if n else t[:0]
    decode_shape = last_n_elems(tf.shape(flat_indexes), self.coding_rank)
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    handle, symbols = gen_ops.entropy_decode_index(
        handle, flat_indexes, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    if self.decode_sanity_check:
      with tf.control_dependencies([tf.debugging.assert_equal(
          sanity, True, message="Sanity check failed.")]):
        symbols = tf.identity(symbols)
    symbols += tf.gather(self.cdf_offset, flat_indexes)
    return tf.cast(symbols, self.bottleneck_dtype)

  def get_config(self):
    """Returns the configuration of the entropy model."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")

  @classmethod
  def from_config(cls, config):
    """Instantiates an entropy model from a configuration dictionary."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")


class LocationScaleIndexedEntropyModel(ContinuousIndexedEntropyModel):
  """Indexed entropy model for location-scale family of random variables.

  This class is a common special case of `ContinuousIndexedEntropyModel`. The
  specified distribution is parameterized with `num_scales` values of scale
  parameters. An element-wise location parameter is handled by shifting the
  distributions to zero.

  This method is illustrated in Figure 10 of:
  > "Nonlinear Transform Coding"<br />
  > J. Ballé, P.A. Chou, D. Minnen, S. Singh, N. Johnston, E. Agustsson,
  > S.J. Hwang, G. Toderici<br />
  > https://doi.org/10.1109/JSTSP.2020.3034501

  Note: this only works for shift-invariant `tfpd.Distribution` objects, where
  the `loc` parameter really denotes a translation (i.e., not for the log-normal
  distribution).
  """

  def __init__(self,
               prior_fn,
               num_scales,
               scale_fn,
               coding_rank,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               bottleneck_dtype=None,
               prior_dtype=tf.float32,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      prior_fn: A callable returning a `tfp.distributions.Distribution` object,
        typically a `Distribution` class or factory function. This is a density
        model fitting the marginal distribution of the bottleneck data with
        additive uniform noise, which is shared a priori between the sender and
        the receiver. For best results, the distributions should be flexible
        enough to have a unit-width uniform distribution as a special case,
        since this is the marginal distribution for bottleneck dimensions that
        are constant. The callable will receive keyword arguments as determined
        by `parameter_fns`.
      num_scales: Integer. Values in `indexes` must be in the range
        `[0, num_scales)`.
      scale_fn: Callable. `indexes` is passed to the callable, and the return
        value is given as `scale` keyword argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
      prior_dtype: `tf.dtypes.DType`. Data type of prior and probability
        computations. Defaults to `tf.float32`.
      laplace_tail_mass: Float, or a float-valued tf.Tensor. If positive,
        will augment the prior with a `NoisyLaplace` mixture component for
        training stability. (experimental)
    """
    num_scales = int(num_scales)
    super().__init__(
        prior_fn=prior_fn,
        index_ranges=(num_scales,),
        parameter_fns=dict(
            loc=lambda _: 0.,
            scale=scale_fn,
        ),
        coding_rank=coding_rank,
        channel_axis=None,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        bottleneck_dtype=bottleneck_dtype,
        prior_dtype=prior_dtype,
        laplace_tail_mass=laplace_tail_mass,
    )

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, scale_indexes, loc=None, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    if loc is None:
      return super().__call__(bottleneck, scale_indexes, training=training)
    else:
      bottleneck, bits = super().__call__(
          bottleneck - loc, scale_indexes, training=training)
      return bottleneck + loc, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, loc=None):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo the location parameters of the prior distribution given in `loc`.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return round_ops.round_st(bottleneck, loc)

  @tf.Module.with_name_scope
  def compress(self, bottleneck, scale_indexes, loc=None):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    if loc is not None:
      bottleneck -= loc
    return super().compress(bottleneck, scale_indexes)

  @tf.Module.with_name_scope
  def decompress(self, strings, scale_indexes, loc=None):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each output
        element.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        output elements is assumed to be zero. Otherwise, specifies the location
        parameter for each output element. Must have the same shape as
        `scale_indexes`.

    Returns:
      A `tf.Tensor` of the same shape as `scale_indexes`.
    """
    values = super().decompress(strings, scale_indexes)
    if loc is not None:
      values += loc
    return values


class ContinuousIndexedEntropyModelwithSoftQuant(continuous_base.ContinuousEntropyModelBase):
  """Indexed entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor (a shared "prior" between sender and receiver). It also pre-computes
  integer probability tables, which can then be used to compress and decompress
  bottleneck tensors reliably across different platforms.

  A typical workflow looks like this:

  - Train a model using an instance of this entropy model as a bottleneck,
    passing the bottleneck tensor through it. With `training=True`, the model
    computes a differentiable upper bound on the number of bits needed to
    compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  conditionally independent given some other random variable, possibly depending
  on data. All dependencies must be represented by the `indexes` tensor. For
  each bottleneck tensor element, it selects the appropriate scalar
  distribution.

  The `indexes` tensor must contain only integer values in a pre-specified range
  (but may have floating-point type for purposes of backpropagation). To make
  the distribution conditional on `n`-dimensional indexes, `index_ranges` must
  be specified as an iterable of `n` integers. `indexes` must have the same
  shape as the bottleneck tensor with an additional channel dimension of length
  `n`. The position of the channel dimension is given by `channel_axis`. The
  index values in the `k`th channel must be in the range `[0, index_ranges[k])`.
  If `index_ranges` has only one element (i.e. `n == 1`), `channel_axis` may be
  `None`. In that case, the additional channel dimension is omitted, and the
  `indexes` tensor must have the same shape as the bottleneck tensor.

  The implied distribution for the bottleneck tensor is determined as:
  ```
  prior_fn(**{k: f(indexes) for k, f in parameter_fns.items()})
  ```

  A more detailed description (and motivation) of this indexing scheme can be
  found in the following paper. Please cite the paper when using this code for
  derivative work.

  > "Integer Networks for Data Compression with Latent-Variable Models"<br />
  > J. Ballé, N. Johnston, D. Minnen<br />
  > https://openreview.net/forum?id=S1zz2i0cY7

  Examples:

  To make a parameterized zero-mean normal distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyNormal,
      index_ranges=(64,),
      parameter_fns=dict(
          loc=lambda _: 0.,
          scale=lambda i: tf.exp(i / 8 - 5),
      ),
      coding_rank=1,
      channel_axis=None,
  )
  ```
  Then, each element of `indexes` in the range `[0, 64)` would indicate that the
  corresponding element in `bottleneck` is normally distributed with zero mean
  and a standard deviation between `exp(-5)` and `exp(2.875)`, inclusive.

  To make a parameterized logistic mixture distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyLogisticMixture,
      index_ranges=(10, 10, 5),
      parameter_fns=dict(
          loc=lambda i: i[..., 0:2] - 5,
          scale=lambda _: 1,
          weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
      ),
      coding_rank=1,
      channel_axis=-1,
  )
  ```
  Then, the last dimension of `indexes` would consist of triples of elements in
  the ranges `[0, 10)`, `[0, 10)`, and `[0, 5)`, respectively. Each triple
  would indicate that the element in `bottleneck` corresponding to the other
  dimensions is distributed with a mixture of two logistic distributions, where
  the components each have one of 10 location parameters between `-5` and `+4`,
  inclusive, unit scale parameters, and one of five different mixture
  weightings.
  """

  def __init__(self,
               prior_fn,
               index_ranges,
               parameter_fns,
               coding_rank,
               channel_axis=-1,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               bottleneck_dtype=None,
               prior_dtype=tf.float32,
               decode_sanity_check=True,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      prior_fn: A callable returning a `tfp.distributions.Distribution` object,
        typically a `Distribution` class or factory function. This is a density
        model fitting the marginal distribution of the bottleneck data with
        additive uniform noise, which is shared a priori between the sender and
        the receiver. For best results, the distributions should be flexible
        enough to have a unit-width uniform distribution as a special case,
        since this is the marginal distribution for bottleneck dimensions that
        are constant. The callable will receive keyword arguments as determined
        by `parameter_fns`.
      index_ranges: Iterable of integers. `indexes` must have the same shape as
        the bottleneck tensor, with an additional dimension at position
        `channel_axis`. The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.
      parameter_fns: Dict of strings to callables. Functions mapping `indexes`
        to each distribution parameter. For each item, `indexes` is passed to
        the callable, and the string key and return value make up one keyword
        argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
      channel_axis: Integer or `None`. Determines the position of the channel
        axis in `indexes`. Defaults to the last dimension. If set to `None`,
        the index tensor is expected to have the same shape as the bottleneck
        tensor (only allowed when `index_ranges` has length 1).
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
      prior_dtype: `tf.dtypes.DType`. Data type of prior and probability
        computations. Defaults to `tf.float32`.
      decode_sanity_check: Boolean. If `True`, an raises an error if the binary
        strings passed into `decompress` are not completely decoded.
      laplace_tail_mass: Float, or a float-valued tf.Tensor. If positive,
        will augment the prior with a `NoisyLaplace` mixture component for
        training stability. (experimental)
    """
    if not callable(prior_fn):
      raise TypeError("`prior_fn` must be a class or factory function.")
    for name, fn in parameter_fns.items():
      if not isinstance(name, str):
        raise TypeError("`parameter_fns` must have string keys.")
      if not callable(fn):
        raise TypeError(f"`parameter_fns['{name}']` must be callable.")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        bottleneck_dtype=bottleneck_dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    self._index_ranges = tuple(int(r) for r in index_ranges)
    if not self.index_ranges:
      raise ValueError("`index_ranges` must have at least one element.")
    self._channel_axis = None if channel_axis is None else int(channel_axis)
    if self.channel_axis is None and len(self.index_ranges) > 1:
      raise ValueError(
          "`channel_axis` can't be `None` for `len(index_ranges) > 1`.")
    self._prior_fn = prior_fn
    self._parameter_fns = dict(parameter_fns)
    self._prior_dtype = tf.as_dtype(prior_dtype)
    self.decode_sanity_check = decode_sanity_check

    with self.name_scope:
      if self.compression:
        if self.channel_axis is None:
          index_range, = index_ranges
          indexes = tf.range(index_range, dtype=tf.int32)
        else:
          indexes = [tf.range(r, dtype=tf.int32) for r in index_ranges]
          indexes = tf.meshgrid(*indexes, indexing="ij")
          indexes = tf.stack(indexes, axis=self.channel_axis)
        self._prior = self._make_prior(indexes)
        cdf, cdf_offset = self._build_tables(self.prior, range_coder_precision)
        self._init_compression(cdf, cdf_offset, None)

  @property
  def index_ranges(self):
    """Upper bound(s) on values allowed in `indexes` tensor."""
    return self._index_ranges

  @property
  def parameter_fns(self):
    """Functions mapping `indexes` to each distribution parameter."""
    return self._parameter_fns

  @property
  def prior_dtype(self):
    """Data type of `prior`."""
    return self._prior_dtype

  @property
  def prior_fn(self):
    """Class or factory function returning a `Distribution` object."""
    return self._prior_fn

  @property
  def channel_axis(self):
    """Position of channel axis in `indexes` tensor."""
    return self._channel_axis

  def _make_prior(self, indexes):
    indexes = tf.cast(indexes, self.prior_dtype)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    prior = self.prior_fn(**parameters)
    assert prior.dtype == self.prior_dtype
    if prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")
    return prior

  def _normalize_indexes(self, indexes):
    indexes = math_ops.lower_bound(indexes, 0)
    if self.channel_axis is None:
      index_range, = self.index_ranges
      bounds = index_range - 1
    else:
      axes = [1] * indexes.shape.rank
      axes[self.channel_axis] = len(self.index_ranges)
      bounds = tf.reshape([s - 1 for s in self.index_ranges], axes)
    return math_ops.upper_bound(indexes, tf.cast(bounds, indexes.dtype))

  def _flatten_indexes(self, indexes):
    indexes = tf.cast(indexes, tf.int32)
    if self.channel_axis is None:
      return indexes
    else:
      strides = tf.math.cumprod(self.index_ranges, exclusive=True, reverse=True)
      return tf.linalg.tensordot(indexes, strides, [[self.channel_axis], [0]])

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, indexes, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    indexes = self._normalize_indexes(indexes)
    if training:
      def log_prob_fn(bottleneck_perturbed, indexes):
        # When using expected_grads=True, we will use a tf.custom_gradient on
        # this function. In this case, all non-Variable tensors that determine
        # the result of this function need to be declared explicitly, i.e we
        # need `indexes` to be a declared argument and `prior` instantiated
        # here. If we would instantiate it outside this function declaration and
        # reference here via a closure, we would get a `None` gradient for
        # `indexes`.
        prior = self._make_prior(indexes)
        return self._log_prob(prior, bottleneck_perturbed)
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, indexes, expected_grads=self.expected_grads)
    else:
      prior = self._make_prior(indexes)
      bottleneck_perturbed = self.soft_quantize(bottleneck)
      log_probs = self._log_prob(prior, bottleneck_perturbed)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return round_ops.round_st(bottleneck)
  

  def soft_quantize(self,bottleneck,L=5,temperature=1):
    #print("quantizier is called")
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    centers = tf.cast(tf.range(-2,3), tf.float32)
    w_stack = tf.stack([bottleneck for _ in range(L)], axis=-1)
    w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
    smx = tf.keras.activations.softmax(-1.0/temperature * tf.abs(w_stack - centers), axis=-1)
    w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)
    w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)
    return w_bar

  @tf.Module.with_name_scope
  def compress(self, bottleneck, indexes):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    all_but_last_n_elems = lambda t, n: t[:-n] if n else t
    batch_shape = all_but_last_n_elems(tf.shape(flat_indexes), self.coding_rank)
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols -= tf.gather(self.cdf_offset, flat_indexes)
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    handle = gen_ops.entropy_encode_index(handle, flat_indexes, symbols)
    return gen_ops.entropy_encode_finalize(handle)

  @tf.Module.with_name_scope
  def decompress(self, strings, indexes):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      indexes: `tf.Tensor` specifying the scalar distribution for each output
        element. See class docstring for examples.

    Returns:
      A `tf.Tensor` of the same shape as `indexes` (without the optional channel
      dimension).
    """
    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    last_n_elems = lambda t, n: t[-n:] if n else t[:0]
    decode_shape = last_n_elems(tf.shape(flat_indexes), self.coding_rank)
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    handle, symbols = gen_ops.entropy_decode_index(
        handle, flat_indexes, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    if self.decode_sanity_check:
      with tf.control_dependencies([tf.debugging.assert_equal(
          sanity, True, message="Sanity check failed.")]):
        symbols = tf.identity(symbols)
    symbols += tf.gather(self.cdf_offset, flat_indexes)
    return tf.cast(symbols, self.bottleneck_dtype)

  def get_config(self):
    """Returns the configuration of the entropy model."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")

  @classmethod
  def from_config(cls, config):
    """Instantiates an entropy model from a configuration dictionary."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")


class LocationScaleIndexedEntropySoftModel(ContinuousIndexedEntropyModelwithSoftQuant):
  """Indexed entropy model for location-scale family of random variables.

  This class is a common special case of `ContinuousIndexedEntropyModel`. The
  specified distribution is parameterized with `num_scales` values of scale
  parameters. An element-wise location parameter is handled by shifting the
  distributions to zero.

  This method is illustrated in Figure 10 of:
  > "Nonlinear Transform Coding"<br />
  > J. Ballé, P.A. Chou, D. Minnen, S. Singh, N. Johnston, E. Agustsson,
  > S.J. Hwang, G. Toderici<br />
  > https://doi.org/10.1109/JSTSP.2020.3034501

  Note: this only works for shift-invariant `tfpd.Distribution` objects, where
  the `loc` parameter really denotes a translation (i.e., not for the log-normal
  distribution).
  """

  def __init__(self,
               prior_fn,
               num_scales,
               scale_fn,
               coding_rank,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               bottleneck_dtype=None,
               prior_dtype=tf.float32,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      prior_fn: A callable returning a `tfp.distributions.Distribution` object,
        typically a `Distribution` class or factory function. This is a density
        model fitting the marginal distribution of the bottleneck data with
        additive uniform noise, which is shared a priori between the sender and
        the receiver. For best results, the distributions should be flexible
        enough to have a unit-width uniform distribution as a special case,
        since this is the marginal distribution for bottleneck dimensions that
        are constant. The callable will receive keyword arguments as determined
        by `parameter_fns`.
      num_scales: Integer. Values in `indexes` must be in the range
        `[0, num_scales)`.
      scale_fn: Callable. `indexes` is passed to the callable, and the return
        value is given as `scale` keyword argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
      prior_dtype: `tf.dtypes.DType`. Data type of prior and probability
        computations. Defaults to `tf.float32`.
      laplace_tail_mass: Float, or a float-valued tf.Tensor. If positive,
        will augment the prior with a `NoisyLaplace` mixture component for
        training stability. (experimental)
    """
    num_scales = int(num_scales)
    super().__init__(
        prior_fn=prior_fn,
        index_ranges=(num_scales,),
        parameter_fns=dict(
            loc=lambda _: 0.,
            scale=scale_fn,
        ),
        coding_rank=coding_rank,
        channel_axis=None,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        bottleneck_dtype=bottleneck_dtype,
        prior_dtype=prior_dtype,
        laplace_tail_mass=laplace_tail_mass,
    )

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, scale_indexes, loc=None, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    if loc is None:
      return super().__call__(bottleneck, scale_indexes, training=training)
    else:
      bottleneck, bits = super().__call__(
          bottleneck - loc, scale_indexes, training=training)
      return bottleneck + loc, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, loc=None):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo the location parameters of the prior distribution given in `loc`.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return round_ops.round_st(bottleneck, loc)
  
  @tf.Module.with_name_scope
  def soft_quantize(self,bottleneck,L=5,temperature=1):
      #print("quantizier is called")
      bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
      centers = tf.cast(tf.range(-2,3), tf.float32)
      w_stack = tf.stack([bottleneck for _ in range(L)], axis=-1)
      w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
      smx = tf.keras.activations.softmax(-1.0/temperature * tf.abs(w_stack - centers), axis=-1)
      w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)
      w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)
      return w_bar

  @tf.Module.with_name_scope
  def compress(self, bottleneck, scale_indexes, loc=None):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    if loc is not None:
      bottleneck -= loc
    return super().compress(bottleneck, scale_indexes)

  @tf.Module.with_name_scope
  def decompress(self, strings, scale_indexes, loc=None):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each output
        element.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        output elements is assumed to be zero. Otherwise, specifies the location
        parameter for each output element. Must have the same shape as
        `scale_indexes`.

    Returns:
      A `tf.Tensor` of the same shape as `scale_indexes`.
    """
    values = super().decompress(strings, scale_indexes)
    if loc is not None:
      values += loc
    return values

# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Batched entropy model for continuous random variables."""

import functools
from absl import logging
import tensorflow as tf
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import round_ops


__all__ = [
    "ContinuousBatchedEntropyModel",
]


@tf.keras.utils.register_keras_serializable(package="tensorflow_compression")
class ContinuousBatchedEntropyModelchz(continuous_base.ContinuousEntropyModelBase):
  """Batched entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor (a shared "prior" between sender and receiver). It also pre-computes
  integer probability tables, which can then be used to compress and decompress
  bottleneck tensors reliably across different platforms.

  A typical workflow looks like this:

  - Train a model using an instance of this entropy model as a bottleneck,
    passing the bottleneck tensor through it. With `training=True`, the model
    computes a differentiable upper bound on the number of bits needed to
    compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  statistically independent, and that the parameters of their scalar
  distributions do not depend on data. The innermost dimensions of the
  bottleneck tensor must be broadcastable to the batch shape of `prior`. Any
  dimensions to the left of the batch shape are assumed to be i.i.d., i.e. the
  likelihoods are broadcast to the bottleneck tensor accordingly.

  A more detailed description (and motivation) of this way of performing
  quantization and range coding can be found in the following paper. Please cite
  the paper when using this code for derivative work.

  > "End-to-end Optimized Image Compression"<br />
  > J. Ballé, V. Laparra, E.P. Simoncelli<br />
  > https://openreview.net/forum?id=rJxdQ3jeg

  Entropy models which contain range coding tables (i.e. with
  `compression=True`) can be instantiated in three ways:

  - By providing a continuous "prior" distribution object. The range coding
    tables are then derived from that continuous distribution.
  - From a config as returned by `get_config`, followed by a call to
    `set_weights`. This implements the Keras serialization protocol. In this
    case, the initializer creates empty state variables for the range coding
    tables, which are then filled by `set_weights`. As a consequence, this
    method requires `stateless=False`.
  - In a more low-level way, by directly providing the range coding tables to
    `__init__`, for use cases where the Keras protocol can't be used (e.g., when
    the entropy model must not create variables).

  The `quantization_offset` and `offset_heuristic` arguments control whether
  quantization is performed with respect to integer values, or potentially
  non-integer offsets (i.e., `y = tf.round(x - o) + o`). There are three modes
  of operation:

  - If `quantization_offset` is provided manually (not `None`), these values are
    used and `offset_heuristic` is ineffective.
  - Otherwise, if `offset_heuristic and compression`, the offsets are computed
    once on initialization, and then fixed. If the entropy model is serialized,
    they are preserved.
  - Otherwise, if `offset_heuristic and not compression`, the offsets are
    recomputed every time quantization is performed. Note this may be
    computationally expensive when the prior does not have a mode that is
    computable in closed form (e.g. for `NoisyDeepFactorized`).

  This offset heuristic is discussed in Section III.A of:
  > "Nonlinear Transform Coding"<br />
  > J. Ballé, P.A. Chou, D. Minnen, S. Singh, N. Johnston, E. Agustsson,
  > S.J. Hwang, G. Toderici<br />
  > https://doi.org/10.1109/JSTSP.2020.3034501
  """

  def __init__(self,
               prior=None,
               coding_rank=None,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               bottleneck_dtype=None,
               prior_shape=None,
               cdf=None,
               cdf_offset=None,
               cdf_shapes=None,
               offset_heuristic=True,
               quantization_offset=None,
               decode_sanity_check=True,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      prior: A `tfp.distributions.Distribution` object. A density model fitting
        the marginal distribution of the bottleneck data with additive uniform
        noise, which is shared a priori between the sender and the receiver. For
        best results, the distribution should be flexible enough to have a
        unit-width uniform distribution as a special case, since this is the
        marginal distribution for bottleneck dimensions that are constant. The
        distribution parameters may not depend on data (they must be either
        variables or constants).
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the __call__ method are summed over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
      prior_shape: Batch shape of the prior (dimensions which are not assumed
        i.i.d.). Must be provided if `prior` is omitted.
      cdf: `tf.Tensor` or `None`. If provided, is used for range coding rather
        than tables built from the prior.
      cdf_offset: `tf.Tensor` or `None`. Must be provided along with `cdf`.
      cdf_shapes: Shapes of `cdf` and `cdf_offset`. If provided, empty range
        coding tables are created, which can then be restored using
        `set_weights`. Requires `compression=True` and `stateless=False`.
      offset_heuristic: Boolean. Whether to quantize to non-integer offsets
        heuristically determined from mode/median of prior. Set this to `False`
        if you are using soft quantization during training.
      quantization_offset: `tf.Tensor` or `None`. The quantization offsets to
        use. If provided (not `None`), then `offset_heuristic` is ineffective.
      decode_sanity_check: Boolean. If `True`, an raises an error if the binary
        strings passed into `decompress` are not completely decoded.
      laplace_tail_mass: Float, or a float-valued tf.Tensor. If positive,
        will augment the prior with a `NoisyLaplace` mixture component for
        training stability. (experimental)
    """
    if (prior is None) == (prior_shape is None):
      raise ValueError("Either `prior` or `prior_shape` must be provided.")
    if (prior is None) + (cdf_shapes is None) + (cdf is None) != 2:
      raise ValueError(
          "Must provide exactly one of `prior`, `cdf`, or `cdf_shapes`.")
    if not compression and not (
        cdf is None and cdf_offset is None and cdf_shapes is None):
      raise ValueError("CDFs can't be provided with `compression=False`")
    if prior is not None and prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        bottleneck_dtype=bottleneck_dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    self._prior = prior
    self._offset_heuristic = bool(offset_heuristic)
    self._prior_shape = tf.TensorShape(
        prior_shape if prior is None else prior.batch_shape)
    if self.coding_rank < self.prior_shape.rank:
      raise ValueError("`coding_rank` can't be smaller than `prior_shape`.")
    self.decode_sanity_check = decode_sanity_check

    with self.name_scope:
      if cdf_shapes is not None:
        # `cdf_shapes` being set indicates that we are using the `SavedModel`
        # protocol, which can only provide JSON datatypes. So create a
        # placeholder value depending on whether `quantization_offset` was
        # `None` or not. For this purpose, we expect a Boolean (when in all
        # other cases, we expect either `None` or a tensor).
        assert isinstance(quantization_offset, bool)
        assert self.compression
        if quantization_offset:
          quantization_offset = tf.zeros(self.prior_shape_tensor)
        else:
          quantization_offset = None
      elif quantization_offset is not None:
        # If quantization offset is passed in manually, use it.
        pass
      elif self.offset_heuristic and self.compression:
        # For compression, we need to fix the offset value, so compute it here.
        if self._prior is None:
          raise ValueError(
              "To use the offset heuristic, a `prior` needs to be provided.")
        quantization_offset = helpers.quantization_offset(self.prior)
        # Optimization: if the quantization offset is zero, we don't need to
        # subtract/add it when quantizing, and we don't need to serialize its
        # value. Note that this code will only work in eager mode.
        if (tf.executing_eagerly() and
            tf.reduce_all(tf.equal(quantization_offset, 0.))):
          quantization_offset = None
        else:
          quantization_offset = tf.broadcast_to(
              quantization_offset, self.prior_shape_tensor)
      else:
        quantization_offset = None
      if quantization_offset is None:
        self._quantization_offset = None
      elif self.compression and not self.stateless:
        quantization_offset = tf.cast(
            quantization_offset, self.bottleneck_dtype)
        self._quantization_offset = tf.Variable(
            quantization_offset, trainable=False, name="quantization_offset")
      else:
        quantization_offset = tf.cast(
            quantization_offset, self.bottleneck_dtype)
        self._quantization_offset = tf.convert_to_tensor(
            quantization_offset, name="quantization_offset")
      if self.compression:
        if cdf is None and cdf_shapes is None:
          cdf, cdf_offset = self._build_tables(
              self.prior, range_coder_precision, offset=quantization_offset)
        self._init_compression(cdf, cdf_offset, cdf_shapes)

  @property
  def prior_shape(self):
    """Batch shape of `prior` (dimensions which are not assumed i.i.d.)."""
    return self._prior_shape

  @property
  def prior_shape_tensor(self):
    """Batch shape of `prior` as a `Tensor`."""
    return tf.constant(self.prior_shape.as_list(), dtype=tf.int32)

  @property
  def offset_heuristic(self):
    """Whether to use heuristic to determine quantization offsets."""
    return self._offset_heuristic

  @property
  def quantization_offset(self):
    """The quantization offset used in `quantize` and `compress`."""
    #pdb.set_trace()
    print("into decrator func")
    if self._quantization_offset is not None:
      return tf.convert_to_tensor(self._quantization_offset)
    if self.offset_heuristic and not self.compression:
      if self._prior is None:
        raise RuntimeError(
            "To use the offset heuristic, a `prior` needs to be provided.")
      if not tf.executing_eagerly():
        logging.warning(
            "Computing quantization offsets using offset heuristic within a "
            "tf.function. Ideally, the offset heuristic should only be used "
            "to determine offsets once after training. Depending on the prior, "
            "estimating the offset might be computationally expensive.")
      return tf.cast(
          helpers.quantization_offset(self.prior), self.bottleneck_dtype)
    return None

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution `self.prior`, which is a
        non-differentiable, tight *lower* bound on the number of bits needed to
        compress `bottleneck` using `compress()`. If `True`, returns a somewhat
        looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise, and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    #pdb.set_trace()
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    log_prob_fn = functools.partial(self._log_prob, self.prior)
    if training:
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, expected_grads=self.expected_grads)
    else:
      bottleneck_perturbed = self.quantize(bottleneck)
      log_probs = log_prob_fn(bottleneck_perturbed)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point bottleneck tensor.

    The tensor is rounded to integer values potentially shifted by offsets (if
    `self.quantization_offset is not None`). These offsets can depend on
    `self.prior`. For instance, for a Gaussian distribution, when
    `self.offset_heuristic == True`, the returned values would be rounded
    to the location of the mode of the distribution plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized. The innermost
        dimensions must be broadcastable to `self.prior_shape`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    #pdb.set_trace()
    return round_ops.round_st(bottleneck, self.quantization_offset)

  @tf.Module.with_name_scope
  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables in
    `self.cdf` (derived from `self.prior`). The quantized tensor can later be
    recovered by calling `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    input_shape = tf.shape(bottleneck)
    all_but_last_n_elems = lambda t, n: t[:-n] if n else t
    batch_shape = all_but_last_n_elems(input_shape, self.coding_rank)
    iid_shape = all_but_last_n_elems(input_shape, self.prior_shape.rank)
    offset = self.quantization_offset
    if offset is not None:
      bottleneck -= offset
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols = tf.reshape(symbols, tf.concat([iid_shape, [-1]], 0))
    symbols -= self.cdf_offset
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    handle = gen_ops.entropy_encode_channel(handle, symbols)
    return gen_ops.entropy_encode_finalize(handle)

  @tf.Module.with_name_scope
  def decompress(self, strings, broadcast_shape):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.
    It is necessary to provide a part of the output shape in `broadcast_shape`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      broadcast_shape: Iterable of ints. The part of the output tensor shape
        between the shape of `strings` on the left and `self.prior_shape` on the
        right. This must match the shape of the input to `compress()`.

    Returns:
      A `tf.Tensor` of shape `strings.shape + broadcast_shape +
      self.prior_shape`.
    """
    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    broadcast_shape = tf.convert_to_tensor(broadcast_shape, dtype=tf.int32)
    decode_shape = tf.concat(
        [broadcast_shape, [tf.reduce_prod(self.prior_shape_tensor)]], 0)
    output_shape = tf.concat(
        [tf.shape(strings), broadcast_shape, self.prior_shape_tensor], 0)
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    handle, symbols = gen_ops.entropy_decode_channel(
        handle, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    if self.decode_sanity_check:
      with tf.control_dependencies([tf.debugging.assert_equal(
          sanity, True, message="Sanity check failed.")]):
        symbols = tf.identity(symbols)
    symbols += self.cdf_offset
    symbols = tf.reshape(symbols, output_shape)
    outputs = tf.cast(symbols, self.bottleneck_dtype)
    offset = self.quantization_offset
    if offset is not None:
      outputs += offset
    return outputs

  def get_config(self):
    """Returns the configuration of the entropy model.

    Returns:
      A JSON-serializable Python dict.
    """
    config = super().get_config()
    config.update(
        prior_shape=tuple(map(int, self.prior_shape)),
        offset_heuristic=self.offset_heuristic,
        quantization_offset=self.quantization_offset is not None,
    )
    return config