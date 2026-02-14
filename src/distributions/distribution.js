import * as tf from '@tensorflow/tfjs'
import { toTensor, broadcastShapesMultiple } from '../internal/tensor-util'

/**
 * Abstract base class for probability distributions.
 *
 * Mirrors TensorFlow Probability's Distribution class. Subclasses override
 * internal methods (_logProb, _sampleN, etc.) while the base class handles:
 * - tf.tidy() wrapping for automatic memory management
 * - Parameter validation (when validateArgs is true)
 * - Input casting and broadcasting
 * - Default implementations (prob from logProb, stddev from variance, etc.)
 *
 * Usage pattern:
 *   class Normal extends Distribution {
 *     constructor({ loc, scale, validateArgs, name }) {
 *       super({ dtype: 'float32', validateArgs, name: name || 'Normal' })
 *       this._loc = this._addParameter('loc', loc)
 *       this._scale = this._addParameter('scale', scale)
 *     }
 *     _logProb(value) { ... }
 *     _sampleN(n) { ... }
 *     ...
 *   }
 */
export class Distribution {
  constructor({
    dtype = 'float32',
    validateArgs = true,
    allowNanStats = true,
    name = 'Distribution'
  } = {}) {
    this._dtype = dtype
    this._validateArgs = validateArgs
    this._allowNanStats = allowNanStats
    this._name = name
    this._parameters = {}
    this._paramTensors = {}
  }

  /**
   * Register a parameter tensor. Called by subclass constructors.
   * Converts the value to a tensor and stores it for broadcasting + disposal.
   */
  _addParameter(name, value, dtype) {
    const t = toTensor(value, dtype || this._dtype)
    this._parameters[name] = value
    this._paramTensors[name] = t
    return t
  }

  /**
   * Compute batchShape by broadcasting all parameter tensor shapes.
   */
  _computeBatchShape() {
    const shapes = Object.values(this._paramTensors).map(t => t.shape)
    if (shapes.length === 0) return []
    return broadcastShapesMultiple(...shapes)
  }

  // --- Properties ---

  get dtype() { return this._dtype }
  get name() { return this._name }
  get parameters() { return { ...this._parameters } }

  get batchShape() {
    return this._computeBatchShape()
  }

  get eventShape() {
    return this._eventShape()
  }

  _eventShape() {
    return []
  }

  // --- Public methods with tf.tidy wrapping ---

  /**
   * Draw samples from the distribution.
   * @param {number[]} [shape=[]] - sample shape
   * @returns {tf.Tensor} shape: [...shape, ...batchShape, ...eventShape]
   */
  sample(shape) {
    if (!shape) shape = []
    if (typeof shape === 'number') shape = [shape]
    const n = shape.reduce((a, b) => a * b, 1) || 1
    const result = tf.tidy(() => this._sampleN(n))
    // Reshape to [...shape, ...batchShape, ...eventShape]
    if (shape.length > 1 || (shape.length === 1 && n > 1)) {
      const fullShape = [...shape, ...this.batchShape, ...this.eventShape]
      const reshaped = result.reshape(fullShape)
      result.dispose()
      return reshaped
    }
    return result
  }

  /**
   * Log probability density/mass function.
   */
  logProb(value) {
    return tf.tidy(() => {
      const v = this._castInput(value)
      return this._logProb(v)
    })
  }

  /**
   * Probability density/mass function.
   * Default: exp(logProb(value)). Subclass can override _prob for stability.
   */
  prob(value) {
    return tf.tidy(() => {
      const v = this._castInput(value)
      if (this._prob !== Distribution.prototype._prob) {
        return this._prob(v)
      }
      return tf.exp(this._logProb(v))
    })
  }

  /**
   * Cumulative distribution function.
   */
  cdf(value) {
    return tf.tidy(() => {
      const v = this._castInput(value)
      return this._cdf(v)
    })
  }

  /**
   * Log of the cumulative distribution function.
   * Default: log(cdf(value)).
   */
  logCdf(value) {
    return tf.tidy(() => {
      const v = this._castInput(value)
      if (this._logCdf !== Distribution.prototype._logCdf) {
        return this._logCdf(v)
      }
      return tf.log(this._cdf(v))
    })
  }

  /**
   * Shannon entropy of the distribution.
   */
  entropy() {
    return tf.tidy(() => this._entropy())
  }

  /**
   * Mean of the distribution.
   */
  mean() {
    return tf.tidy(() => this._mean())
  }

  /**
   * Variance of the distribution.
   */
  variance() {
    return tf.tidy(() => this._variance())
  }

  /**
   * Standard deviation. Default: sqrt(variance()).
   */
  stddev() {
    return tf.tidy(() => {
      if (this._stddev !== Distribution.prototype._stddev) {
        return this._stddev()
      }
      return tf.sqrt(this._variance())
    })
  }

  /**
   * Mode of the distribution.
   */
  mode() {
    return tf.tidy(() => this._mode())
  }

  // --- Internal methods (override in subclasses) ---

  _sampleN(n) {
    throw new Error(`${this._name}._sampleN not implemented`)
  }

  _logProb(value) {
    throw new Error(`${this._name}._logProb not implemented`)
  }

  _prob(value) {
    return tf.exp(this._logProb(value))
  }

  _cdf(value) {
    throw new Error(`${this._name}._cdf not implemented`)
  }

  _logCdf(value) {
    return tf.log(this._cdf(value))
  }

  _entropy() {
    throw new Error(`${this._name}._entropy not implemented`)
  }

  _mean() {
    throw new Error(`${this._name}._mean not implemented`)
  }

  _variance() {
    throw new Error(`${this._name}._variance not implemented`)
  }

  _stddev() {
    return tf.sqrt(this._variance())
  }

  _mode() {
    throw new Error(`${this._name}._mode not implemented`)
  }

  // --- Helpers ---

  /**
   * Cast an input value to a tensor compatible with this distribution.
   */
  _castInput(value) {
    return toTensor(value, this._dtype)
  }

  /**
   * Dispose all parameter tensors to free memory.
   * Call when the distribution is no longer needed.
   */
  dispose() {
    for (const t of Object.values(this._paramTensors)) {
      if (t instanceof tf.Tensor && !t.isDisposed) {
        t.dispose()
      }
    }
  }

  toString() {
    return `${this._name}(${JSON.stringify(this._parameters)})`
  }
}
