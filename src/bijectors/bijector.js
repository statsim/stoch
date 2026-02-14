import * as tf from '@tensorflow/tfjs'
import { toTensor } from '../internal/tensor-util'

/**
 * Abstract base class for bijectors (differentiable, invertible transforms).
 *
 * Mirrors TensorFlow Probability's Bijector class. Subclasses override
 * internal methods (_forward, _inverse, etc.) while the base class handles:
 * - tf.tidy() wrapping for automatic memory management
 * - Input casting
 * - Log-det-Jacobian event dimension reduction
 *
 * A bijector maps between two spaces:
 *   forward: x → y = f(x)
 *   inverse: y → x = f⁻¹(y)
 *
 * The log-det-Jacobian measures how the bijector stretches/compresses volume:
 *   forwardLogDetJacobian:  log|det(df/dx)|
 *   inverseLogDetJacobian:  log|det(df⁻¹/dy)|
 *
 * Usage pattern:
 *   class Exp extends Bijector {
 *     constructor() {
 *       super({ name: 'Exp' })
 *     }
 *     _forward(x) { return tf.exp(x) }
 *     _inverse(y) { return tf.log(y) }
 *     _forwardLogDetJacobian(x) { return x }
 *     _inverseLogDetJacobian(y) { return tf.neg(tf.log(y)) }
 *   }
 */
export class Bijector {
  constructor({
    forwardMinEventNdims = 0,
    inverseMinEventNdims = 0,
    isConstantJacobian = false,
    validateArgs = false,
    name = 'Bijector'
  } = {}) {
    this._forwardMinEventNdims = forwardMinEventNdims
    this._inverseMinEventNdims = inverseMinEventNdims
    this._isConstantJacobian = isConstantJacobian
    this._validateArgs = validateArgs
    this._name = name
  }

  // --- Properties ---

  get name() { return this._name }
  get forwardMinEventNdims() { return this._forwardMinEventNdims }
  get inverseMinEventNdims() { return this._inverseMinEventNdims }
  get isConstantJacobian() { return this._isConstantJacobian }

  // --- Public API (wrap in tf.tidy, delegate to internal) ---

  /**
   * Apply the forward transformation: y = f(x).
   * @param {tf.Tensor|number|number[]} x - Input value(s)
   * @returns {tf.Tensor}
   */
  forward(x) {
    return tf.tidy(() => {
      const xTensor = this._castInput(x)
      return this._forward(xTensor)
    })
  }

  /**
   * Apply the inverse transformation: x = f⁻¹(y).
   * @param {tf.Tensor|number|number[]} y - Input value(s)
   * @returns {tf.Tensor}
   */
  inverse(y) {
    return tf.tidy(() => {
      const yTensor = this._castInput(y)
      return this._inverse(yTensor)
    })
  }

  /**
   * Log absolute determinant of the forward Jacobian.
   * @param {tf.Tensor|number|number[]} x - Input value(s)
   * @param {number} [eventNdims] - Number of event dimensions to reduce over.
   *   Defaults to forwardMinEventNdims.
   * @returns {tf.Tensor}
   */
  forwardLogDetJacobian(x, eventNdims) {
    return tf.tidy(() => {
      const xTensor = this._castInput(x)
      const ndims = eventNdims !== undefined ? eventNdims : this._forwardMinEventNdims
      const ldj = this._forwardLogDetJacobian(xTensor)
      return this._reduceLogDetJacobian(ldj, xTensor.shape, ndims)
    })
  }

  /**
   * Log absolute determinant of the inverse Jacobian.
   * @param {tf.Tensor|number|number[]} y - Input value(s)
   * @param {number} [eventNdims] - Number of event dimensions to reduce over.
   *   Defaults to inverseMinEventNdims.
   * @returns {tf.Tensor}
   */
  inverseLogDetJacobian(y, eventNdims) {
    return tf.tidy(() => {
      const yTensor = this._castInput(y)
      const ndims = eventNdims !== undefined ? eventNdims : this._inverseMinEventNdims
      const ldj = this._inverseLogDetJacobian(yTensor)
      return this._reduceLogDetJacobian(ldj, yTensor.shape, ndims)
    })
  }

  // --- Internal methods (override in subclasses) ---

  _forward(x) {
    throw new Error(`${this._name}._forward not implemented`)
  }

  _inverse(y) {
    throw new Error(`${this._name}._inverse not implemented`)
  }

  _forwardLogDetJacobian(x) {
    throw new Error(`${this._name}._forwardLogDetJacobian not implemented`)
  }

  _inverseLogDetJacobian(y) {
    throw new Error(`${this._name}._inverseLogDetJacobian not implemented`)
  }

  // --- Helpers ---

  /**
   * Cast input to tensor.
   */
  _castInput(value) {
    if (value instanceof tf.Tensor) return value
    return toTensor(value, 'float32')
  }

  /**
   * Reduce log-det-Jacobian over event dimensions.
   * For element-wise bijectors (eventNdims=0), this is a no-op.
   * For higher-dimensional bijectors, sum over the trailing event dims.
   */
  _reduceLogDetJacobian(ldj, inputShape, eventNdims) {
    if (eventNdims === 0) return ldj
    // Sum over the last eventNdims dimensions
    const rank = ldj.shape.length
    if (rank === 0) return ldj
    const axes = []
    for (let i = rank - eventNdims; i < rank; i++) {
      if (i >= 0) axes.push(i)
    }
    if (axes.length === 0) return ldj
    return tf.sum(ldj, axes)
  }

  toString() {
    return `${this._name}()`
  }
}
