import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'

/**
 * Distribution transformed by a bijector.
 *
 * If X ~ base_distribution and Y = bijector.forward(X), then this represents
 * the distribution of Y. The log probability is adjusted by the inverse
 * log-det-Jacobian of the bijector.
 *
 * Example:
 *   // LogNormal = Normal transformed by Exp
 *   const logNormal = new TransformedDistribution({
 *     distribution: new Normal({ loc: 0, scale: 1 }),
 *     bijector: new Exp()
 *   })
 *
 * logProb(y) = base.logProb(bijector.inverse(y)) + bijector.inverseLogDetJacobian(y)
 */
export class TransformedDistribution extends Distribution {
  constructor({ distribution, bijector, validateArgs, name } = {}) {
    super({
      dtype: distribution.dtype,
      validateArgs: validateArgs != null ? validateArgs : distribution._validateArgs,
      name: name || `Transformed${distribution.name}`
    })
    this._distribution = distribution
    this._bijector = bijector
  }

  get distribution() { return this._distribution }
  get bijector() { return this._bijector }

  get batchShape() {
    return this._distribution.batchShape
  }

  get eventShape() {
    return this._distribution.eventShape
  }

  _sampleN(n) {
    const baseSamples = this._distribution._sampleN(n)
    return this._bijector._forward(baseSamples)
  }

  _logProb(value) {
    const x = this._bijector._inverse(value)
    const baseLogProb = this._distribution._logProb(x)
    const ildj = this._bijector._inverseLogDetJacobian(value)
    return tf.add(baseLogProb, ildj)
  }

  _mean() {
    // Only valid for bijectors where E[f(X)] = f(E[X]), e.g., affine.
    // For general bijectors, fall back to the base class error.
    throw new Error(`${this._name}._mean not implemented for general bijectors`)
  }

  _variance() {
    throw new Error(`${this._name}._variance not implemented for general bijectors`)
  }

  _entropy() {
    // H(Y) = H(X) - E[log|det(df/dx)|]
    // Only simple for constant-Jacobian bijectors
    if (this._bijector.isConstantJacobian) {
      const baseEntropy = this._distribution._entropy()
      // For constant Jacobian, FLDJ is the same everywhere
      const dummy = tf.zeros(this._distribution.batchShape.length > 0
        ? this._distribution.batchShape : [1])
      const fldj = this._bijector._forwardLogDetJacobian(dummy)
      return tf.add(baseEntropy, fldj)
    }
    throw new Error(`${this._name}._entropy not implemented for non-constant Jacobian bijectors`)
  }

  dispose() {
    this._distribution.dispose()
    if (this._bijector.dispose) this._bijector.dispose()
  }
}
