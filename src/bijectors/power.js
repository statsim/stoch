import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'
import { toTensor } from '../internal/tensor-util'

/**
 * Power bijector: f(x) = x^power.
 * Maps R+ → R+ (for positive power). Generalizes Exp (power→∞) and Identity (power=1).
 *
 * Forward:  y = x^p
 * Inverse:  x = y^(1/p)
 * FLDJ:     log|p| + (p-1)*log|x|
 * ILDJ:     log|1/p| + (1/p - 1)*log|y|
 */
export class Power extends Bijector {
  constructor({ power, validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Power'
    })
    this._power = toTensor(power, 'float32')
  }

  get power() { return this._power }

  _forward(x) {
    return tf.pow(x, this._power)
  }

  _inverse(y) {
    return tf.pow(y, tf.div(1, this._power))
  }

  _forwardLogDetJacobian(x) {
    // log|p * x^(p-1)| = log|p| + (p-1)*log|x|
    return tf.add(
      tf.log(tf.abs(this._power)),
      tf.mul(tf.sub(this._power, 1), tf.log(tf.abs(x)))
    )
  }

  _inverseLogDetJacobian(y) {
    // log|(1/p) * y^(1/p - 1)| = -log|p| + (1/p - 1)*log|y|
    const invP = tf.div(1, this._power)
    return tf.add(
      tf.neg(tf.log(tf.abs(this._power))),
      tf.mul(tf.sub(invP, 1), tf.log(tf.abs(y)))
    )
  }

  dispose() {
    if (this._power instanceof tf.Tensor && !this._power.isDisposed) {
      this._power.dispose()
    }
  }
}
