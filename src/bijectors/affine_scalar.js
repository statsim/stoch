import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'
import { toTensor } from '../internal/tensor-util'

/**
 * AffineScalar bijector: f(x) = shift + scale * x.
 * Combines shift and scale in a single bijector. Constant Jacobian.
 *
 * Forward:  y = shift + scale * x
 * Inverse:  x = (y - shift) / scale
 * FLDJ:     log|scale|
 * ILDJ:     -log|scale|
 */
export class AffineScalar extends Bijector {
  constructor({ shift = 0, scale = 1, validateArgs, name } = {}) {
    super({
      isConstantJacobian: true,
      validateArgs,
      name: name || 'AffineScalar'
    })
    this._shift = toTensor(shift, 'float32')
    this._scale = toTensor(scale, 'float32')
  }

  get shift() { return this._shift }
  get scale() { return this._scale }

  _forward(x) {
    return tf.add(this._shift, tf.mul(this._scale, x))
  }

  _inverse(y) {
    return tf.div(tf.sub(y, this._shift), this._scale)
  }

  _forwardLogDetJacobian(x) {
    const logAbsScale = tf.log(tf.abs(this._scale))
    return tf.fill(x.shape, logAbsScale.dataSync()[0])
  }

  _inverseLogDetJacobian(y) {
    const logAbsScale = tf.log(tf.abs(this._scale))
    return tf.fill(y.shape, -logAbsScale.dataSync()[0])
  }

  dispose() {
    if (this._shift instanceof tf.Tensor && !this._shift.isDisposed) {
      this._shift.dispose()
    }
    if (this._scale instanceof tf.Tensor && !this._scale.isDisposed) {
      this._scale.dispose()
    }
  }
}
