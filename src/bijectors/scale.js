import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'
import { toTensor } from '../internal/tensor-util'

/**
 * Scale bijector: f(x) = x * scale.
 * Constant Jacobian (linear transform).
 *
 * Forward:  y = x * scale
 * Inverse:  x = y / scale
 * FLDJ:     log|scale| (constant, broadcast to match input shape)
 * ILDJ:     -log|scale|
 */
export class Scale extends Bijector {
  constructor({ scale, validateArgs, name } = {}) {
    super({
      isConstantJacobian: true,
      validateArgs,
      name: name || 'Scale'
    })
    this._scale = toTensor(scale, 'float32')
  }

  get scale() { return this._scale }

  _forward(x) { return tf.mul(x, this._scale) }
  _inverse(y) { return tf.div(y, this._scale) }

  _forwardLogDetJacobian(x) {
    const logAbsScale = tf.log(tf.abs(this._scale))
    return tf.fill(x.shape, logAbsScale.dataSync()[0])
  }

  _inverseLogDetJacobian(y) {
    const logAbsScale = tf.log(tf.abs(this._scale))
    return tf.fill(y.shape, -logAbsScale.dataSync()[0])
  }

  dispose() {
    if (this._scale instanceof tf.Tensor && !this._scale.isDisposed) {
      this._scale.dispose()
    }
  }
}
