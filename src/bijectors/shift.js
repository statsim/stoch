import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'
import { toTensor } from '../internal/tensor-util'

/**
 * Shift bijector: f(x) = x + shift.
 * Constant Jacobian (shift doesn't change volume).
 *
 * Forward:  y = x + shift
 * Inverse:  x = y - shift
 * FLDJ:     0 (constant)
 * ILDJ:     0 (constant)
 */
export class Shift extends Bijector {
  constructor({ shift, validateArgs, name } = {}) {
    super({
      isConstantJacobian: true,
      validateArgs,
      name: name || 'Shift'
    })
    this._shift = toTensor(shift, 'float32')
  }

  get shift() { return this._shift }

  _forward(x) { return tf.add(x, this._shift) }
  _inverse(y) { return tf.sub(y, this._shift) }
  _forwardLogDetJacobian(x) { return tf.zeros(x.shape) }
  _inverseLogDetJacobian(y) { return tf.zeros(y.shape) }

  dispose() {
    if (this._shift instanceof tf.Tensor && !this._shift.isDisposed) {
      this._shift.dispose()
    }
  }
}
