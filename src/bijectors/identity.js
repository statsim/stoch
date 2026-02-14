import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Identity bijector: f(x) = x.
 * Useful as a no-op placeholder in Chain or TransformedDistribution.
 */
export class Identity extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      isConstantJacobian: true,
      validateArgs,
      name: name || 'Identity'
    })
  }

  _forward(x) { return x.clone() }
  _inverse(y) { return y.clone() }
  _forwardLogDetJacobian(x) { return tf.zeros(x.shape) }
  _inverseLogDetJacobian(y) { return tf.zeros(y.shape) }
}
