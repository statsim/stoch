import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Tanh bijector: f(x) = tanh(x).
 * Maps R → (-1, 1). Useful for bounding outputs.
 *
 * Forward:  y = tanh(x)
 * Inverse:  x = atanh(y) = 0.5 * log((1+y)/(1-y))
 * FLDJ:     log(1 - tanh(x)²) = 2*(log(2) - x - softplus(-2x))
 * ILDJ:     -log(1 - y²)
 */
export class Tanh extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Tanh'
    })
  }

  _forward(x) {
    return tf.tanh(x)
  }

  _inverse(y) {
    return tf.atanh(y)
  }

  _forwardLogDetJacobian(x) {
    // log(1 - tanh²(x)) = log(sech²(x)) = 2*log(sech(x))
    // = 2*(log(2) - x - softplus(-2*x))
    return tf.mul(2, tf.sub(tf.sub(Math.log(2), x), tf.softplus(tf.mul(-2, x))))
  }

  _inverseLogDetJacobian(y) {
    // -log(1 - y²)
    return tf.neg(tf.log(tf.sub(1, tf.mul(y, y))))
  }
}
