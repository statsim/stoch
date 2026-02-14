import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Log bijector: f(x) = log(x).
 * Maps R+ → R. Inverse of Exp.
 *
 * Forward:  y = log(x)
 * Inverse:  x = exp(y)
 * FLDJ:     log|dy/dx| = -log(x)  (since dy/dx = 1/x)
 * ILDJ:     log|dx/dy| = y  (since dx/dy = exp(y))
 */
export class Log extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Log'
    })
  }

  _forward(x) { return tf.log(x) }
  _inverse(y) { return tf.exp(y) }
  _forwardLogDetJacobian(x) { return tf.neg(tf.log(x)) }
  _inverseLogDetJacobian(y) { return y }
}
