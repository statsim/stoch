import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Exp bijector: f(x) = exp(x).
 * Maps R → R+. Used for positive-valued parameters (scale, rate, etc.).
 *
 * Forward:  y = exp(x)
 * Inverse:  x = log(y)
 * FLDJ:     log|dy/dx| = x  (since dy/dx = exp(x) and log(exp(x)) = x)
 * ILDJ:     log|dx/dy| = -log(y)
 */
export class Exp extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Exp'
    })
  }

  _forward(x) { return tf.exp(x) }
  _inverse(y) { return tf.log(y) }
  _forwardLogDetJacobian(x) { return x }
  _inverseLogDetJacobian(y) { return tf.neg(tf.log(y)) }
}
