import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'
import { softplusInverse } from '../math/generic'

/**
 * Softplus bijector: f(x) = log(1 + exp(x)).
 * Smooth approximation of ReLU. Maps R → R+.
 * More numerically stable than Exp for constrained positive parameters.
 *
 * Forward:  y = log(1 + exp(x))
 * Inverse:  x = log(exp(y) - 1)   (softplus inverse)
 * FLDJ:     -log(1 + exp(-x)) = -softplus(-x)
 * ILDJ:     -log(1 - exp(-y)) = y - log(exp(y) - 1)
 */
export class Softplus extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Softplus'
    })
  }

  _forward(x) {
    return tf.softplus(x)
  }

  _inverse(y) {
    return softplusInverse(y)
  }

  _forwardLogDetJacobian(x) {
    // log(sigmoid(x)) = -softplus(-x) = x - softplus(x)
    return tf.sub(x, tf.softplus(x))
  }

  _inverseLogDetJacobian(y) {
    // log|dx/dy| = -log(1 - exp(-y))
    // = -log(expm1(y)/exp(y)) = log(exp(y)) - log(expm1(y)) = y - log(expm1(y))
    // Use softplusInverse(y) = log(exp(y) - 1), so ILDJ = y - softplusInverse(y)
    // But that's actually log(sigmoid(softplusInverse(y))), which = -FLDJ(softplusInverse(y))
    // Simpler: ILDJ = -FLDJ(inverse(y)), by inverse function theorem
    const x = softplusInverse(y)
    return tf.sub(tf.softplus(x), x)
  }
}
