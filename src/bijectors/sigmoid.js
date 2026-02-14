import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Sigmoid bijector: f(x) = 1 / (1 + exp(-x)).
 * Maps R → (0, 1). Used for probability parameters.
 *
 * Forward:  y = sigmoid(x)
 * Inverse:  x = logit(y) = log(y / (1 - y))
 * FLDJ:     log|dy/dx| = -softplus(-x) - softplus(x) = x - 2*softplus(x)
 * ILDJ:     log|dx/dy| = -log(y) - log(1-y)
 */
export class Sigmoid extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      validateArgs,
      name: name || 'Sigmoid'
    })
  }

  _forward(x) {
    return tf.sigmoid(x)
  }

  _inverse(y) {
    // logit(y) = log(y / (1 - y)) = log(y) - log(1 - y)
    return tf.sub(tf.log(y), tf.log(tf.sub(1, y)))
  }

  _forwardLogDetJacobian(x) {
    // log sigmoid'(x) = log(sigmoid(x) * (1 - sigmoid(x)))
    // = log(sigmoid(x)) + log(1 - sigmoid(x))
    // = -softplus(-x) + (-softplus(x))
    // = -(softplus(-x) + softplus(x))
    return tf.neg(tf.add(tf.softplus(tf.neg(x)), tf.softplus(x)))
  }

  _inverseLogDetJacobian(y) {
    // log|dx/dy| = -log(y) - log(1 - y)
    return tf.neg(tf.add(tf.log(y), tf.log(tf.sub(1, y))))
  }
}
