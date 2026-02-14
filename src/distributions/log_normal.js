import * as tf from '@tensorflow/tfjs'
import { Normal } from './normal'
import { TransformedDistribution } from './transformed_distribution'
import { Exp } from '../bijectors/exp'

/**
 * LogNormal distribution.
 *
 * If X ~ Normal(loc, scale), then Y = exp(X) ~ LogNormal(loc, scale).
 * Implemented as TransformedDistribution(Normal, Exp).
 *
 * Common prior for positive-valued parameters (e.g., variance, rate).
 *
 * Properties:
 *   mean = exp(loc + scale²/2)
 *   variance = (exp(scale²) - 1) * exp(2*loc + scale²)
 *   mode = exp(loc - scale²)
 */
export class LogNormal extends TransformedDistribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    const normal = new Normal({ loc, scale, validateArgs })
    super({
      distribution: normal,
      bijector: new Exp(),
      validateArgs,
      name: name || 'LogNormal'
    })
  }

  get loc() { return this._distribution.loc }
  get scale() { return this._distribution.scale }

  _mean() {
    // E[Y] = exp(μ + σ²/2)
    return tf.exp(tf.add(this.loc, tf.mul(0.5, tf.square(this.scale))))
  }

  _variance() {
    // Var[Y] = (exp(σ²) - 1) * exp(2μ + σ²)
    const s2 = tf.square(this.scale)
    return tf.mul(
      tf.sub(tf.exp(s2), 1),
      tf.exp(tf.add(tf.mul(2, this.loc), s2))
    )
  }

  _mode() {
    // mode = exp(μ - σ²)
    return tf.exp(tf.sub(this.loc, tf.square(this.scale)))
  }

  _entropy() {
    // H = μ + 0.5 + log(σ) + 0.5*log(2π)
    const LOG_2PI = Math.log(2 * Math.PI)
    return tf.add(
      tf.add(this.loc, 0.5),
      tf.add(tf.log(this.scale), 0.5 * LOG_2PI)
    )
  }
}
