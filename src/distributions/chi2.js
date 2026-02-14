import * as tf from '@tensorflow/tfjs'
import { Gamma } from './gamma'

/**
 * Chi-squared distribution.
 *
 * A special case of Gamma: Chi2(df) = Gamma(df/2, 0.5)
 *
 * pdf(x; df) = x^(df/2-1) * exp(-x/2) / (2^(df/2) * Γ(df/2))
 */
export class Chi2 extends Gamma {
  constructor({ df = 1, validateArgs, name } = {}) {
    super({
      concentration: typeof df === 'number' ? df / 2 : tf.div(df, 2),
      rate: 0.5,
      validateArgs,
      name: name || 'Chi2'
    })
    this._df = this._concentration
  }

  get df() {
    return tf.tidy(() => tf.mul(this._df, 2))
  }
}
