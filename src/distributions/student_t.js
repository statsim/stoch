import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { LOG_PI } from '../math/numeric'
import { assertPositive } from '../internal/assert-util'

/**
 * Student's t-distribution.
 *
 * Heavy-tailed alternative to Normal. Common in robust regression.
 *
 * pdf(x; df, loc, scale) = Γ((df+1)/2) / (Γ(df/2) * sqrt(df*π) * scale)
 *                           * (1 + ((x-loc)/scale)²/df)^(-(df+1)/2)
 *
 * Parameterized by df (degrees of freedom), loc (location), and scale.
 */
export class StudentT extends Distribution {
  constructor({ df, loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'StudentT'
    })

    this._df = this._addParameter('df', df)
    this._loc = this._addParameter('loc', loc)
    this._scale = this._addParameter('scale', scale)

    if (this._validateArgs) {
      assertPositive(this._df, 'df')
      assertPositive(this._scale, 'scale')
    }
  }

  get df() { return this._df }
  get loc() { return this._loc }
  get scale() { return this._scale }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    // Sample using: T = loc + scale * Z / sqrt(V/df)
    // where Z ~ Normal(0,1) and V ~ Gamma(df/2, df/2) (i.e., Chi2(df)/df)
    const z = tf.randomStandardNormal(shape, 'float32')
    // Gamma(alpha=df/2, beta=1/(df/2))  → scale = 2/df
    const halfDf = tf.div(this._df, 2)
    const alpha = halfDf.dataSync()[0]
    const beta = 1.0 / alpha // scale = 1/rate = 1/(df/2) = 2/df
    const v = tf.randomGamma(shape, alpha, beta) // V ~ Chi2(df) / df would be Gamma(df/2, rate=df/2)
    // Actually: Gamma(alpha, beta) with beta = scale = 2/df gives Gamma(df/2, scale=2/df)
    // mean = alpha * beta = (df/2)*(2/df) = 1. So V has mean 1.
    // t = Z / sqrt(V)
    const t = tf.div(z, tf.sqrt(v))
    return tf.add(tf.mul(t, this._scale), this._loc)
  }

  _logProb(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    const halfDfPlus1 = tf.div(tf.add(this._df, 1), 2)
    const halfDf = tf.div(this._df, 2)

    // log pdf = logΓ((df+1)/2) - logΓ(df/2) - 0.5*log(df*π) - log(scale)
    //           - ((df+1)/2) * log(1 + z²/df)
    const normConst = tf.sub(
      logGamma(halfDfPlus1),
      tf.add(
        logGamma(halfDf),
        tf.add(
          tf.mul(0.5, tf.log(tf.mul(this._df, Math.PI))),
          tf.log(this._scale)
        )
      )
    )

    const logKernel = tf.mul(
      tf.neg(halfDfPlus1),
      tf.log(tf.add(1, tf.div(tf.square(z), this._df)))
    )

    return tf.add(normConst, logKernel)
  }

  _mean() {
    // mean = loc for df > 1, undefined (NaN) otherwise
    return tf.where(
      this._df.greater(1),
      tf.clone(this._loc),
      tf.fill(this._loc.shape, NaN)
    )
  }

  _variance() {
    // var = scale² * df / (df - 2) for df > 2
    // var = Infinity for 1 < df <= 2
    // undefined for df <= 1
    const raw = tf.mul(
      tf.square(this._scale),
      tf.div(this._df, tf.sub(this._df, 2))
    )
    return tf.where(
      this._df.greater(2),
      raw,
      tf.where(
        this._df.greater(1),
        tf.fill(raw.shape, Infinity),
        tf.fill(raw.shape, NaN)
      )
    )
  }

  _mode() {
    return tf.clone(this._loc)
  }

  _entropy() {
    const halfDfPlus1 = tf.div(tf.add(this._df, 1), 2)
    const halfDf = tf.div(this._df, 2)

    // H = 0.5*log(df) + 0.5*log(π) + logΓ(df/2) - logΓ((df+1)/2)
    //     + (df+1)/2 * (ψ((df+1)/2) - ψ(df/2)) + log(scale)
    const { digamma } = require('../math/special')
    return tf.add(
      tf.add(
        tf.mul(0.5, tf.add(tf.log(this._df), LOG_PI)),
        tf.sub(logGamma(halfDf), logGamma(halfDfPlus1))
      ),
      tf.add(
        tf.mul(halfDfPlus1, tf.sub(digamma(halfDfPlus1), digamma(halfDf))),
        tf.log(this._scale)
      )
    )
  }
}
