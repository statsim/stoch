import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { ndtr, logNdtr } from '../math/special'
import { LOG_2PI } from '../math/numeric'
import { assertPositive } from '../internal/assert-util'

/**
 * Truncated Normal distribution.
 *
 * Normal distribution restricted to the interval [low, high].
 *
 * pdf(x; μ, σ, a, b) = φ((x-μ)/σ) / (σ * (Φ((b-μ)/σ) - Φ((a-μ)/σ)))
 *   for a <= x <= b, 0 otherwise
 *
 * where φ is the standard normal pdf and Φ is the standard normal cdf.
 *
 * Parameterized by loc (μ), scale (σ > 0), low (a), and high (b > a).
 *
 * Mirrors TFP Python's TruncatedNormal distribution.
 */
export class TruncatedNormal extends Distribution {
  constructor({ loc = 0, scale = 1, low, high, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'TruncatedNormal'
    })

    this._loc = this._addParameter('loc', loc)
    this._scale = this._addParameter('scale', scale)
    this._low = this._addParameter('low', low)
    this._high = this._addParameter('high', high)

    if (this._validateArgs) {
      assertPositive(this._scale, 'scale')
      const lo = this._low.dataSync()[0]
      const hi = this._high.dataSync()[0]
      if (lo >= hi) {
        throw new Error(`low must be less than high, got low=${lo}, high=${hi}`)
      }
    }
  }

  get loc() { return this._loc }
  get scale() { return this._scale }
  get low() { return this._low }
  get high() { return this._high }

  /**
   * Standardize: z = (x - μ) / σ
   */
  _standardize(value) {
    return tf.div(tf.sub(value, this._loc), this._scale)
  }

  /**
   * Standardized bounds
   */
  _stdLow() {
    return tf.div(tf.sub(this._low, this._loc), this._scale)
  }

  _stdHigh() {
    return tf.div(tf.sub(this._high, this._loc), this._scale)
  }

  /**
   * Log of the normalizing constant: log(Φ(high_std) - Φ(low_std))
   */
  _logNormConst() {
    const phiHigh = ndtr(this._stdHigh())
    const phiLow = ndtr(this._stdLow())
    return tf.log(tf.sub(phiHigh, phiLow))
  }

  _sampleN(n) {
    // Inverse CDF sampling via rejection
    // Draw from uniform, then invert the truncated normal CDF
    // CDF^{-1}(u) = Φ^{-1}(Φ(a_std) + u * (Φ(b_std) - Φ(a_std))) * σ + μ
    // For simplicity, use rejection sampling from normal
    const shape = [n, ...this.batchShape]
    const totalSize = shape.reduce((a, b) => a * b, 1)
    const mu = this._loc.dataSync()[0]
    const sigma = this._scale.dataSync()[0]
    const lo = this._low.dataSync()[0]
    const hi = this._high.dataSync()[0]

    const samples = new Float32Array(totalSize)
    for (let i = 0; i < totalSize; i++) {
      let accepted = false
      while (!accepted) {
        // Use uniform CDF inversion via the probit function approximation
        // For efficiency, use rejection sampling from normal
        const z = _randomNormal()
        const x = mu + sigma * z
        if (x >= lo && x <= hi) {
          samples[i] = x
          accepted = true
        }
      }
    }
    return tf.tensor(samples, shape)
  }

  _logProb(value) {
    // log pdf = -0.5*(z² + log(2π)) - log(σ) - log(Φ(high_std) - Φ(low_std))
    // for x in [low, high], -Infinity otherwise
    const z = this._standardize(value)
    const logNormPdf = tf.sub(
      tf.mul(-0.5, tf.add(tf.square(z), LOG_2PI)),
      tf.log(this._scale)
    )
    const logP = tf.sub(logNormPdf, this._logNormConst())
    const inSupport = tf.logicalAnd(
      value.greaterEqual(this._low),
      value.lessEqual(this._high)
    )
    return tf.where(inSupport, logP, tf.fill(value.shape, -Infinity))
  }

  _cdf(value) {
    // CDF = (Φ(z) - Φ(low_std)) / (Φ(high_std) - Φ(low_std))
    // clipped to [0, 1] for values outside [low, high]
    const z = this._standardize(value)
    const phiZ = ndtr(z)
    const phiLow = ndtr(this._stdLow())
    const phiHigh = ndtr(this._stdHigh())
    const raw = tf.div(tf.sub(phiZ, phiLow), tf.sub(phiHigh, phiLow))
    return tf.clipByValue(raw, 0, 1)
  }

  _entropy() {
    // H = 0.5*log(2πe) + log(σ) + log(Φ(b_std) - Φ(a_std))
    //     + (a_std * φ(a_std) - b_std * φ(b_std)) / (2 * (Φ(b_std) - Φ(a_std)))
    // where φ is standard normal pdf
    const aStd = this._stdLow()
    const bStd = this._stdHigh()
    const phiLow = ndtr(aStd)
    const phiHigh = ndtr(bStd)
    const Z = tf.sub(phiHigh, phiLow)
    const logZ = tf.log(Z)

    // Standard normal pdf: φ(z) = exp(-0.5*z²) / sqrt(2π)
    const pdfLow = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(aStd))),
      1 / Math.sqrt(2 * Math.PI)
    )
    const pdfHigh = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(bStd))),
      1 / Math.sqrt(2 * Math.PI)
    )

    const correction = tf.div(
      tf.sub(tf.mul(aStd, pdfLow), tf.mul(bStd, pdfHigh)),
      tf.mul(2, Z)
    )

    return tf.add(
      tf.add(0.5 * (LOG_2PI + 1), tf.log(this._scale)),
      tf.add(logZ, correction)
    )
  }

  _mean() {
    // mean = μ + σ * (φ(a_std) - φ(b_std)) / (Φ(b_std) - Φ(a_std))
    const aStd = this._stdLow()
    const bStd = this._stdHigh()
    const phiLow = ndtr(aStd)
    const phiHigh = ndtr(bStd)
    const Z = tf.sub(phiHigh, phiLow)

    // Standard normal pdf
    const pdfLow = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(aStd))),
      1 / Math.sqrt(2 * Math.PI)
    )
    const pdfHigh = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(bStd))),
      1 / Math.sqrt(2 * Math.PI)
    )

    return tf.add(
      this._loc,
      tf.mul(this._scale, tf.div(tf.sub(pdfLow, pdfHigh), Z))
    )
  }

  _variance() {
    // var = σ² * [1 + (a_std*φ(a_std) - b_std*φ(b_std)) / Z
    //            - ((φ(a_std) - φ(b_std)) / Z)²]
    const aStd = this._stdLow()
    const bStd = this._stdHigh()
    const phiLow = ndtr(aStd)
    const phiHigh = ndtr(bStd)
    const Z = tf.sub(phiHigh, phiLow)

    const pdfLow = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(aStd))),
      1 / Math.sqrt(2 * Math.PI)
    )
    const pdfHigh = tf.mul(
      tf.exp(tf.mul(-0.5, tf.square(bStd))),
      1 / Math.sqrt(2 * Math.PI)
    )

    const pdfDiff = tf.sub(pdfLow, pdfHigh)
    const term1 = tf.div(
      tf.sub(tf.mul(aStd, pdfLow), tf.mul(bStd, pdfHigh)),
      Z
    )
    const term2 = tf.square(tf.div(pdfDiff, Z))

    return tf.mul(
      tf.square(this._scale),
      tf.sub(tf.add(1, term1), term2)
    )
  }

  _mode() {
    // Mode is loc clamped to [low, high]
    return tf.clipByValue(this._loc, this._low.dataSync()[0], this._high.dataSync()[0])
  }
}

/**
 * Generate a standard normal random variate using Box-Muller transform.
 */
function _randomNormal() {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}
