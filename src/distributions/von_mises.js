import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { besselI0, besselI1, logBesselI0 } from '../math/special'
import { LOG_2PI } from '../math/numeric'
import { assertNonNegative } from '../internal/assert-util'

/**
 * Von Mises distribution (circular normal distribution).
 *
 * Circular distribution on [-π, π].
 *
 * pdf(x; μ, κ) = exp(κ * cos(x - μ)) / (2π * I₀(κ))
 *
 * Parameterized by loc (mean direction μ) and concentration (κ >= 0).
 *
 * Mirrors TFP Python's VonMises distribution.
 */
export class VonMises extends Distribution {
  constructor({ loc = 0, concentration, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'VonMises'
    })

    this._loc = this._addParameter('loc', loc)
    this._concentration = this._addParameter('concentration', concentration)

    if (this._validateArgs) {
      assertNonNegative(this._concentration, 'concentration')
    }
  }

  get loc() { return this._loc }
  get concentration() { return this._concentration }

  _sampleN(n) {
    // Best-Fisher algorithm for Von Mises sampling
    // Reference: Best & Fisher (1979), Applied Statistics 28(2), pp. 152-157
    const shape = [n, ...this.batchShape]
    const totalSize = shape.reduce((a, b) => a * b, 1)
    const kappa = this._concentration.dataSync()[0]
    const mu = this._loc.dataSync()[0]

    const samples = new Float32Array(totalSize)

    if (kappa < 1e-6) {
      // For very small kappa, sample uniformly on [-pi, pi]
      for (let i = 0; i < totalSize; i++) {
        samples[i] = (Math.random() * 2 - 1) * Math.PI
      }
    } else {
      // Best-Fisher algorithm
      const tau = 1 + Math.sqrt(1 + 4 * kappa * kappa)
      const rho = (tau - Math.sqrt(2 * tau)) / (2 * kappa)
      const r = (1 + rho * rho) / (2 * rho)

      for (let i = 0; i < totalSize; i++) {
        let accepted = false
        while (!accepted) {
          const u1 = Math.random()
          const u2 = Math.random()
          const z = Math.cos(Math.PI * u1)
          const f = (1 + r * z) / (r + z)
          const c = kappa * (r - f)

          if (c * (2 - c) > u2 || Math.log(c / u2) + 1 >= c) {
            const u3 = Math.random()
            const theta = (u3 > 0.5 ? 1 : -1) * Math.acos(f)
            // Wrap to [-pi, pi] relative to loc
            let s = theta + mu
            // Normalize to [-pi, pi]
            s = s - 2 * Math.PI * Math.round(s / (2 * Math.PI))
            samples[i] = s
            accepted = true
          }
        }
      }
    }

    return tf.tensor(samples, shape)
  }

  _logProb(value) {
    // log pdf = κ * cos(x - μ) - log(2π) - logI₀(κ)
    const kappa = this._concentration.dataSync()[0]
    const logNorm = LOG_2PI + logBesselI0(kappa)
    return tf.sub(
      tf.mul(this._concentration, tf.cos(tf.sub(value, this._loc))),
      logNorm
    )
  }

  _entropy() {
    // H = -κ * I₁(κ)/I₀(κ) + log(2π * I₀(κ))
    const kappa = this._concentration.dataSync()[0]
    const i0 = besselI0(kappa)
    const i1 = besselI1(kappa)
    const h = -kappa * (i1 / i0) + Math.log(2 * Math.PI * i0)
    return tf.scalar(h)
  }

  _mean() {
    // Circular mean = loc
    return tf.clone(this._loc)
  }

  _variance() {
    // Circular variance = 1 - I₁(κ)/I₀(κ)
    const kappa = this._concentration.dataSync()[0]
    const i0 = besselI0(kappa)
    const i1 = besselI1(kappa)
    return tf.scalar(1 - i1 / i0)
  }

  _mode() {
    // Mode = loc
    return tf.clone(this._loc)
  }
}
