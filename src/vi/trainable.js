import * as tf from '@tensorflow/tfjs'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * Inverse softplus: x → log(exp(x) - 1).
 * Used to initialize unconstrained variables for positive parameters.
 */
function softplusInverse(x) {
  if (x > 20) return x  // For large x, softplus(x) ≈ x
  return Math.log(Math.exp(x) - 1)
}

/**
 * Trainable Normal distribution for variational inference.
 *
 * Creates a Normal distribution with tf.variable() parameters
 * that can be optimized via gradient descent. Uses the
 * reparameterization trick: z = μ + σ * ε, ε ~ N(0,1).
 *
 * Scale is internally parameterized via softplus to ensure σ > 0.
 *
 * @param {Object} params
 * @param {number} [params.loc=0] - initial mean
 * @param {number} [params.scale=1] - initial standard deviation
 * @param {string} [params.name=''] - variable name prefix
 * @returns {TrainableNormal}
 */
let _varCounter = 0

export function trainableNormal({ loc = 0, scale = 1, name = '' } = {}) {
  const id = _varCounter++
  const prefix = name ? `${name}_${id}` : `v${id}`
  const locVar = tf.variable(tf.scalar(loc), true, `${prefix}_loc`)
  const unconstrainedScale = tf.variable(
    tf.scalar(softplusInverse(scale)), true, `${prefix}_unconstrained_scale`
  )

  return new TrainableNormal(locVar, unconstrainedScale)
}

class TrainableNormal {
  constructor(locVar, unconstrainedScale) {
    this._locVar = locVar
    this._unconstrainedScale = unconstrainedScale
  }

  get trainableVariables() {
    return [this._locVar, this._unconstrainedScale]
  }

  get loc() { return this._locVar }

  getScale() {
    return tf.softplus(this._unconstrainedScale)
  }

  /**
   * Sample using the reparameterization trick.
   * z = μ + σ * ε, ε ~ N(0, 1)
   *
   * @param {number[]} [shape=[]] - sample shape
   * @returns {tf.Tensor}
   */
  sample(shape) {
    const s = tf.softplus(this._unconstrainedScale)
    const eps = tf.randomStandardNormal(shape || [])
    return tf.add(this._locVar, tf.mul(s, eps))
  }

  /**
   * Log probability: log N(value; μ, σ)
   * @param {tf.Tensor} value
   * @returns {tf.Tensor}
   */
  logProb(value) {
    const s = tf.softplus(this._unconstrainedScale)
    const z = tf.div(tf.sub(value, this._locVar), s)
    return tf.sub(
      tf.mul(-0.5, tf.add(tf.square(z), LOG_2PI)),
      tf.log(s)
    )
  }

  /**
   * Get current parameter values as JS numbers.
   */
  getParameters() {
    const scale = tf.tidy(() => tf.softplus(this._unconstrainedScale))
    const params = {
      loc: this._locVar.dataSync()[0],
      scale: scale.dataSync()[0]
    }
    scale.dispose()
    return params
  }

  dispose() {
    this._locVar.dispose()
    this._unconstrainedScale.dispose()
  }
}

/**
 * Build a mean-field (independent Normal) surrogate posterior
 * for a multi-parameter state.
 *
 * Creates one trainable Normal per parameter in the initial state.
 *
 * @param {Object} initialState - { paramName: initialValue, ... }
 *   Values should be JS numbers (scalars).
 * @param {Object} [options]
 * @param {number} [options.initialScale=1.0] - initial std dev for all params
 * @returns {{ sample, logProb, trainableVariables, getParameters, dispose }}
 */
export function buildMeanFieldPosterior(initialState, { initialScale = 1.0 } = {}) {
  const keys = Object.keys(initialState)
  const posteriors = {}

  for (const key of keys) {
    posteriors[key] = trainableNormal({
      loc: initialState[key],
      scale: initialScale,
      name: key
    })
  }

  const trainableVariables = keys.flatMap(k => posteriors[k].trainableVariables)

  return {
    trainableVariables,

    /**
     * Sample all parameters independently.
     * @returns {Object} { paramName: tf.Tensor (scalar), ... }
     */
    sample() {
      const result = {}
      for (const key of keys) {
        result[key] = posteriors[key].sample([])
      }
      return result
    },

    /**
     * Compute log q(values) = Σ log q_k(value_k).
     * @param {Object} values - { paramName: tf.Tensor, ... }
     * @returns {tf.Tensor} scalar
     */
    logProb(values) {
      let total = tf.scalar(0)
      for (const key of keys) {
        const lp = posteriors[key].logProb(values[key])
        const newTotal = tf.add(total, lp)
        total.dispose()
        lp.dispose()
        total = newTotal
      }
      return total
    },

    getParameters() {
      const params = {}
      for (const key of keys) {
        params[key] = posteriors[key].getParameters()
      }
      return params
    },

    dispose() {
      for (const key of keys) {
        posteriors[key].dispose()
      }
    }
  }
}
