import * as tf from '@tensorflow/tfjs'

/**
 * Joint distribution specified as a sequential list.
 *
 * Each element is a function that returns a distribution, optionally
 * conditioned on earlier variables. Dependencies are passed as positional
 * arguments in reverse order (most recent first), following TFP convention.
 *
 * Example:
 *   const model = new JointDistributionSequential([
 *     () => new Normal({ loc: 0, scale: 10 }),           // [0] mu
 *     () => new LogNormal({ loc: 0, scale: 1 }),         // [1] sigma
 *     (sigma, mu) => new Normal({ loc: mu, scale: sigma }) // [2] y
 *   ])
 *
 *   model.sample()    // [Tensor, Tensor, Tensor]
 *   model.logProb([muVal, sigmaVal, yVal])  // scalar Tensor
 */
export class JointDistributionSequential {
  constructor(model, { validateArgs, name } = {}) {
    this._name = name || 'JointDistributionSequential'
    this._validateArgs = validateArgs != null ? validateArgs : false

    if (!Array.isArray(model)) {
      throw new Error('JointDistributionSequential expects an array of functions')
    }
    this._model = model
    this._numVars = model.length
  }

  get name() { return this._name }
  get numVariables() { return this._numVars }

  /**
   * Determine how many args each function expects.
   * This tells us how many preceding variables it depends on.
   */
  _numDeps(idx) {
    return this._model[idx].length
  }

  /**
   * Draw samples from the joint distribution.
   * Returns array of Tensors.
   */
  sample(shape) {
    if (!shape) shape = []
    if (typeof shape === 'number') shape = [shape]
    const n = shape.reduce((a, b) => a * b, 1) || 1

    const samples = []
    const tempDists = []

    for (let i = 0; i < this._numVars; i++) {
      const fn = this._model[i]
      const numDeps = this._numDeps(i)
      const isRoot = numDeps === 0

      // Collect dependencies in reverse order (most recent first)
      const args = []
      for (let j = 0; j < numDeps; j++) {
        args.push(samples[i - 1 - j])
      }

      const dist = fn(...args)
      tempDists.push(dist)
      samples.push(this._sampleFromDist(dist, isRoot ? shape : [], isRoot ? n : 1))
    }

    // Dispose temporary distributions
    for (const dist of tempDists) {
      if (dist.dispose) dist.dispose()
    }

    return samples
  }

  /**
   * Sample from a single component distribution.
   */
  _sampleFromDist(dist, shape, n) {
    return tf.tidy(() => {
      const raw = dist._sampleN(n)
      if (n === 1 && shape.length === 0) {
        return raw.reshape(raw.shape.slice(1))
      }
      if (shape.length > 0) {
        return raw.reshape([...shape, ...dist.batchShape, ...dist.eventShape])
      }
      return raw
    })
  }

  /**
   * Joint log probability: sum of all component log probabilities.
   * @param {Array} values - array of Tensor|number, one per variable
   * @returns {tf.Tensor} scalar
   */
  logProb(values) {
    return tf.tidy(() => {
      const parts = this._logProbPartsInternal(values)
      return parts.reduce((sum, lp) => tf.add(sum, lp))
    })
  }

  /**
   * Per-component log probabilities.
   * @param {Array} values - array of Tensor|number, one per variable
   * @returns {Array<tf.Tensor>}
   */
  logProbParts(values) {
    return this._logProbPartsInternal(values)
  }

  /**
   * Internal: compute per-component logProbs.
   */
  _logProbPartsInternal(values) {
    const parts = []
    const tempDists = []

    for (let i = 0; i < this._numVars; i++) {
      const fn = this._model[i]
      const numDeps = this._numDeps(i)

      // Collect dependencies in reverse order
      const args = []
      for (let j = 0; j < numDeps; j++) {
        args.push(values[i - 1 - j])
      }

      const dist = fn(...args)
      tempDists.push(dist)
      parts.push(dist.logProb(values[i]))
    }

    // Dispose temporary distributions
    for (const dist of tempDists) {
      if (dist.dispose) dist.dispose()
    }

    return parts
  }

  dispose() {
    // No persistent tensors
  }
}
