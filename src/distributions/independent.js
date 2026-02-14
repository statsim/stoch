import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'

/**
 * Independent distribution: reinterprets batch dims as event dims.
 *
 * This wraps a base distribution and sums its logProb over the last
 * `reinterpretedBatchNdims` batch dimensions. This is critical for
 * multivariate distributions built from independent components.
 *
 * Example:
 *   // Normal([0,0], [1,1]) has batchShape=[2], logProb returns shape [2]
 *   const ind = new Independent({
 *     distribution: new Normal({ loc: [0,0], scale: [1,1] }),
 *     reinterpretedBatchNdims: 1
 *   })
 *   // ind has batchShape=[], logProb returns scalar (sum of per-element logProbs)
 */
export class Independent extends Distribution {
  constructor({ distribution, reinterpretedBatchNdims = 1, validateArgs, name } = {}) {
    super({
      dtype: distribution.dtype,
      validateArgs: validateArgs != null ? validateArgs : distribution._validateArgs,
      name: name || `Independent${distribution.name}`
    })
    this._distribution = distribution
    this._reinterpretedBatchNdims = reinterpretedBatchNdims
  }

  get distribution() { return this._distribution }
  get reinterpretedBatchNdims() { return this._reinterpretedBatchNdims }

  get batchShape() {
    const baseBatch = this._distribution.batchShape
    return baseBatch.slice(0, baseBatch.length - this._reinterpretedBatchNdims)
  }

  _eventShape() {
    const baseBatch = this._distribution.batchShape
    const baseEvent = this._distribution.eventShape
    const reinterpreted = baseBatch.slice(baseBatch.length - this._reinterpretedBatchNdims)
    return [...reinterpreted, ...baseEvent]
  }

  _sampleN(n) {
    return this._distribution._sampleN(n)
  }

  _logProb(value) {
    const baseLogProb = this._distribution._logProb(value)
    return this._reduceOverReinterpretedDims(baseLogProb)
  }

  _prob(value) {
    // prob = exp(logProb) — use logProb to avoid underflow
    return tf.exp(this._logProb(value))
  }

  _entropy() {
    const baseEntropy = this._distribution._entropy()
    return this._reduceOverReinterpretedDims(baseEntropy)
  }

  _mean() {
    return this._distribution._mean()
  }

  _variance() {
    return this._distribution._variance()
  }

  _mode() {
    return this._distribution._mode()
  }

  /**
   * Sum over the last reinterpretedBatchNdims dims.
   */
  _reduceOverReinterpretedDims(tensor) {
    if (this._reinterpretedBatchNdims === 0) return tensor
    const rank = tensor.shape.length
    const axes = []
    for (let i = rank - this._reinterpretedBatchNdims; i < rank; i++) {
      axes.push(i)
    }
    return tf.sum(tensor, axes)
  }

  dispose() {
    this._distribution.dispose()
  }
}
