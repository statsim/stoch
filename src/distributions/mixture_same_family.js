import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'

/**
 * Mixture of distributions from the same family.
 *
 * p(x) = Σ_k π_k * p_k(x)
 *
 * where π_k are mixture weights (from mixtureDist, typically Categorical)
 * and p_k(x) are component distributions (from componentDist, with batch dim = K).
 *
 * Example:
 *   new MixtureSameFamily({
 *     mixtureDist: new Categorical({ logits: [0, 0, 0] }),   // 3 components
 *     componentDist: new Normal({ loc: [-1, 0, 1], scale: [0.5, 0.5, 0.5] })
 *   })
 */
export class MixtureSameFamily extends Distribution {
  constructor({ mixtureDist, componentDist, validateArgs, name } = {}) {
    super({
      dtype: componentDist.dtype,
      validateArgs: validateArgs != null ? validateArgs : false,
      name: name || 'MixtureSameFamily'
    })
    this._mixtureDist = mixtureDist
    this._componentDist = componentDist
    this._numComponents = mixtureDist.numCategories
  }

  get mixtureDist() { return this._mixtureDist }
  get componentDist() { return this._componentDist }
  get numComponents() { return this._numComponents }

  get batchShape() {
    // Batch shape excludes the mixture component dimension
    const compBatch = this._componentDist.batchShape
    // The last batch dim of componentDist is the number of components
    return compBatch.slice(0, compBatch.length - 1)
  }

  _eventShape() {
    return this._componentDist.eventShape
  }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]

    // 1. Sample component indices from mixture distribution
    const indices = this._mixtureDist._sampleN(n) // [n, ...batchShape], values in [0, K)

    // 2. Sample from all components
    const allSamples = this._componentDist._sampleN(n) // [n, ...batchShape, K, ...eventShape]

    // 3. Select samples based on indices
    // For scalar event shape, allSamples is [n, K] and indices is [n]
    const K = this._numComponents
    const eventShape = this.eventShape
    const totalN = shape.reduce((a, b) => a * b, 1)

    const flatSamples = allSamples.reshape([totalN, K, ...eventShape])
    const flatIndices = indices.reshape([totalN]).cast('int32')

    // Batch-gather: for each sample i, pick flatSamples[i, flatIndices[i]]
    const batchIdx = tf.range(0, totalN, 1, 'int32')
    const gatherIdx = tf.stack([batchIdx, flatIndices], 1) // [totalN, 2]
    const gathered = tf.gatherND(flatSamples, gatherIdx) // [totalN, ...eventShape]

    if (eventShape.length === 0) {
      return gathered.reshape(shape)
    }
    return gathered.reshape([...shape, ...eventShape])
  }

  _logProb(value) {
    // logprob = logSumExp(log_π_k + logProb_k(x))
    const K = this._numComponents

    // Get log mixture weights
    const logPi = this._mixtureDist._logits
      ? tf.logSoftmax(this._mixtureDist._logits)
      : tf.log(this._mixtureDist.probs) // [K] or [...batchShape, K]

    // Evaluate each component's logProb
    // value shape: [...batchShape, ...eventShape]
    // Need to expand value for each component
    const eventNdims = this.eventShape.length

    // Expand value to evaluate against all components
    let expandedValue
    if (eventNdims === 0) {
      // Scalar event: value is [...batchShape], need [...batchShape, 1] then broadcast
      expandedValue = value.expandDims(-1) // [...batchShape, 1]
    } else {
      // Vector event: insert component dim before event dims
      const rank = value.shape.length
      expandedValue = value.expandDims(rank - eventNdims)
    }

    const componentLogProbs = this._componentDist._logProb(expandedValue) // [...batchShape, K]

    // logSumExp(log_π_k + logProb_k(x))
    const logWeighted = tf.add(logPi, componentLogProbs) // [...batchShape, K]
    return tf.logSumExp(logWeighted, -1) // [...batchShape]
  }

  _mean() {
    // E[X] = Σ π_k * μ_k
    const probs = this._mixtureDist.probs // [K] or [...batchShape, K]
    const compMeans = this._componentDist._mean() // [...batchShape, K, ...eventShape]

    if (this.eventShape.length === 0) {
      return tf.sum(tf.mul(probs, compMeans), -1)
    }
    // For vector events, need to broadcast probs over event dims
    const expandedProbs = probs.expandDims(-1) // [...batchShape, K, 1]
    return tf.sum(tf.mul(expandedProbs, compMeans), -2)
  }

  _variance() {
    // Var[X] = Σ π_k * (σ²_k + μ_k²) - (Σ π_k * μ_k)²
    // = Σ π_k * σ²_k + Σ π_k * μ_k² - (Σ π_k * μ_k)²
    const probs = this._mixtureDist.probs
    const compMeans = this._componentDist._mean()
    const compVars = this._componentDist._variance()

    if (this.eventShape.length === 0) {
      const mixMean = tf.sum(tf.mul(probs, compMeans), -1)
      const meanOfVar = tf.sum(tf.mul(probs, compVars), -1)
      const varOfMean = tf.sub(
        tf.sum(tf.mul(probs, tf.square(compMeans)), -1),
        tf.square(mixMean)
      )
      return tf.add(meanOfVar, varOfMean)
    }
    // For vector events
    const expandedProbs = probs.expandDims(-1)
    const mixMean = tf.sum(tf.mul(expandedProbs, compMeans), -2)
    const meanOfVar = tf.sum(tf.mul(expandedProbs, compVars), -2)
    const varOfMean = tf.sub(
      tf.sum(tf.mul(expandedProbs, tf.square(compMeans)), -2),
      tf.square(mixMean)
    )
    return tf.add(meanOfVar, varOfMean)
  }

  dispose() {
    this._mixtureDist.dispose()
    this._componentDist.dispose()
  }
}
