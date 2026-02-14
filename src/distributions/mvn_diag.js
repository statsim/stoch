import * as tf from '@tensorflow/tfjs'
import { Normal } from './normal'
import { Independent } from './independent'

/**
 * Multivariate Normal distribution with diagonal covariance.
 *
 * Thin wrapper around Independent(Normal({loc, scale: scaleDiag}), 1).
 * This treats the last batch dimension as the event dimension.
 *
 * Example:
 *   const mvn = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [1, 2] })
 *   mvn.eventShape  // [2]
 *   mvn.sample()    // shape [2]
 *   mvn.logProb([0, 0])  // scalar (joint log probability)
 */
export class MultivariateNormalDiag extends Independent {
  constructor({ loc, scaleDiag, validateArgs, name } = {}) {
    const normal = new Normal({
      loc,
      scale: scaleDiag,
      validateArgs,
      name: 'MultivariateNormalDiag_Normal'
    })
    super({
      distribution: normal,
      reinterpretedBatchNdims: 1,
      validateArgs,
      name: name || 'MultivariateNormalDiag'
    })
  }

  get loc() { return this._distribution.loc }
  get scaleDiag() { return this._distribution.scale }
}
