import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Chain bijector: composition of multiple bijectors.
 *
 * Applies bijectors right-to-left in forward direction (last element first),
 * left-to-right in inverse direction. This matches TFP Python convention.
 *
 * Example:
 *   new Chain({ bijectors: [Exp(), Scale({scale: 2})] })
 *   // forward: x → Scale(x) → Exp(Scale(x)) = exp(2x)
 *   // inverse: y → Exp⁻¹(y) → Scale⁻¹(Exp⁻¹(y)) = log(y)/2
 *
 * FLDJ: sum of individual FLDJs along the chain.
 */
export class Chain extends Bijector {
  constructor({ bijectors, validateArgs, name } = {}) {
    const allConstant = bijectors.every(b => b.isConstantJacobian)
    super({
      isConstantJacobian: allConstant,
      validateArgs,
      name: name || 'Chain'
    })
    this._bijectors = bijectors
  }

  get bijectors() { return this._bijectors }

  _forward(x) {
    let result = x
    // Apply right-to-left
    for (let i = this._bijectors.length - 1; i >= 0; i--) {
      result = this._bijectors[i]._forward(result)
    }
    return result
  }

  _inverse(y) {
    let result = y
    // Apply left-to-right
    for (let i = 0; i < this._bijectors.length; i++) {
      result = this._bijectors[i]._inverse(result)
    }
    return result
  }

  _forwardLogDetJacobian(x) {
    let totalLdj = null
    let current = x
    // Apply right-to-left, accumulating FLDJ at each step
    for (let i = this._bijectors.length - 1; i >= 0; i--) {
      const ldj = this._bijectors[i]._forwardLogDetJacobian(current)
      totalLdj = totalLdj === null ? ldj : tf.add(totalLdj, ldj)
      current = this._bijectors[i]._forward(current)
    }
    return totalLdj || tf.scalar(0)
  }

  _inverseLogDetJacobian(y) {
    let totalLdj = null
    let current = y
    // Apply left-to-right, accumulating ILDJ at each step
    for (let i = 0; i < this._bijectors.length; i++) {
      const ldj = this._bijectors[i]._inverseLogDetJacobian(current)
      totalLdj = totalLdj === null ? ldj : tf.add(totalLdj, ldj)
      current = this._bijectors[i]._inverse(current)
    }
    return totalLdj || tf.scalar(0)
  }

  dispose() {
    for (const b of this._bijectors) {
      if (b.dispose) b.dispose()
    }
  }
}
