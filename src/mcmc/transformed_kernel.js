import * as tf from '@tensorflow/tfjs'
import { TransitionKernel } from './kernel'
import { stateToArray, arrayToState, disposeState } from './state_util'

/**
 * TransformedTransitionKernel: bijector↔MCMC bridge.
 *
 * Wraps an inner kernel to sample in unconstrained space while the
 * user's targetLogProbFn operates in constrained space. Automatically
 * applies bijector.inverse (unconstrained → constrained) and adds
 * the log-det-Jacobian correction.
 *
 * Example:
 *   new TransformedTransitionKernel({
 *     innerKernel: new HamiltonianMonteCarlo({
 *       targetLogProbFn: constrainedLogProb,
 *       stepSize: 0.1,
 *       numLeapfrogSteps: 3
 *     }),
 *     bijectors: { sigma: new Exp() }  // sigma > 0
 *   })
 *
 * The inner kernel samples log_sigma in R, then exp(log_sigma)
 * is passed to constrainedLogProb with the ILDJ correction.
 *
 * @param {Object} params
 * @param {TransitionKernel} params.innerKernel
 * @param {Object|Bijector} params.bijectors
 *   If state is a single tensor: a single bijector
 *   If state is an object: { paramName: bijector, ... }
 *   Only constrained parameters need bijectors; unconstrained ones
 *   are passed through unchanged.
 */
export class TransformedTransitionKernel extends TransitionKernel {
  constructor({ innerKernel, bijectors }) {
    super()
    this._innerKernel = innerKernel
    this._bijectors = bijectors
  }

  get innerKernel() { return this._innerKernel }

  /**
   * Transform constrained state → unconstrained for inner kernel.
   */
  _toUnconstrained(constrainedState) {
    if (constrainedState instanceof tf.Tensor) {
      // Single-tensor state
      return this._bijectors.inverse(constrainedState)
    }
    const result = {}
    for (const [key, val] of Object.entries(constrainedState)) {
      if (this._bijectors[key]) {
        result[key] = this._bijectors[key].inverse(val)
      } else {
        result[key] = val.clone()
      }
    }
    return result
  }

  /**
   * Transform unconstrained state → constrained.
   */
  _toConstrained(unconstrainedState) {
    if (unconstrainedState instanceof tf.Tensor) {
      return this._bijectors.forward(unconstrainedState)
    }
    const result = {}
    for (const [key, val] of Object.entries(unconstrainedState)) {
      if (this._bijectors[key]) {
        result[key] = this._bijectors[key].forward(val)
      } else {
        result[key] = val.clone()
      }
    }
    return result
  }

  /**
   * Compute sum of inverse log-det-Jacobian corrections.
   */
  _ildj(unconstrainedState) {
    return tf.tidy(() => {
      if (unconstrainedState instanceof tf.Tensor) {
        return this._bijectors.forwardLogDetJacobian(unconstrainedState, 0)
      }
      let total = tf.scalar(0)
      for (const [key, val] of Object.entries(unconstrainedState)) {
        if (this._bijectors[key]) {
          const ldj = this._bijectors[key].forwardLogDetJacobian(val, 0)
          total = tf.add(total, ldj)
        }
      }
      return total
    })
  }

  /**
   * Temporarily swap the inner kernel's targetLogProbFn with a version
   * that transforms unconstrained → constrained + adds the LDJ correction,
   * run the callback, then restore the original.
   */
  _withWrappedLogProb(callback) {
    const originalTargetLogProbFn = this._innerKernel._targetLogProbFn
    this._innerKernel._targetLogProbFn = (unconstrState) => {
      return tf.tidy(() => {
        const constrained = this._toConstrained(unconstrState)
        const logProb = originalTargetLogProbFn(constrained)
        const ldj = this._ildj(unconstrState)
        disposeState(constrained)
        return tf.add(logProb, ldj)
      })
    }
    const result = callback()
    this._innerKernel._targetLogProbFn = originalTargetLogProbFn
    return result
  }

  bootstrapResults(initState) {
    // Transform to unconstrained space
    const unconstrainedState = this._toUnconstrained(initState)

    // Bootstrap with the wrapped log prob so gradients are correct
    const innerKR = this._withWrappedLogProb(() =>
      this._innerKernel.bootstrapResults(unconstrainedState)
    )
    disposeState(unconstrainedState)

    return {
      innerResults: innerKR,
      unconstrainedState: this._toUnconstrained(initState)
    }
  }

  oneStep(currentState, previousKernelResults) {
    const unconstrainedState = previousKernelResults.unconstrainedState

    const { nextState: nextUnconstrained, kernelResults: innerKR } =
      this._withWrappedLogProb(() =>
        this._innerKernel.oneStep(unconstrainedState, previousKernelResults.innerResults)
      )

    // Transform back to constrained space
    const nextConstrained = this._toConstrained(nextUnconstrained)

    // Dispose old unconstrained state
    disposeState(previousKernelResults.unconstrainedState)

    return {
      nextState: nextConstrained,
      kernelResults: {
        innerResults: innerKR,
        unconstrainedState: nextUnconstrained
      }
    }
  }
}
