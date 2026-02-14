import * as tf from '@tensorflow/tfjs'
import { TransitionKernel } from './kernel'
import { stateToArray, arrayToState, cloneState, disposeState } from './state_util'

/**
 * Random Walk Metropolis-Hastings transition kernel.
 *
 * Gradient-free MCMC. Proposes new states by adding noise to the
 * current state, then accepts/rejects via the Metropolis criterion.
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) → scalar Tensor
 * @param {Function} [params.newStateProposalFn] - (state) → proposed state
 *   Default: adds Gaussian noise with scale 0.1
 * @param {number} [params.proposalScale=0.1] - std dev of default Gaussian proposal
 */
export class RandomWalkMetropolis extends TransitionKernel {
  constructor({ targetLogProbFn, newStateProposalFn, proposalScale = 0.1 }) {
    super()
    this._targetLogProbFn = targetLogProbFn
    this._proposalScale = proposalScale
    this._newStateProposalFn = newStateProposalFn || this._defaultProposal.bind(this)
  }

  _defaultProposal(state) {
    if (state instanceof tf.Tensor) {
      return tf.tidy(() =>
        tf.add(state, tf.randomNormal(state.shape, 0, this._proposalScale))
      )
    }
    const result = {}
    for (const [key, val] of Object.entries(state)) {
      result[key] = tf.tidy(() =>
        tf.add(val, tf.randomNormal(val.shape, 0, this._proposalScale))
      )
    }
    return result
  }

  bootstrapResults(initState) {
    const targetLogProb = tf.tidy(() => this._targetLogProbFn(initState))
    return {
      targetLogProb,
      isAccepted: tf.scalar(1),
      logAcceptRatio: tf.scalar(0)
    }
  }

  oneStep(currentState, previousKernelResults) {
    const currentLogProb = previousKernelResults.targetLogProb

    // Propose new state
    const proposedState = this._newStateProposalFn(currentState)

    // Evaluate target at proposed state
    const proposedLogProb = tf.tidy(() => this._targetLogProbFn(proposedState))

    // Metropolis acceptance
    const logAcceptRatio = tf.tidy(() =>
      tf.sub(proposedLogProb, currentLogProb)
    )

    const logU = tf.tidy(() => tf.log(tf.randomUniform([])))
    const isAccepted = tf.tidy(() => tf.greater(logAcceptRatio, logU))
    const accepted = isAccepted.dataSync()[0] > 0

    let nextState, nextLogProb
    if (accepted) {
      nextState = proposedState
      nextLogProb = proposedLogProb
    } else {
      nextState = cloneState(currentState)
      nextLogProb = currentLogProb.clone()
      disposeState(proposedState)
      proposedLogProb.dispose()
    }

    logU.dispose()

    return {
      nextState,
      kernelResults: {
        targetLogProb: nextLogProb,
        isAccepted,
        logAcceptRatio
      }
    }
  }
}
