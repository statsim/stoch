import * as tf from '@tensorflow/tfjs'
import { TransitionKernel } from './kernel'
import { leapfrogIntegrate } from './leapfrog'
import {
  stateToArray,
  arrayToState,
  cloneState,
  disposeState,
  computeGrads
} from './state_util'

/**
 * Hamiltonian Monte Carlo (HMC) transition kernel.
 *
 * Uses the leapfrog integrator to propose new states, then
 * accepts/rejects via the Metropolis criterion.
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) → scalar Tensor
 * @param {number|tf.Tensor} params.stepSize - leapfrog step size ε
 * @param {number} params.numLeapfrogSteps - number of leapfrog steps L
 */
export class HamiltonianMonteCarlo extends TransitionKernel {
  constructor({ targetLogProbFn, stepSize = 0.1, numLeapfrogSteps = 3 }) {
    super()
    this._targetLogProbFn = targetLogProbFn
    this._stepSize = stepSize
    this._numLeapfrogSteps = numLeapfrogSteps
  }

  get stepSize() { return this._stepSize }
  get numLeapfrogSteps() { return this._numLeapfrogSteps }

  /**
   * Create initial kernel results by evaluating the target at initState.
   */
  bootstrapResults(initState) {
    const { value, grads } = computeGrads(this._targetLogProbFn, initState)
    return {
      targetLogProb: value,
      gradsTargetLogProb: grads,
      isAccepted: tf.scalar(1), // First state is trivially "accepted"
      logAcceptRatio: tf.scalar(0),
      stepSize: typeof this._stepSize === 'number'
        ? tf.scalar(this._stepSize) : this._stepSize
    }
  }

  /**
   * Take one HMC step:
   * 1. Sample momentum ~ Normal(0, I)
   * 2. Leapfrog integrate
   * 3. Metropolis accept/reject
   */
  oneStep(currentState, previousKernelResults) {
    const { values: stateArr, keys } = stateToArray(currentState)

    // 1. Sample momentum from standard normal
    const momentumArr = stateArr.map(s => tf.randomNormal(s.shape))
    const momentum = arrayToState(momentumArr, keys)

    // Current Hamiltonian: H = -logProb + 0.5 * Σ p²
    const currentLogProb = previousKernelResults.targetLogProb
    const currentKE = this._kineticEnergy(momentumArr)

    // 2. Leapfrog integration
    const targetLogProbGradFn = (state) => computeGrads(this._targetLogProbFn, state)

    const { finalState, finalMomentum, finalTargetLogProb, finalGrads } =
      leapfrogIntegrate({
        currentState,
        momentum,
        stepSize: this._stepSize,
        numSteps: this._numLeapfrogSteps,
        targetLogProbGradFn,
        currentTargetLogProb: currentLogProb.clone(),
        currentGrads: cloneState(previousKernelResults.gradsTargetLogProb)
      })

    // 3. Compute proposed kinetic energy
    const { values: proposedMomArr } = stateToArray(finalMomentum)
    const proposedKE = this._kineticEnergy(proposedMomArr)

    // Metropolis acceptance criterion
    // log(accept_ratio) = (proposed_logProb - proposed_KE) - (current_logProb - current_KE)
    //                    = proposed_logProb - current_logProb - proposed_KE + current_KE
    const logAcceptRatio = tf.tidy(() =>
      tf.sub(
        tf.sub(finalTargetLogProb, currentLogProb),
        tf.sub(proposedKE, currentKE)
      )
    )

    // Accept if log(U) < logAcceptRatio
    const logU = tf.tidy(() => tf.log(tf.randomUniform([])))
    const isAccepted = tf.tidy(() => tf.greater(logAcceptRatio, logU))

    // Select next state based on acceptance
    const accepted = isAccepted.dataSync()[0] > 0
    let nextState, nextLogProb, nextGrads

    if (accepted) {
      nextState = finalState
      nextLogProb = finalTargetLogProb
      nextGrads = finalGrads
      // Don't need to clone — finalState already owns these tensors
    } else {
      nextState = cloneState(currentState)
      nextLogProb = currentLogProb.clone()
      nextGrads = cloneState(previousKernelResults.gradsTargetLogProb)
      // Dispose rejected proposal
      disposeState(finalState)
      finalTargetLogProb.dispose()
      disposeState(finalGrads)
    }

    // Dispose momentum and intermediate tensors
    disposeState(momentum)
    disposeState(finalMomentum)
    currentKE.dispose()
    proposedKE.dispose()
    logU.dispose()

    return {
      nextState,
      kernelResults: {
        targetLogProb: nextLogProb,
        gradsTargetLogProb: nextGrads,
        isAccepted,
        logAcceptRatio,
        stepSize: typeof this._stepSize === 'number'
          ? tf.scalar(this._stepSize) : this._stepSize.clone()
      }
    }
  }

  /**
   * Compute kinetic energy: 0.5 * Σ p²
   */
  _kineticEnergy(momentumArr) {
    return tf.tidy(() => {
      let total = tf.scalar(0)
      for (const p of momentumArr) {
        total = tf.add(total, tf.sum(tf.square(p)))
      }
      return tf.mul(0.5, total)
    })
  }
}
