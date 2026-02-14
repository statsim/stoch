import * as tf from '@tensorflow/tfjs'
import { stateToArray, cloneState, disposeState } from './state_util'

/**
 * Run an MCMC sampling chain.
 *
 * Manages burn-in, thinning, and tensor lifecycle.
 *
 * @param {Object} params
 * @param {number} params.numResults - number of samples to collect
 * @param {number} [params.numBurninSteps=0] - steps to discard before collecting
 * @param {Object|tf.Tensor} params.currentState - initial state
 * @param {TransitionKernel} params.kernel - MCMC kernel
 * @param {number} [params.numStepsBetweenResults=0] - thinning (skip N steps between samples)
 * @param {Function} [params.traceFn] - (state, kr) → any, extracts diagnostics as JS values
 * @returns {{ samples: Object|tf.Tensor, trace: Array }}
 *   samples: if state is a tensor, stacked tensor [numResults, ...shape]
 *            if state is an object, object of stacked tensors
 *   trace: array of traceFn results (one per collected sample)
 */
export function sampleChain({
  numResults,
  numBurninSteps = 0,
  currentState,
  kernel,
  numStepsBetweenResults = 0,
  traceFn
}) {
  let state = cloneState(currentState)
  let kr = kernel.bootstrapResults(state)

  const collectedStates = []
  const traceResults = []

  // Helper to dispose kernel results
  function disposeKR(kernelResults) {
    if (kernelResults.targetLogProb) kernelResults.targetLogProb.dispose()
    if (kernelResults.gradsTargetLogProb) {
      disposeState(kernelResults.gradsTargetLogProb)
    }
    if (kernelResults.isAccepted) kernelResults.isAccepted.dispose()
    if (kernelResults.logAcceptRatio) kernelResults.logAcceptRatio.dispose()
    if (kernelResults.stepSize && kernelResults.stepSize instanceof tf.Tensor) {
      kernelResults.stepSize.dispose()
    }
  }

  // Burn-in phase: run kernel, dispose intermediates
  for (let i = 0; i < numBurninSteps; i++) {
    const { nextState, kernelResults } = kernel.oneStep(state, kr)
    disposeState(state)
    disposeKR(kr)
    state = nextState
    kr = kernelResults
  }

  // Sampling phase
  for (let i = 0; i < numResults; i++) {
    // Take one step
    const { nextState, kernelResults } = kernel.oneStep(state, kr)
    disposeState(state)
    disposeKR(kr)
    state = nextState
    kr = kernelResults

    // Thinning: take additional steps, disposing intermediates
    for (let j = 0; j < numStepsBetweenResults; j++) {
      const { nextState: ns, kernelResults: nkr } = kernel.oneStep(state, kr)
      disposeState(state)
      disposeKR(kr)
      state = ns
      kr = nkr
    }

    // Collect this sample (clone to preserve it)
    collectedStates.push(cloneState(state))

    // Extract trace diagnostics as JS values (not tensors)
    if (traceFn) {
      traceResults.push(traceFn(state, kr))
    }
  }

  // Dispose the final working state and kernel results
  disposeState(state)
  disposeKR(kr)

  // Stack collected samples into tensors
  const samples = stackSamples(collectedStates)

  return { samples, trace: traceResults }
}

/**
 * Stack an array of state objects/tensors into a single stacked structure.
 * Disposes the individual collected states after stacking.
 */
function stackSamples(collectedStates) {
  if (collectedStates.length === 0) return null

  const first = collectedStates[0]

  if (first instanceof tf.Tensor) {
    // Single tensor state: stack into [numResults, ...shape]
    const stacked = tf.stack(collectedStates)
    collectedStates.forEach(s => s.dispose())
    return stacked
  }

  // Object state: stack each key separately
  const keys = Object.keys(first)
  const result = {}

  for (const key of keys) {
    const tensors = collectedStates.map(s => s[key])
    result[key] = tf.stack(tensors)
  }

  // Dispose individual states
  collectedStates.forEach(s => disposeState(s))

  return result
}
