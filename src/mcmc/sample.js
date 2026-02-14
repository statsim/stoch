import * as tf from '@tensorflow/tfjs'
import { NoUTurnSampler } from './nuts'
import { HamiltonianMonteCarlo } from './hmc'
import { DualAveragingStepSizeAdaptation } from './dual_averaging'
import { TransformedTransitionKernel } from './transformed_kernel'
import { sampleChain } from './sample_chain'
import { effectiveSampleSize, potentialScaleReduction } from './diagnostics'
import { cloneState, disposeState } from './state_util'

/**
 * High-level MCMC sampling function with auto-NUTS.
 *
 * Wraps the low-level MCMC API into a single call that:
 *   1. Configures NUTS (or HMC) with step size adaptation
 *   2. Optionally wraps with bijectors for constrained parameters
 *   3. Runs multiple chains
 *   4. Returns samples with convergence diagnostics
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) => tf.Tensor scalar
 * @param {Object|tf.Tensor} params.initialState - starting point for sampling
 * @param {number} [params.numResults=1000] - samples to collect per chain
 * @param {number} [params.numBurninSteps=500] - warmup/burn-in steps
 * @param {number} [params.numChains=1] - number of independent chains
 * @param {number} [params.stepSize=0.1] - initial step size
 * @param {string} [params.kernel='nuts'] - 'nuts' or 'hmc'
 * @param {number} [params.maxTreeDepth=10] - NUTS max tree depth
 * @param {number} [params.numLeapfrogSteps=10] - HMC leapfrog steps (if kernel='hmc')
 * @param {Object} [params.bijectors] - map of parameter name → Bijector for constrained params
 * @param {number} [params.numAdaptationSteps] - defaults to numBurninSteps
 * @param {number} [params.targetAcceptProb=0.8] - target acceptance rate
 * @param {number} [params.numStepsBetweenResults=0] - thinning
 * @param {Function} [params.traceFn] - (state, kr) => any, extracts diagnostics
 * @returns {{ samples, diagnostics }}
 *   samples: stacked tensor or object of stacked tensors per chain
 *   diagnostics: { ess, rhat } per parameter (if multi-chain)
 */
export function sample({
  targetLogProbFn,
  initialState,
  numResults = 1000,
  numBurninSteps = 500,
  numChains = 1,
  stepSize = 0.1,
  kernel: kernelType = 'nuts',
  maxTreeDepth = 10,
  numLeapfrogSteps = 10,
  bijectors,
  numAdaptationSteps,
  targetAcceptProb = 0.8,
  numStepsBetweenResults = 0,
  traceFn
}) {
  const adaptSteps = numAdaptationSteps !== undefined
    ? numAdaptationSteps
    : numBurninSteps

  const chainResults = []
  const chainTraces = []

  for (let c = 0; c < numChains; c++) {
    // Build the inner kernel
    let innerKernel
    if (kernelType === 'nuts') {
      innerKernel = new NoUTurnSampler({
        targetLogProbFn,
        stepSize,
        maxTreeDepth
      })
    } else {
      innerKernel = new HamiltonianMonteCarlo({
        targetLogProbFn,
        stepSize,
        numLeapfrogSteps
      })
    }

    // Optionally wrap with bijectors
    if (bijectors) {
      innerKernel = new TransformedTransitionKernel({
        innerKernel,
        bijectors
      })
    }

    // Wrap with step size adaptation
    const adaptedKernel = new DualAveragingStepSizeAdaptation({
      innerKernel,
      numAdaptationSteps: adaptSteps,
      targetAcceptProb
    })

    // Clone initial state for this chain
    const chainState = c === 0
      ? cloneState(initialState)
      : jitterState(initialState)

    // Build trace function: always track divergences for NUTS
    const innerTraceFn = traceFn || (kernelType === 'nuts'
      ? (_state, kr) => {
        // Dig through adapter and wrapper layers to get NUTS results
        let inner = kr
        while (inner.innerResults) inner = inner.innerResults
        return {
          hasDivergence: !!inner.hasDivergence,
          reachMaxDepth: !!inner.reachMaxDepth,
          leapfrogsTaken: inner.leapfrogsTaken || 0
        }
      }
      : undefined
    )

    const { samples: chainSamples, trace } = sampleChain({
      numResults,
      numBurninSteps,
      currentState: chainState,
      kernel: adaptedKernel,
      numStepsBetweenResults,
      traceFn: innerTraceFn
    })

    chainResults.push(chainSamples)
    if (trace && trace.length > 0) chainTraces.push(trace)

    // Dispose the cloned initial state
    disposeState(chainState)
  }

  // Compute diagnostics
  const diagnostics = computeDiagnostics(chainResults, numChains)

  // Summarize divergences from traces
  if (chainTraces.length > 0) {
    let numDivergent = 0
    let numMaxDepth = 0
    let totalLeapfrogs = 0
    let totalSteps = 0
    for (const trace of chainTraces) {
      for (const t of trace) {
        if (t.hasDivergence) numDivergent++
        if (t.reachMaxDepth) numMaxDepth++
        totalLeapfrogs += t.leapfrogsTaken || 0
        totalSteps++
      }
    }
    diagnostics.numDivergent = numDivergent
    diagnostics.numMaxDepth = numMaxDepth
    diagnostics.meanLeapfrogs = totalSteps > 0 ? totalLeapfrogs / totalSteps : 0
  }

  // If single chain, return samples directly; if multi-chain, return array
  const samples = numChains === 1 ? chainResults[0] : chainResults

  return {
    samples,
    diagnostics,
    trace: chainTraces.length > 0 ? chainTraces : undefined
  }
}

/**
 * Jitter initial state slightly to get diverse chains.
 */
function jitterState(state) {
  if (state instanceof tf.Tensor) {
    return tf.add(state, tf.mul(tf.randomNormal(state.shape), 0.1))
  }
  const result = {}
  for (const [key, val] of Object.entries(state)) {
    result[key] = tf.add(val, tf.mul(tf.randomNormal(val.shape), 0.1))
  }
  return result
}

/**
 * Compute ESS and R-hat diagnostics from chain results.
 */
function computeDiagnostics(chainResults, numChains) {
  if (numChains < 1) return {}

  const first = chainResults[0]

  if (first instanceof tf.Tensor) {
    // Single parameter
    const chains = chainResults.map(s => Array.from(s.dataSync()))
    const ess = effectiveSampleSize(chains[0])
    const rhat = numChains >= 2 ? potentialScaleReduction(chains) : NaN
    return { ess, rhat }
  }

  // Object state: compute per-parameter
  const result = {}
  for (const key of Object.keys(first)) {
    const chains = chainResults.map(s => Array.from(s[key].dataSync()))
    const ess = effectiveSampleSize(chains[0])
    const rhat = numChains >= 2 ? potentialScaleReduction(chains) : NaN
    result[key] = { ess, rhat }
  }
  return result
}
