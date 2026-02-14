import * as tf from '@tensorflow/tfjs'
import { stateToArray, arrayToState, disposeState } from './state_util'

/**
 * Leapfrog (Stormer-Verlet) symplectic integrator.
 *
 * Numerically integrates Hamilton's equations:
 *   dq/dt = ∂H/∂p = p
 *   dp/dt = -∂H/∂q = ∇ log π(q)
 *
 * Uses the "leapfrog" scheme:
 *   1. Half-step momentum:  p ← p + (ε/2) · ∇ log π(q)
 *   2. Full-step position:  q ← q + ε · p
 *   3. Half-step momentum:  p ← p + (ε/2) · ∇ log π(q)
 *
 * Repeats steps 1-3 for numSteps. Steps 3 and 1 of consecutive
 * iterations merge into a single full momentum step.
 *
 * @param {Object} params
 * @param {Object|tf.Tensor} params.currentState - q
 * @param {Object|tf.Tensor} params.momentum - p (same structure as state)
 * @param {number|tf.Tensor} params.stepSize - ε
 * @param {number} params.numSteps - L
 * @param {Function} params.targetLogProbGradFn
 *   (state) → { value: Tensor, grads: state-like }
 * @param {tf.Tensor} [params.currentTargetLogProb] - log π(q) at current state
 * @param {Object|tf.Tensor} [params.currentGrads] - ∇ log π(q) at current state
 * @returns {{ finalState, finalMomentum, finalTargetLogProb, finalGrads }}
 */
export function leapfrogIntegrate({
  currentState,
  momentum,
  stepSize,
  numSteps,
  targetLogProbGradFn,
  currentTargetLogProb,
  currentGrads
}) {
  const { values: stateArr, keys } = stateToArray(currentState)
  let { values: momArr } = stateToArray(momentum)
  let gradsArr = currentGrads
    ? stateToArray(currentGrads).values
    : null

  const eps = typeof stepSize === 'number' ? tf.scalar(stepSize) : stepSize
  const halfEps = tf.mul(eps, 0.5)
  const epsIsOwned = typeof stepSize === 'number'

  // If we don't have initial grads, compute them
  if (!gradsArr) {
    const result = targetLogProbGradFn(arrayToState(stateArr, keys))
    gradsArr = stateToArray(result.grads).values
    if (!currentTargetLogProb) {
      currentTargetLogProb = result.value
    } else {
      result.value.dispose()
    }
  }

  // Working copies (will be mutated during integration)
  let q = stateArr.map(t => t.clone())
  let p = momArr.map(t => t.clone())
  let g = gradsArr.map(t => t.clone())

  // Dispose the input grads (we have working copies now)
  gradsArr.forEach(t => t.dispose())

  for (let step = 0; step < numSteps; step++) {
    // Half-step momentum: p += (ε/2) · ∇ log π(q)
    // Wrap in tidy to dispose intermediate tf.mul results
    const newP1 = p.map((pi, i) => tf.tidy(() => tf.add(pi, tf.mul(halfEps, g[i]))))
    p.forEach(t => t.dispose())
    p = newP1

    // Full-step position: q += ε · p
    const newQ = q.map((qi, i) => tf.tidy(() => tf.add(qi, tf.mul(eps, p[i]))))
    q.forEach(t => t.dispose())
    q = newQ

    // Recompute gradients at new position
    const gradResult = targetLogProbGradFn(arrayToState(q, keys))
    const newG = stateToArray(gradResult.grads).values
    g.forEach(t => t.dispose())
    g = newG

    // Half-step momentum: p += (ε/2) · ∇ log π(q)
    const newP2 = p.map((pi, i) => tf.tidy(() => tf.add(pi, tf.mul(halfEps, g[i]))))
    p.forEach(t => t.dispose())
    p = newP2

    // Dispose old logProb, keep the final one
    if (currentTargetLogProb) currentTargetLogProb.dispose()
    currentTargetLogProb = gradResult.value
  }

  // Clean up eps tensors
  halfEps.dispose()
  if (epsIsOwned) eps.dispose()

  return {
    finalState: arrayToState(q, keys),
    finalMomentum: arrayToState(p, keys),
    finalTargetLogProb: currentTargetLogProb,
    finalGrads: arrayToState(g, keys)
  }
}
