import * as tf from '@tensorflow/tfjs'
import { stateToArray, arrayToState, cloneState, disposeState, computeGrads } from './state_util'

/**
 * NUTS tree-building utilities.
 *
 * Implements the recursive tree-doubling algorithm from
 * Hoffman & Gelman 2014 (Algorithm 3), with the classic
 * U-turn termination criterion.
 *
 * References:
 *   [1] Hoffman, Gelman. "The No-U-Turn Sampler." JMLR 2014.
 *   [2] Betancourt. "A Conceptual Introduction to HMC." 2018.
 */

/**
 * Check if the trajectory exhibits a U-turn.
 *
 * U-turn detected when:
 *   dot(p_minus, q_plus - q_minus) < 0  OR
 *   dot(p_plus,  q_plus - q_minus) < 0
 *
 * @param {Object|tf.Tensor} stateMinus - leftmost position
 * @param {Object|tf.Tensor} statePlus  - rightmost position
 * @param {Object|tf.Tensor} momentumMinus - leftmost momentum
 * @param {Object|tf.Tensor} momentumPlus  - rightmost momentum
 * @returns {boolean} true if U-turn detected
 */
export function checkUTurn(stateMinus, statePlus, momentumMinus, momentumPlus) {
  return tf.tidy(() => {
    const { values: qMinus } = stateToArray(stateMinus)
    const { values: qPlus } = stateToArray(statePlus)
    const { values: pMinus } = stateToArray(momentumMinus)
    const { values: pPlus } = stateToArray(momentumPlus)

    let dotLeft = 0
    let dotRight = 0

    for (let i = 0; i < qMinus.length; i++) {
      const diff = tf.sub(qPlus[i], qMinus[i])
      dotLeft += tf.sum(tf.mul(pMinus[i], diff)).dataSync()[0]
      dotRight += tf.sum(tf.mul(pPlus[i], diff)).dataSync()[0]
    }

    return dotLeft < 0 || dotRight < 0
  })
}

/**
 * Compute the log joint density: logProb - 0.5 * ||p||²
 *
 * This is the negative Hamiltonian. Used for the slice criterion
 * and divergence check.
 *
 * @param {tf.Tensor} targetLogProb - log probability at current state
 * @param {Object|tf.Tensor} momentum
 * @returns {number} log joint as JS scalar
 */
export function computeLogJoint(targetLogProb, momentum) {
  return tf.tidy(() => {
    const { values: momArr } = stateToArray(momentum)
    let ke = 0
    for (const p of momArr) {
      ke += tf.sum(tf.square(p)).dataSync()[0]
    }
    return targetLogProb.dataSync()[0] - 0.5 * ke
  })
}

/**
 * Take a single leapfrog step without disposing inputs.
 *
 * Uses the leapfrog (Stormer-Verlet) scheme:
 *   p ← p + (ε/2) ∇logπ(q)
 *   q ← q + ε p
 *   compute ∇logπ(q)
 *   p ← p + (ε/2) ∇logπ(q)
 *
 * Requires 1 gradient evaluation (at the new position).
 *
 * @param {Object} params
 * @param {Object|tf.Tensor} params.state - current position q
 * @param {Object|tf.Tensor} params.momentum - current momentum p
 * @param {Object|tf.Tensor} params.grads - ∇logπ at current position
 * @param {number} params.stepSize - ε (signed: negative for backward)
 * @param {Function} params.targetLogProbGradFn - (state) → {value, grads}
 * @returns {{ state, momentum, targetLogProb, grads }}
 */
export function singleLeapfrogStep({ state, momentum, grads, stepSize, targetLogProbGradFn }) {
  const { values: q, keys } = stateToArray(state)
  const { values: p } = stateToArray(momentum)
  const { values: g } = stateToArray(grads)

  const eps = tf.scalar(stepSize)
  const halfEps = tf.scalar(stepSize * 0.5)

  // Half-step momentum: p += (ε/2) · g
  const p1 = p.map((pi, i) => tf.tidy(() => tf.add(pi, tf.mul(halfEps, g[i]))))

  // Full-step position: q += ε · p
  const q1 = q.map((qi, i) => tf.tidy(() => tf.add(qi, tf.mul(eps, p1[i]))))

  // Recompute gradients at new position
  const gradResult = targetLogProbGradFn(arrayToState(q1, keys))
  const g1 = stateToArray(gradResult.grads).values

  // Half-step momentum: p += (ε/2) · g'
  const p2 = p1.map((pi, i) => tf.tidy(() => tf.add(pi, tf.mul(halfEps, g1[i]))))

  // Dispose intermediates
  p1.forEach(t => t.dispose())
  eps.dispose()
  halfEps.dispose()

  return {
    state: arrayToState(q1, keys),
    momentum: arrayToState(p2, keys),
    targetLogProb: gradResult.value,
    grads: arrayToState(g1, keys)
  }
}

/**
 * Build a NUTS subtree recursively (Algorithm 3, BuildTree).
 *
 * At depth 0: takes one leapfrog step and evaluates slice/divergence.
 * At depth j>0: builds two half-trees and combines them.
 *
 * @param {Object} params
 * @param {Object|tf.Tensor} params.state - edge state to extend from
 * @param {Object|tf.Tensor} params.momentum - edge momentum
 * @param {Object|tf.Tensor} params.grads - ∇logπ at edge state
 * @param {number} params.direction - +1 (forward) or -1 (backward)
 * @param {number} params.depth - tree depth j
 * @param {number} params.stepSize - leapfrog step size ε
 * @param {Function} params.targetLogProbGradFn - (state) → {value, grads}
 * @param {number} params.logSliceThreshold - log(u) for slice sampling
 * @param {number} [params.maxEnergyDiff=1000] - divergence threshold
 * @returns {TreeResult} see below
 */
export function buildTree({
  state, momentum, grads, direction, depth, stepSize,
  targetLogProbGradFn, logSliceThreshold, maxEnergyDiff = 1000
}) {
  if (depth === 0) {
    return buildTreeBase({
      state, momentum, grads, direction, stepSize,
      targetLogProbGradFn, logSliceThreshold, maxEnergyDiff
    })
  }

  // Build first half-tree
  const first = buildTree({
    state, momentum, grads, direction, depth: depth - 1, stepSize,
    targetLogProbGradFn, logSliceThreshold, maxEnergyDiff
  })

  if (!first.isValid) {
    return first
  }

  // Build second half-tree from the appropriate edge
  const edgeState = direction === -1 ? first.stateMinus : first.statePlus
  const edgeMomentum = direction === -1 ? first.momentumMinus : first.momentumPlus
  const edgeGrads = direction === -1 ? first.gradsMinus : first.gradsPlus

  const second = buildTree({
    state: edgeState, momentum: edgeMomentum, grads: edgeGrads,
    direction, depth: depth - 1, stepSize,
    targetLogProbGradFn, logSliceThreshold, maxEnergyDiff
  })

  // Combine subtrees
  return combineSubtrees(first, second, direction)
}

/**
 * Base case of buildTree: take one leapfrog step.
 */
function buildTreeBase({
  state, momentum, grads, direction, stepSize,
  targetLogProbGradFn, logSliceThreshold, maxEnergyDiff
}) {
  const step = singleLeapfrogStep({
    state, momentum, grads,
    stepSize: direction * stepSize,
    targetLogProbGradFn
  })

  // Compute log joint (logProb - 0.5 * ||p||²)
  const logJoint = computeLogJoint(step.targetLogProb, step.momentum)

  // Slice criterion: state is valid if logJoint >= logSliceThreshold
  const numValid = logJoint >= logSliceThreshold ? 1 : 0

  // Divergence: energy error exceeds threshold
  const hasDivergence = (logSliceThreshold - logJoint) > maxEnergyDiff
  const isValid = !hasDivergence

  return {
    stateMinus: cloneState(step.state),
    momentumMinus: cloneState(step.momentum),
    gradsMinus: cloneState(step.grads),
    statePlus: cloneState(step.state),
    momentumPlus: cloneState(step.momentum),
    gradsPlus: cloneState(step.grads),
    candidate: step.state,
    candidateLogProb: step.targetLogProb,
    candidateGrads: step.grads,
    numValid,
    isValid,
    numLeapfrogs: 1,
    hasDivergence
  }
}

/**
 * Combine two subtrees into a single tree.
 * Handles candidate selection, endpoint merging, and U-turn checking.
 */
function combineSubtrees(first, second, direction) {
  const totalValid = first.numValid + second.numValid

  // Select candidate: accept second with prob (second.numValid / totalValid)
  let candidate, candidateLogProb, candidateGrads
  if (totalValid > 0 && Math.random() < second.numValid / totalValid) {
    disposeState(first.candidate)
    first.candidateLogProb.dispose()
    disposeState(first.candidateGrads)
    candidate = second.candidate
    candidateLogProb = second.candidateLogProb
    candidateGrads = second.candidateGrads
  } else {
    disposeState(second.candidate)
    second.candidateLogProb.dispose()
    disposeState(second.candidateGrads)
    candidate = first.candidate
    candidateLogProb = first.candidateLogProb
    candidateGrads = first.candidateGrads
  }

  // Merge endpoints based on direction
  let stateMinus, momentumMinus, gradsMinus
  let statePlus, momentumPlus, gradsPlus

  if (direction === -1) {
    // Extended backward: second tree is the new left edge
    stateMinus = second.stateMinus
    momentumMinus = second.momentumMinus
    gradsMinus = second.gradsMinus
    statePlus = first.statePlus
    momentumPlus = first.momentumPlus
    gradsPlus = first.gradsPlus
    // Dispose overwritten endpoints
    disposeState(first.stateMinus)
    disposeState(first.momentumMinus)
    disposeState(first.gradsMinus)
    disposeState(second.statePlus)
    disposeState(second.momentumPlus)
    disposeState(second.gradsPlus)
  } else {
    // Extended forward: second tree is the new right edge
    stateMinus = first.stateMinus
    momentumMinus = first.momentumMinus
    gradsMinus = first.gradsMinus
    statePlus = second.statePlus
    momentumPlus = second.momentumPlus
    gradsPlus = second.gradsPlus
    // Dispose overwritten endpoints
    disposeState(first.statePlus)
    disposeState(first.momentumPlus)
    disposeState(first.gradsPlus)
    disposeState(second.stateMinus)
    disposeState(second.momentumMinus)
    disposeState(second.gradsMinus)
  }

  // Check U-turn on the combined tree
  const hasUTurn = checkUTurn(stateMinus, statePlus, momentumMinus, momentumPlus)
  const isValid = second.isValid && !hasUTurn
  const hasDivergence = first.hasDivergence || second.hasDivergence

  return {
    stateMinus, momentumMinus, gradsMinus,
    statePlus, momentumPlus, gradsPlus,
    candidate, candidateLogProb, candidateGrads,
    numValid: totalValid,
    isValid,
    numLeapfrogs: first.numLeapfrogs + second.numLeapfrogs,
    hasDivergence
  }
}

/**
 * Dispose all tensor fields in a tree result.
 */
export function disposeTreeResult(tree) {
  disposeState(tree.stateMinus)
  disposeState(tree.momentumMinus)
  disposeState(tree.gradsMinus)
  disposeState(tree.statePlus)
  disposeState(tree.momentumPlus)
  disposeState(tree.gradsPlus)
  disposeState(tree.candidate)
  tree.candidateLogProb.dispose()
  disposeState(tree.candidateGrads)
}
