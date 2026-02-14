import * as tf from '@tensorflow/tfjs'
import { TransitionKernel } from './kernel'
import {
  stateToArray,
  arrayToState,
  cloneState,
  disposeState,
  computeGrads
} from './state_util'
import { buildTree, checkUTurn, computeLogJoint, disposeTreeResult } from './nuts_util'

/**
 * No-U-Turn Sampler (NUTS) transition kernel.
 *
 * Implements Algorithm 3 from Hoffman & Gelman (2014) with
 * slice sampling and the classic U-turn criterion.
 *
 * NUTS adaptively selects trajectory length by building a balanced
 * binary tree of leapfrog steps and stopping when the trajectory
 * makes a U-turn. This eliminates the need to manually tune
 * numLeapfrogSteps as in HMC.
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) → scalar Tensor
 * @param {number} params.stepSize - leapfrog step size ε
 * @param {number} [params.maxTreeDepth=10] - maximum tree depth (2^d max steps)
 * @param {number} [params.maxEnergyDiff=1000] - divergence threshold
 */
export class NoUTurnSampler extends TransitionKernel {
  constructor({ targetLogProbFn, stepSize = 0.1, maxTreeDepth = 10, maxEnergyDiff = 1000 }) {
    super()
    this._targetLogProbFn = targetLogProbFn
    this._stepSize = stepSize
    this._maxTreeDepth = maxTreeDepth
    this._maxEnergyDiff = maxEnergyDiff
  }

  get stepSize() { return this._stepSize }
  get maxTreeDepth() { return this._maxTreeDepth }

  /**
   * Create initial kernel results.
   */
  bootstrapResults(initState) {
    const { value, grads } = computeGrads(this._targetLogProbFn, initState)
    return {
      targetLogProb: value,
      gradsTargetLogProb: grads,
      isAccepted: tf.scalar(1),
      logAcceptRatio: tf.scalar(0),
      stepSize: tf.scalar(this._stepSize),
      leapfrogsTaken: 0,
      reachMaxDepth: false,
      hasDivergence: false
    }
  }

  /**
   * Take one NUTS step (Algorithm 3).
   *
   * 1. Sample momentum ~ N(0, I)
   * 2. Set slice variable u ~ Uniform(0, exp(logJoint))
   * 3. Build tree by doubling: randomly extend forward or backward
   * 4. Stop on U-turn, divergence, or max depth
   * 5. Return candidate from balanced tree
   */
  oneStep(currentState, previousKernelResults) {
    const { values: stateArr, keys } = stateToArray(currentState)
    const targetLogProbGradFn = (state) => computeGrads(this._targetLogProbFn, state)

    // 1. Sample momentum
    const momentumArr = stateArr.map(s => tf.randomNormal(s.shape))
    const momentum = arrayToState(momentumArr, keys)

    // 2. Compute initial log joint and slice variable
    const currentLogProb = previousKernelResults.targetLogProb
    const initLogJoint = computeLogJoint(currentLogProb, momentum)
    const logSliceThreshold = initLogJoint - Math.abs(tf.tidy(() =>
      tf.log(tf.randomUniform([])).dataSync()[0]
    ))

    // 3. Initialize tree
    let tree = {
      stateMinus: cloneState(currentState),
      momentumMinus: cloneState(momentum),
      gradsMinus: cloneState(previousKernelResults.gradsTargetLogProb),
      statePlus: cloneState(currentState),
      momentumPlus: cloneState(momentum),
      gradsPlus: cloneState(previousKernelResults.gradsTargetLogProb),
      candidate: cloneState(currentState),
      candidateLogProb: currentLogProb.clone(),
      candidateGrads: cloneState(previousKernelResults.gradsTargetLogProb),
      numValid: 1,
      isValid: true,
      numLeapfrogs: 0,
      hasDivergence: false
    }

    let depth = 0

    // 4. Tree doubling loop
    while (tree.isValid && depth < this._maxTreeDepth) {
      // Choose random direction
      const direction = Math.random() < 0.5 ? -1 : 1

      // Extend tree in chosen direction
      const edgeState = direction === -1 ? tree.stateMinus : tree.statePlus
      const edgeMomentum = direction === -1 ? tree.momentumMinus : tree.momentumPlus
      const edgeGrads = direction === -1 ? tree.gradsMinus : tree.gradsPlus

      const subtree = buildTree({
        state: edgeState,
        momentum: edgeMomentum,
        grads: edgeGrads,
        direction,
        depth,
        stepSize: this._stepSize,
        targetLogProbGradFn,
        logSliceThreshold,
        maxEnergyDiff: this._maxEnergyDiff
      })

      // Update candidate with probability (subtree.numValid / total)
      if (subtree.isValid) {
        const total = tree.numValid + subtree.numValid
        if (Math.random() < subtree.numValid / total) {
          disposeState(tree.candidate)
          tree.candidateLogProb.dispose()
          disposeState(tree.candidateGrads)
          tree.candidate = cloneState(subtree.candidate)
          tree.candidateLogProb = subtree.candidateLogProb.clone()
          tree.candidateGrads = cloneState(subtree.candidateGrads)
        }
        tree.numValid += subtree.numValid
      }

      // Merge endpoints
      if (direction === -1) {
        disposeState(tree.stateMinus)
        disposeState(tree.momentumMinus)
        disposeState(tree.gradsMinus)
        tree.stateMinus = cloneState(subtree.stateMinus)
        tree.momentumMinus = cloneState(subtree.momentumMinus)
        tree.gradsMinus = cloneState(subtree.gradsMinus)
      } else {
        disposeState(tree.statePlus)
        disposeState(tree.momentumPlus)
        disposeState(tree.gradsPlus)
        tree.statePlus = cloneState(subtree.statePlus)
        tree.momentumPlus = cloneState(subtree.momentumPlus)
        tree.gradsPlus = cloneState(subtree.gradsPlus)
      }

      // Check U-turn on full tree
      const hasUTurn = checkUTurn(
        tree.stateMinus, tree.statePlus,
        tree.momentumMinus, tree.momentumPlus
      )

      tree.isValid = subtree.isValid && !hasUTurn
      tree.hasDivergence = tree.hasDivergence || subtree.hasDivergence
      tree.numLeapfrogs += subtree.numLeapfrogs

      // Dispose subtree
      disposeTreeResult(subtree)

      depth++
    }

    // 5. Build result
    const nextState = tree.candidate
    const nextLogProb = tree.candidateLogProb
    const nextGrads = tree.candidateGrads

    // Compute acceptance ratio approximation
    const proposedLogJoint = computeLogJoint(nextLogProb, momentum)
    const logAcceptRatio = tf.scalar(Math.min(0, proposedLogJoint - initLogJoint))

    // Dispose tree endpoints and momentum
    disposeState(tree.stateMinus)
    disposeState(tree.momentumMinus)
    disposeState(tree.gradsMinus)
    disposeState(tree.statePlus)
    disposeState(tree.momentumPlus)
    disposeState(tree.gradsPlus)
    disposeState(momentum)

    return {
      nextState,
      kernelResults: {
        targetLogProb: nextLogProb,
        gradsTargetLogProb: nextGrads,
        isAccepted: tf.scalar(1), // NUTS always "accepts" (built into tree)
        logAcceptRatio,
        stepSize: tf.scalar(this._stepSize),
        leapfrogsTaken: tree.numLeapfrogs,
        reachMaxDepth: depth >= this._maxTreeDepth,
        hasDivergence: tree.hasDivergence
      }
    }
  }
}
