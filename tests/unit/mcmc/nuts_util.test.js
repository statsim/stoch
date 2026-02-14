import * as tf from '@tensorflow/tfjs'
import {
  checkUTurn,
  computeLogJoint,
  singleLeapfrogStep,
  buildTree,
  disposeTreeResult
} from '../../../src/mcmc/nuts_util'
import { computeGrads, disposeState } from '../../../src/mcmc/state_util'

describe('NUTS utilities', () => {
  const normalLogProb = (x) => tf.mul(-0.5, tf.square(x))
  const normalGradFn = (state) => computeGrads(normalLogProb, state)

  describe('checkUTurn', () => {
    test('no U-turn when moving apart', () => {
      // Positions moving apart, momenta in same direction as difference
      const qMinus = tf.scalar(-1)
      const qPlus = tf.scalar(1)
      const pMinus = tf.scalar(1)   // dot(p-, q+ - q-) = 1*2 > 0
      const pPlus = tf.scalar(1)    // dot(p+, q+ - q-) = 1*2 > 0

      const result = checkUTurn(qMinus, qPlus, pMinus, pPlus)
      expect(result).toBe(false)

      qMinus.dispose(); qPlus.dispose()
      pMinus.dispose(); pPlus.dispose()
    })

    test('U-turn when converging', () => {
      // Momenta pointing toward each other
      const qMinus = tf.scalar(-1)
      const qPlus = tf.scalar(1)
      const pMinus = tf.scalar(-1)  // dot(p-, q+ - q-) = -1*2 < 0 → U-turn!
      const pPlus = tf.scalar(1)

      const result = checkUTurn(qMinus, qPlus, pMinus, pPlus)
      expect(result).toBe(true)

      qMinus.dispose(); qPlus.dispose()
      pMinus.dispose(); pPlus.dispose()
    })

    test('works with multi-parameter state', () => {
      const qMinus = { a: tf.scalar(-1), b: tf.scalar(-1) }
      const qPlus = { a: tf.scalar(1), b: tf.scalar(1) }
      const pMinus = { a: tf.scalar(1), b: tf.scalar(1) }
      const pPlus = { a: tf.scalar(1), b: tf.scalar(1) }

      const result = checkUTurn(qMinus, qPlus, pMinus, pPlus)
      expect(result).toBe(false)

      disposeState(qMinus); disposeState(qPlus)
      disposeState(pMinus); disposeState(pPlus)
    })
  })

  describe('computeLogJoint', () => {
    test('computes logProb - 0.5 * ||p||^2', () => {
      const logProb = tf.scalar(-2)
      const momentum = tf.scalar(1)

      const result = computeLogJoint(logProb, momentum)
      // -2 - 0.5 * 1^2 = -2.5
      expect(result).toBeCloseTo(-2.5, 5)

      logProb.dispose(); momentum.dispose()
    })

    test('handles multi-parameter momentum', () => {
      const logProb = tf.scalar(0)
      const momentum = { a: tf.scalar(2), b: tf.scalar(3) }

      const result = computeLogJoint(logProb, momentum)
      // 0 - 0.5 * (4 + 9) = -6.5
      expect(result).toBeCloseTo(-6.5, 5)

      logProb.dispose(); disposeState(momentum)
    })
  })

  describe('singleLeapfrogStep', () => {
    test('takes one forward step', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(1)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)

      const result = singleLeapfrogStep({
        state, momentum, grads,
        stepSize: 0.1,
        targetLogProbGradFn: normalGradFn
      })

      expect(result.state instanceof tf.Tensor).toBe(true)
      expect(result.momentum instanceof tf.Tensor).toBe(true)
      expect(result.targetLogProb instanceof tf.Tensor).toBe(true)
      expect(result.grads instanceof tf.Tensor).toBe(true)

      // State should have moved forward
      expect(result.state.dataSync()[0]).not.toBe(0)
      expect(isFinite(result.state.dataSync()[0])).toBe(true)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      disposeState(result.state); disposeState(result.momentum)
      result.targetLogProb.dispose(); disposeState(result.grads)
    })

    test('takes one backward step', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(1)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)

      const forward = singleLeapfrogStep({
        state, momentum, grads,
        stepSize: 0.1,
        targetLogProbGradFn: normalGradFn
      })

      const { value: logProb2, grads: grads2 } = computeGrads(normalLogProb, state)
      const backward = singleLeapfrogStep({
        state, momentum, grads: grads2,
        stepSize: -0.1,
        targetLogProbGradFn: normalGradFn
      })

      // Forward and backward should move in opposite directions
      const fwd = forward.state.dataSync()[0]
      const bwd = backward.state.dataSync()[0]
      expect(fwd > 0).toBe(true)
      expect(bwd < 0).toBe(true)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      logProb2.dispose(); disposeState(grads2)
      disposeState(forward.state); disposeState(forward.momentum)
      forward.targetLogProb.dispose(); disposeState(forward.grads)
      disposeState(backward.state); disposeState(backward.momentum)
      backward.targetLogProb.dispose(); disposeState(backward.grads)
    })

    test('does not dispose inputs', () => {
      const state = tf.scalar(1.0)
      const momentum = tf.scalar(0.5)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)

      singleLeapfrogStep({
        state, momentum, grads,
        stepSize: 0.1,
        targetLogProbGradFn: normalGradFn
      })

      // Inputs should still be valid
      expect(state.isDisposed).toBe(false)
      expect(momentum.isDisposed).toBe(false)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
    })
  })

  describe('buildTree', () => {
    test('depth 0 takes one leapfrog step', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(1)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)
      const logJoint = computeLogJoint(logProb, momentum)

      const tree = buildTree({
        state, momentum, grads,
        direction: 1, depth: 0, stepSize: 0.1,
        targetLogProbGradFn: normalGradFn,
        logSliceThreshold: logJoint - 1000
      })

      expect(tree.numLeapfrogs).toBe(1)
      expect(tree.isValid).toBe(true)
      expect(tree.candidate instanceof tf.Tensor).toBe(true)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      disposeTreeResult(tree)
    })

    test('depth 1 takes two leapfrog steps', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(1)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)
      const logJoint = computeLogJoint(logProb, momentum)

      const tree = buildTree({
        state, momentum, grads,
        direction: 1, depth: 1, stepSize: 0.1,
        targetLogProbGradFn: normalGradFn,
        logSliceThreshold: logJoint - 1000
      })

      expect(tree.numLeapfrogs).toBe(2)
      expect(tree.isValid).toBe(true)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      disposeTreeResult(tree)
    })

    test('depth 3 takes eight leapfrog steps', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(1)
      const { value: logProb, grads } = computeGrads(normalLogProb, state)
      const logJoint = computeLogJoint(logProb, momentum)

      const tree = buildTree({
        state, momentum, grads,
        direction: 1, depth: 3, stepSize: 0.1,
        targetLogProbGradFn: normalGradFn,
        logSliceThreshold: logJoint - 1000
      })

      expect(tree.numLeapfrogs).toBe(8)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      disposeTreeResult(tree)
    })

    test('detects divergence with tiny threshold', () => {
      const state = tf.scalar(0)
      const momentum = tf.scalar(10) // very high momentum
      const { value: logProb, grads } = computeGrads(normalLogProb, state)

      const tree = buildTree({
        state, momentum, grads,
        direction: 1, depth: 0, stepSize: 5.0, // huge step → energy explosion
        targetLogProbGradFn: normalGradFn,
        logSliceThreshold: 0,
        maxEnergyDiff: 0.001 // very strict
      })

      // Should detect divergence
      expect(tree.hasDivergence).toBe(true)

      state.dispose(); momentum.dispose()
      logProb.dispose(); disposeState(grads)
      disposeTreeResult(tree)
    })
  })
})
