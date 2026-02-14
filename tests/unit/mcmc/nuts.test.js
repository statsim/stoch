import * as tf from '@tensorflow/tfjs'
import { NoUTurnSampler } from '../../../src/mcmc/nuts'
import { sampleChain } from '../../../src/mcmc/sample_chain'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('NoUTurnSampler', () => {
  const normalLogProb = (x) => tf.mul(-0.5, tf.square(x))

  describe('constructor', () => {
    test('stores parameters', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.1,
        maxTreeDepth: 8
      })
      expect(kernel.stepSize).toBe(0.1)
      expect(kernel.maxTreeDepth).toBe(8)
      expect(kernel.isCalibrated).toBe(true)
    })
  })

  describe('bootstrapResults', () => {
    test('computes initial log prob and grads', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.1
      })
      const state = tf.scalar(2.0)
      const kr = kernel.bootstrapResults(state)

      // logProb at x=2: -0.5*4 = -2
      expectClose(kr.targetLogProb.dataSync()[0], -2, { atol: 1e-5 })
      expect(kr.isAccepted.dataSync()[0]).toBe(1)
      expect(kr.leapfrogsTaken).toBe(0)

      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
      state.dispose()
    })
  })

  describe('oneStep', () => {
    test('produces a valid new state', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.1,
        maxTreeDepth: 3
      })
      const state = tf.scalar(1.0)
      const kr = kernel.bootstrapResults(state)

      const { nextState, kernelResults } = kernel.oneStep(state, kr)

      expect(nextState instanceof tf.Tensor).toBe(true)
      expect(isFinite(nextState.dataSync()[0])).toBe(true)
      expect(isFinite(kernelResults.targetLogProb.dataSync()[0])).toBe(true)
      expect(kernelResults.leapfrogsTaken).toBeGreaterThan(0)
      expect(typeof kernelResults.hasDivergence).toBe('boolean')
      expect(typeof kernelResults.reachMaxDepth).toBe('boolean')

      nextState.dispose()
      kernelResults.targetLogProb.dispose()
      disposeState(kernelResults.gradsTargetLogProb)
      kernelResults.isAccepted.dispose()
      kernelResults.logAcceptRatio.dispose()
      kernelResults.stepSize.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
      state.dispose()
    })

    test('multi-parameter state', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: ({ a, b }) => tf.add(
          tf.mul(-0.5, tf.square(a)),
          tf.mul(-0.5, tf.square(b))
        ),
        stepSize: 0.1,
        maxTreeDepth: 3
      })
      const state = { a: tf.scalar(1.0), b: tf.scalar(-1.0) }
      const kr = kernel.bootstrapResults(state)

      const { nextState, kernelResults } = kernel.oneStep(state, kr)

      expect(nextState).toHaveProperty('a')
      expect(nextState).toHaveProperty('b')
      expect(isFinite(nextState.a.dataSync()[0])).toBe(true)
      expect(isFinite(nextState.b.dataSync()[0])).toBe(true)

      disposeState(nextState)
      kernelResults.targetLogProb.dispose()
      disposeState(kernelResults.gradsTargetLogProb)
      kernelResults.isAccepted.dispose()
      kernelResults.logAcceptRatio.dispose()
      kernelResults.stepSize.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
      disposeState(state)
    })
  })

  describe('sampling from known distributions', () => {
    test('recovers N(0,1) mean', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.2,
        maxTreeDepth: 5
      })

      let state = tf.scalar(0)
      let kr = kernel.bootstrapResults(state)
      const samples = []

      const numBurnin = 100
      const numSamples = 300

      for (let i = 0; i < numBurnin + numSamples; i++) {
        const { nextState, kernelResults } = kernel.oneStep(state, kr)

        state.dispose()
        kr.targetLogProb.dispose()
        disposeState(kr.gradsTargetLogProb)
        kr.isAccepted.dispose()
        kr.logAcceptRatio.dispose()
        kr.stepSize.dispose()

        state = nextState
        kr = kernelResults

        if (i >= numBurnin) {
          samples.push(state.dataSync()[0])
        }
      }

      const stats = sampleStats(samples)
      expectClose(stats.mean, 0, { atol: 0.4 })

      state.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
    })

    test('works with sampleChain', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.2,
        maxTreeDepth: 5
      })

      const { samples } = sampleChain({
        numResults: 300,
        numBurninSteps: 100,
        currentState: tf.scalar(0),
        kernel
      })

      const stats = sampleStats(samples.dataSync())
      expectClose(stats.mean, 0, { atol: 0.4 })

      samples.dispose()
    })
  })

  describe('diagnostics', () => {
    test('reports tree depth info', () => {
      const kernel = new NoUTurnSampler({
        targetLogProbFn: normalLogProb,
        stepSize: 0.2,
        maxTreeDepth: 5
      })

      let state = tf.scalar(0)
      let kr = kernel.bootstrapResults(state)

      const { nextState, kernelResults } = kernel.oneStep(state, kr)

      // Should have taken some leapfrog steps
      expect(kernelResults.leapfrogsTaken).toBeGreaterThanOrEqual(1)
      expect(kernelResults.leapfrogsTaken).toBeLessThanOrEqual(32) // 2^5

      nextState.dispose()
      kernelResults.targetLogProb.dispose()
      disposeState(kernelResults.gradsTargetLogProb)
      kernelResults.isAccepted.dispose()
      kernelResults.logAcceptRatio.dispose()
      kernelResults.stepSize.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
      state.dispose()
    })
  })
})
