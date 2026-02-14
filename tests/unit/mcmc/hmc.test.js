import * as tf from '@tensorflow/tfjs'
import { HamiltonianMonteCarlo } from '../../../src/mcmc/hmc'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('HamiltonianMonteCarlo', () => {
  describe('constructor', () => {
    test('stores parameters', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: (x) => tf.mul(-0.5, tf.square(x)),
        stepSize: 0.1,
        numLeapfrogSteps: 5
      })
      expect(kernel.stepSize).toBe(0.1)
      expect(kernel.numLeapfrogSteps).toBe(5)
      expect(kernel.isCalibrated).toBe(true)
    })
  })

  describe('bootstrapResults', () => {
    test('computes initial log prob and grads', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: (x) => tf.mul(-0.5, tf.square(x)),
        stepSize: 0.1,
        numLeapfrogSteps: 3
      })
      const state = tf.scalar(2.0)
      const kr = kernel.bootstrapResults(state)

      // logProb at x=2: -0.5*4 = -2
      expectClose(kr.targetLogProb.dataSync()[0], -2, { atol: 1e-5 })
      // grad at x=2: -2
      expectClose(kr.gradsTargetLogProb.dataSync()[0], -2, { atol: 1e-5 })
      expect(kr.isAccepted.dataSync()[0]).toBe(1)

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
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: (x) => tf.mul(-0.5, tf.square(x)),
        stepSize: 0.1,
        numLeapfrogSteps: 5
      })
      const state = tf.scalar(1.0)
      const kr = kernel.bootstrapResults(state)

      const { nextState, kernelResults } = kernel.oneStep(state, kr)

      expect(nextState instanceof tf.Tensor).toBe(true)
      expect(isFinite(nextState.dataSync()[0])).toBe(true)
      expect(isFinite(kernelResults.targetLogProb.dataSync()[0])).toBe(true)
      expect(kernelResults.isAccepted instanceof tf.Tensor).toBe(true)

      // Clean up
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
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: ({ a, b }) => tf.add(
          tf.mul(-0.5, tf.square(a)),
          tf.mul(-0.5, tf.square(b))
        ),
        stepSize: 0.1,
        numLeapfrogSteps: 5
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
    test('recovers N(0,1) mean and variance', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: (x) => tf.mul(-0.5, tf.square(x)),
        stepSize: 0.2,
        numLeapfrogSteps: 15
      })

      let state = tf.scalar(0)
      let kr = kernel.bootstrapResults(state)
      const samples = []

      // Burn-in + sampling
      const numBurnin = 300
      const numSamples = 1000

      for (let i = 0; i < numBurnin + numSamples; i++) {
        const { nextState, kernelResults } = kernel.oneStep(state, kr)

        // Dispose previous state and kernel results
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
      expectClose(stats.mean, 0, { atol: 0.3 })
      expectClose(stats.variance, 1, { atol: 0.7 })

      // Clean up final state
      state.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
    })

    test('recovers N(3, 2²) mean', () => {
      // N(3, 4): logπ(x) = -0.5 * ((x-3)/2)² - log(2)
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: (x) => tf.tidy(() =>
          tf.mul(-0.5, tf.square(tf.div(tf.sub(x, 3), 2)))
        ),
        stepSize: 0.5,
        numLeapfrogSteps: 10
      })

      let state = tf.scalar(0)
      let kr = kernel.bootstrapResults(state)
      const samples = []

      for (let i = 0; i < 200 + 500; i++) {
        const { nextState, kernelResults } = kernel.oneStep(state, kr)
        state.dispose()
        kr.targetLogProb.dispose()
        disposeState(kr.gradsTargetLogProb)
        kr.isAccepted.dispose()
        kr.logAcceptRatio.dispose()
        kr.stepSize.dispose()
        state = nextState
        kr = kernelResults
        if (i >= 200) samples.push(state.dataSync()[0])
      }

      const stats = sampleStats(samples)
      expectClose(stats.mean, 3, { atol: 0.5 })

      state.dispose()
      kr.targetLogProb.dispose()
      disposeState(kr.gradsTargetLogProb)
      kr.isAccepted.dispose()
      kr.logAcceptRatio.dispose()
      kr.stepSize.dispose()
    })
  })
})
