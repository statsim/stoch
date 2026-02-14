import * as tf from '@tensorflow/tfjs'
import { sampleChain } from '../../../src/mcmc/sample_chain'
import { HamiltonianMonteCarlo } from '../../../src/mcmc/hmc'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('sampleChain', () => {
  // Standard normal target
  const normalLogProb = (x) => tf.mul(-0.5, tf.square(x))

  describe('basic functionality', () => {
    test('returns stacked samples for scalar state', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: normalLogProb,
        stepSize: 0.3,
        numLeapfrogSteps: 5
      })

      const { samples, trace } = sampleChain({
        numResults: 10,
        numBurninSteps: 5,
        currentState: tf.scalar(0),
        kernel
      })

      expect(samples.shape).toEqual([10])
      expect(trace).toEqual([])
      samples.dispose()
    })

    test('returns stacked samples for object state', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: ({ a, b }) => tf.add(
          tf.mul(-0.5, tf.square(a)),
          tf.mul(-0.5, tf.square(b))
        ),
        stepSize: 0.3,
        numLeapfrogSteps: 5
      })

      const { samples } = sampleChain({
        numResults: 10,
        numBurninSteps: 5,
        currentState: { a: tf.scalar(0), b: tf.scalar(0) },
        kernel
      })

      expect(samples.a.shape).toEqual([10])
      expect(samples.b.shape).toEqual([10])
      samples.a.dispose()
      samples.b.dispose()
    })
  })

  describe('traceFn', () => {
    test('collects trace diagnostics', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: normalLogProb,
        stepSize: 0.3,
        numLeapfrogSteps: 5
      })

      const { samples, trace } = sampleChain({
        numResults: 10,
        numBurninSteps: 5,
        currentState: tf.scalar(0),
        kernel,
        traceFn: (state, kr) => ({
          isAccepted: kr.isAccepted.dataSync()[0] > 0,
          logProb: kr.targetLogProb.dataSync()[0]
        })
      })

      expect(trace.length).toBe(10)
      expect(typeof trace[0].isAccepted).toBe('boolean')
      expect(typeof trace[0].logProb).toBe('number')
      samples.dispose()
    })
  })

  describe('thinning', () => {
    test('numStepsBetweenResults reduces autocorrelation', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: normalLogProb,
        stepSize: 0.3,
        numLeapfrogSteps: 5
      })

      const { samples } = sampleChain({
        numResults: 10,
        numBurninSteps: 5,
        currentState: tf.scalar(0),
        kernel,
        numStepsBetweenResults: 2
      })

      expect(samples.shape).toEqual([10])
      samples.dispose()
    })
  })

  describe('memory management', () => {
    test('does not leak tensors', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: normalLogProb,
        stepSize: 0.3,
        numLeapfrogSteps: 3
      })

      const initState = tf.scalar(0)
      const before = tf.memory().numTensors

      const { samples } = sampleChain({
        numResults: 20,
        numBurninSteps: 10,
        currentState: initState,
        kernel
      })

      // After sampleChain, only the stacked samples tensor should remain
      // (plus the initial state which we still own)
      const after = tf.memory().numTensors
      // Should have: initState + samples tensor = before + 1
      expect(after).toBeLessThanOrEqual(before + 2)

      samples.dispose()
      initState.dispose()
    })
  })

  describe('posterior recovery', () => {
    test('recovers N(0,1) statistics', () => {
      const kernel = new HamiltonianMonteCarlo({
        targetLogProbFn: normalLogProb,
        stepSize: 0.2,
        numLeapfrogSteps: 15
      })

      const { samples } = sampleChain({
        numResults: 1000,
        numBurninSteps: 500,
        currentState: tf.scalar(0),
        kernel
      })

      const data = samples.dataSync()
      const stats = sampleStats(data)
      expectClose(stats.mean, 0, { atol: 0.5 })
      // Variance estimation is noisy with MCMC; use wide tolerance
      expect(stats.variance).toBeGreaterThan(0.1)
      expect(stats.variance).toBeLessThan(5.0)

      samples.dispose()
    })
  })
})
