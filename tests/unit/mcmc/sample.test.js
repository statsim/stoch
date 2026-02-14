import * as tf from '@tensorflow/tfjs'
import { sample } from '../../../src/mcmc/sample'

describe('sample (high-level)', () => {
  test('samples from N(0,1) with NUTS', () => {
    const targetLogProb = (x) => tf.mul(-0.5, tf.square(x))

    const { samples, diagnostics } = sample({
      targetLogProbFn: targetLogProb,
      initialState: tf.scalar(0),
      numResults: 200,
      numBurninSteps: 100,
      stepSize: 0.5,
      kernel: 'nuts'
    })

    expect(samples.shape).toEqual([200])
    const data = samples.dataSync()
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i]
    const mean = sum / data.length
    // Mean should be roughly 0
    expect(Math.abs(mean)).toBeLessThan(1.0)
    expect(diagnostics.ess).toBeGreaterThan(10)
    samples.dispose()
  }, 30000)

  test('samples from N(0,1) with HMC', () => {
    const targetLogProb = (x) => tf.mul(-0.5, tf.square(x))

    const { samples, diagnostics } = sample({
      targetLogProbFn: targetLogProb,
      initialState: tf.scalar(0),
      numResults: 200,
      numBurninSteps: 100,
      stepSize: 0.1,
      kernel: 'hmc',
      numLeapfrogSteps: 10
    })

    expect(samples.shape).toEqual([200])
    const data = samples.dataSync()
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i]
    const mean = sum / data.length
    expect(Math.abs(mean)).toBeLessThan(1.0)
    samples.dispose()
  }, 30000)

  test('multi-parameter state', () => {
    const targetLogProb = ({ mu, sigma }) =>
      tf.add(tf.mul(-0.5, tf.square(mu)), tf.mul(-0.5, tf.square(sigma)))

    const { samples, diagnostics } = sample({
      targetLogProbFn: targetLogProb,
      initialState: { mu: tf.scalar(0), sigma: tf.scalar(0) },
      numResults: 100,
      numBurninSteps: 50,
      stepSize: 0.3,
      kernel: 'nuts'
    })

    expect(samples.mu.shape).toEqual([100])
    expect(samples.sigma.shape).toEqual([100])
    expect(diagnostics.mu.ess).toBeGreaterThan(5)
    expect(diagnostics.sigma.ess).toBeGreaterThan(5)
    samples.mu.dispose()
    samples.sigma.dispose()
  }, 30000)

  test('NUTS tracks divergences', () => {
    const targetLogProb = (x) => tf.mul(-0.5, tf.square(x))

    const { samples, diagnostics, trace } = sample({
      targetLogProbFn: targetLogProb,
      initialState: tf.scalar(0),
      numResults: 50,
      numBurninSteps: 20,
      stepSize: 0.5,
      kernel: 'nuts'
    })

    // Diagnostics should have divergence counts
    expect(typeof diagnostics.numDivergent).toBe('number')
    expect(typeof diagnostics.numMaxDepth).toBe('number')
    expect(diagnostics.meanLeapfrogs).toBeGreaterThan(0)

    // Trace should be populated
    expect(trace).toBeDefined()
    expect(trace[0]).toHaveLength(50)
    expect(trace[0][0]).toHaveProperty('hasDivergence')
    expect(trace[0][0]).toHaveProperty('leapfrogsTaken')
    samples.dispose()
  }, 30000)

  test('multi-chain produces R-hat', () => {
    const targetLogProb = (x) => tf.mul(-0.5, tf.square(x))

    const { samples, diagnostics } = sample({
      targetLogProbFn: targetLogProb,
      initialState: tf.scalar(0),
      numResults: 200,
      numBurninSteps: 100,
      numChains: 2,
      stepSize: 0.3,
      kernel: 'hmc',
      numLeapfrogSteps: 10
    })

    expect(samples).toHaveLength(2)
    expect(samples[0].shape).toEqual([200])
    expect(samples[1].shape).toEqual([200])
    // R-hat should be finite for multi-chain
    expect(isFinite(diagnostics.rhat)).toBe(true)
    expect(diagnostics.rhat).toBeGreaterThan(0)
    samples[0].dispose()
    samples[1].dispose()
  }, 60000)
})
