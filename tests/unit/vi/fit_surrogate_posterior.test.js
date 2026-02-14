import * as tf from '@tensorflow/tfjs'
import { fitSurrogatePosterior } from '../../../src/vi/fit_surrogate_posterior'
import { trainableNormal, buildMeanFieldPosterior } from '../../../src/vi/trainable'

describe('fitSurrogatePosterior', () => {
  test('loss decreases over optimization', () => {
    // Target: N(0, 1)
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))
    const q = trainableNormal({ loc: 5, scale: 3 }) // Start far from target

    const { losses } = fitSurrogatePosterior({
      targetLogProbFn,
      surrogatePosterior: q,
      optimizer: tf.train.adam(0.05),
      numSteps: 100,
      numElboSamples: 5
    })

    expect(losses.length).toBe(100)
    // Loss should generally decrease (compare first 10 avg vs last 10 avg)
    const firstAvg = losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10
    const lastAvg = losses.slice(-10).reduce((a, b) => a + b, 0) / 10
    expect(lastAvg).toBeLessThan(firstAvg)

    q.dispose()
  })

  test('recovers N(0,1) parameters', () => {
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))
    const q = trainableNormal({ loc: 3, scale: 0.5 })

    fitSurrogatePosterior({
      targetLogProbFn,
      surrogatePosterior: q,
      optimizer: tf.train.adam(0.05),
      numSteps: 300,
      numElboSamples: 10
    })

    const params = q.getParameters()
    // Should be close to N(0, 1)
    expect(Math.abs(params.loc)).toBeLessThan(1.0)
    expect(Math.abs(params.scale - 1)).toBeLessThan(1.0)

    q.dispose()
  })

  test('supports early stopping via convergenceFn', () => {
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))
    const q = trainableNormal({ loc: 0, scale: 1 })

    const { losses } = fitSurrogatePosterior({
      targetLogProbFn,
      surrogatePosterior: q,
      optimizer: tf.train.adam(0.01),
      numSteps: 1000,
      convergenceFn: (step) => step >= 10 // Stop after 10 steps
    })

    expect(losses.length).toBe(11) // 0..10 inclusive

    q.dispose()
  })

  test('supports traceLogProbFn callback', () => {
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))
    const q = trainableNormal({ loc: 0, scale: 1 })
    const traced = []

    fitSurrogatePosterior({
      targetLogProbFn,
      surrogatePosterior: q,
      optimizer: tf.train.adam(0.01),
      numSteps: 5,
      traceLogProbFn: (step, loss) => traced.push({ step, loss })
    })

    expect(traced.length).toBe(5)
    expect(traced[0].step).toBe(0)
    expect(typeof traced[0].loss).toBe('number')

    q.dispose()
  })

  test('multi-parameter VI with buildMeanFieldPosterior', () => {
    // Target: independent N(3, 1) and N(-2, 1)
    const targetLogProbFn = ({ a, b }) => {
      const lpA = tf.mul(-0.5, tf.square(tf.sub(a, 3)))
      const lpB = tf.mul(-0.5, tf.square(tf.add(b, 2)))
      return tf.add(lpA, lpB)
    }

    const q = buildMeanFieldPosterior({ a: 0, b: 0 })

    fitSurrogatePosterior({
      targetLogProbFn,
      surrogatePosterior: q,
      optimizer: tf.train.adam(0.05),
      numSteps: 300,
      numElboSamples: 5
    })

    const params = q.getParameters()
    // Should recover approximate means
    expect(Math.abs(params.a.loc - 3)).toBeLessThan(1.5)
    expect(Math.abs(params.b.loc - (-2))).toBeLessThan(1.5)

    q.dispose()
  })
})
