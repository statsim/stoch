import * as tf from '@tensorflow/tfjs'
import { posteriorPredictive, priorPredictive } from '../../../src/mcmc/posterior_predictive'

describe('posteriorPredictive', () => {
  test('generates predictions from tensor samples', () => {
    // Simulate posterior samples of a slope parameter
    const samples = tf.tensor1d([1, 2, 3, 4, 5])
    const x = 2

    const preds = posteriorPredictive({
      samples,
      predictFn: (slope) => tf.mul(slope, x)
    })

    expect(preds.shape).toEqual([5])
    const data = preds.dataSync()
    expect(data[0]).toBeCloseTo(2)
    expect(data[1]).toBeCloseTo(4)
    expect(data[2]).toBeCloseTo(6)
    expect(data[3]).toBeCloseTo(8)
    expect(data[4]).toBeCloseTo(10)
    preds.dispose()
    samples.dispose()
  })

  test('generates predictions from object samples', () => {
    const samples = {
      slope: tf.tensor1d([1, 2, 3]),
      intercept: tf.tensor1d([0, 1, 2])
    }
    const x = 3

    const preds = posteriorPredictive({
      samples,
      predictFn: ({ slope, intercept }) => tf.add(tf.mul(slope, x), intercept)
    })

    expect(preds.shape).toEqual([3])
    const data = preds.dataSync()
    expect(data[0]).toBeCloseTo(3)   // 1*3 + 0
    expect(data[1]).toBeCloseTo(7)   // 2*3 + 1
    expect(data[2]).toBeCloseTo(11)  // 3*3 + 2
    preds.dispose()
    samples.slope.dispose()
    samples.intercept.dispose()
  })

  test('respects numSamples limit', () => {
    const samples = tf.tensor1d([1, 2, 3, 4, 5])
    const preds = posteriorPredictive({
      samples,
      predictFn: (x) => tf.mul(x, 2),
      numSamples: 3
    })
    expect(preds.shape).toEqual([3])
    preds.dispose()
    samples.dispose()
  })
})

describe('priorPredictive', () => {
  test('generates predictions from prior draws', () => {
    let callCount = 0
    const preds = priorPredictive({
      priorFn: () => {
        callCount++
        return tf.scalar(callCount)
      },
      predictFn: (sample) => tf.mul(sample, 2),
      numSamples: 5
    })

    expect(preds.shape).toEqual([5])
    expect(callCount).toBe(5)
    const data = preds.dataSync()
    expect(data[0]).toBeCloseTo(2)
    expect(data[4]).toBeCloseTo(10)
    preds.dispose()
  })

  test('defaults to 100 samples', () => {
    const preds = priorPredictive({
      priorFn: () => tf.scalar(1),
      predictFn: (x) => x
    })
    expect(preds.shape).toEqual([100])
    preds.dispose()
  })
})
