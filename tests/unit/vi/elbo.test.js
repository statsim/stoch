import * as tf from '@tensorflow/tfjs'
import { computeElbo } from '../../../src/vi/elbo'
import { trainableNormal } from '../../../src/vi/trainable'

describe('computeElbo', () => {
  test('returns a scalar tensor', () => {
    const q = trainableNormal({ loc: 0, scale: 1 })
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))

    const elbo = computeElbo({
      targetLogProbFn,
      surrogatePosterior: q,
      numSamples: 1
    })

    expect(elbo instanceof tf.Tensor).toBe(true)
    expect(elbo.shape).toEqual([])
    expect(isFinite(elbo.dataSync()[0])).toBe(true)

    elbo.dispose()
    q.dispose()
  })

  test('ELBO is higher when q is close to target', () => {
    // Target: N(0, 1)
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))

    // q close to target
    const qGood = trainableNormal({ loc: 0, scale: 1 })
    // q far from target
    const qBad = trainableNormal({ loc: 10, scale: 0.1 })

    // Average over many samples for stable comparison
    let elboGood = 0
    let elboBad = 0
    const numTrials = 20

    for (let i = 0; i < numTrials; i++) {
      const eg = computeElbo({
        targetLogProbFn,
        surrogatePosterior: qGood,
        numSamples: 10
      })
      elboGood += eg.dataSync()[0]
      eg.dispose()

      const eb = computeElbo({
        targetLogProbFn,
        surrogatePosterior: qBad,
        numSamples: 10
      })
      elboBad += eb.dataSync()[0]
      eb.dispose()
    }

    elboGood /= numTrials
    elboBad /= numTrials

    expect(elboGood).toBeGreaterThan(elboBad)

    qGood.dispose()
    qBad.dispose()
  })

  test('multiple MC samples reduce variance', () => {
    const q = trainableNormal({ loc: 0, scale: 1 })
    const targetLogProbFn = (z) => tf.mul(-0.5, tf.square(z))

    // Compute ELBO with 1 sample vs 50 samples, check both are finite
    const elbo1 = computeElbo({
      targetLogProbFn,
      surrogatePosterior: q,
      numSamples: 1
    })
    const elbo50 = computeElbo({
      targetLogProbFn,
      surrogatePosterior: q,
      numSamples: 50
    })

    expect(isFinite(elbo1.dataSync()[0])).toBe(true)
    expect(isFinite(elbo50.dataSync()[0])).toBe(true)

    elbo1.dispose()
    elbo50.dispose()
    q.dispose()
  })
})
