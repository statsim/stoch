import * as tf from '@tensorflow/tfjs'
import { logGamma, digamma, logBeta, ndtr, logNdtr } from '../../../src/math/special'
import { expectClose } from '../../helpers/tolerance'

describe('math/special', () => {
  describe('logGamma', () => {
    // Reference values from scipy.special.gammaln
    const cases = [
      [1, 0],
      [2, 0],
      [3, Math.log(2)],
      [4, Math.log(6)],
      [5, Math.log(24)],
      [0.5, Math.log(Math.PI) / 2], // Γ(1/2) = √π
      [10, 12.80182748],
    ]

    cases.forEach(([x, expected]) => {
      test(`logGamma(${x}) ≈ ${expected}`, () => {
        const result = logGamma(x)
        expectClose(result.dataSync()[0], expected, { rtol: 1e-4, atol: 1e-5 })
        result.dispose()
      })
    })

    test('works element-wise on tensors', () => {
      const x = tf.tensor([1, 2, 3, 4, 5])
      const result = logGamma(x)
      const data = result.dataSync()
      expectClose(data[0], 0, { atol: 1e-5 })
      expectClose(data[1], 0, { atol: 1e-5 })
      expectClose(data[2], Math.log(2), { rtol: 1e-4 })
      expectClose(data[3], Math.log(6), { rtol: 1e-4 })
      expectClose(data[4], Math.log(24), { rtol: 1e-4 })
      result.dispose()
      x.dispose()
    })
  })

  describe('digamma', () => {
    // Reference values from scipy.special.digamma
    const cases = [
      [1, -0.5772156649],
      [2, 0.4227843351],
      [5, 1.5061176685],
      [10, 2.2517525890],
      [0.5, -1.9635100260],
    ]

    cases.forEach(([x, expected]) => {
      test(`digamma(${x}) ≈ ${expected}`, () => {
        const result = digamma(x)
        expectClose(result.dataSync()[0], expected, { rtol: 1e-3, atol: 1e-3 })
        result.dispose()
      })
    })

    test('works element-wise on tensors', () => {
      const x = tf.tensor([1, 2, 5, 10])
      const result = digamma(x)
      const data = result.dataSync()
      expectClose(data[0], -0.5772156649, { rtol: 1e-3, atol: 1e-3 })
      expectClose(data[1], 0.4227843351, { rtol: 1e-3, atol: 1e-3 })
      result.dispose()
      x.dispose()
    })
  })

  describe('logBeta', () => {
    // logBeta(a, b) = logGamma(a) + logGamma(b) - logGamma(a+b)
    // Reference: scipy.special.betaln
    const cases = [
      [1, 1, 0],           // Beta(1,1) = 1, log(1) = 0
      [2, 2, -1.791759469], // scipy.special.betaln(2, 2)
      [0.5, 0.5, Math.log(Math.PI)], // Beta(0.5, 0.5) = π
      [5, 1, -Math.log(5)],  // Beta(5,1) = 1/5
    ]

    cases.forEach(([a, b, expected]) => {
      test(`logBeta(${a}, ${b}) ≈ ${expected}`, () => {
        const result = logBeta(a, b)
        expectClose(result.dataSync()[0], expected, { rtol: 1e-3, atol: 1e-4 })
        result.dispose()
      })
    })
  })

  describe('ndtr', () => {
    // Reference: scipy.stats.norm.cdf
    const cases = [
      [0, 0.5],
      [1, 0.8413447],
      [-1, 0.1586553],
      [2, 0.9772499],
      [-2, 0.0227501],
      [3, 0.9986501],
      [-3, 0.0013499],
    ]

    cases.forEach(([x, expected]) => {
      test(`ndtr(${x}) ≈ ${expected}`, () => {
        const result = ndtr(x)
        expectClose(result.dataSync()[0], expected, { rtol: 1e-4, atol: 1e-5 })
        result.dispose()
      })
    })

    test('works element-wise on tensors', () => {
      const x = tf.tensor([-2, -1, 0, 1, 2])
      const result = ndtr(x)
      const data = result.dataSync()
      expectClose(data[2], 0.5, { atol: 1e-5 })
      // Symmetry: Φ(-x) = 1 - Φ(x)
      expectClose(data[0] + data[4], 1.0, { atol: 1e-4 })
      expectClose(data[1] + data[3], 1.0, { atol: 1e-4 })
      result.dispose()
      x.dispose()
    })
  })

  describe('logNdtr', () => {
    test('logNdtr(0) = log(0.5)', () => {
      const result = logNdtr(0)
      expectClose(result.dataSync()[0], Math.log(0.5), { rtol: 1e-4 })
      result.dispose()
    })

    test('logNdtr(3) ≈ log(0.9987)', () => {
      const result = logNdtr(3)
      expectClose(result.dataSync()[0], Math.log(0.9986501), { rtol: 1e-3 })
      result.dispose()
    })
  })
})
