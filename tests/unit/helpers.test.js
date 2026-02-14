import { expectClose, expectTensorClose, autoTolerance, sampleStats } from '../helpers/tolerance'
import * as tf from '@tensorflow/tfjs'

describe('expectClose', () => {
  test('passes for exact match', () => {
    expectClose(1.0, 1.0)
  })

  test('passes within default tolerance', () => {
    expectClose(1.0, 1.0 + 1e-7)
  })

  test('fails for values outside tolerance', () => {
    expect(() => expectClose(1.0, 2.0)).toThrow()
  })

  test('respects custom rtol', () => {
    expectClose(1.1, 1.0, { rtol: 0.2 })
  })

  test('respects custom atol', () => {
    expectClose(1.0, 1.05, { atol: 0.1 })
  })
})

describe('expectTensorClose', () => {
  test('passes for close tensors', () => {
    const a = tf.tensor([1.0, 2.0, 3.0])
    const b = tf.tensor([1.0, 2.0, 3.0])
    expectTensorClose(a, b)
    a.dispose()
    b.dispose()
  })

  test('fails for different tensors', () => {
    const a = tf.tensor([1.0, 2.0])
    const b = tf.tensor([1.0, 5.0])
    expect(() => expectTensorClose(a, b)).toThrow()
    a.dispose()
    b.dispose()
  })
})

describe('autoTolerance', () => {
  test('mean tolerance decreases with n', () => {
    const tol1 = autoTolerance('mean', 1000, 1.0)
    const tol2 = autoTolerance('mean', 100000, 1.0)
    expect(tol2).toBeLessThan(tol1)
  })

  test('variance tolerance is positive', () => {
    const tol = autoTolerance('variance', 10000, 1.0, 3.0)
    expect(tol).toBeGreaterThan(0)
  })

  test('skew tolerance is computed', () => {
    const tol = autoTolerance('skew', 10000)
    expect(tol).toBeGreaterThan(0)
  })

  test('kurtosis tolerance is computed', () => {
    const tol = autoTolerance('kurtosis', 10000)
    expect(tol).toBeGreaterThan(0)
  })

  test('throws for unknown stat', () => {
    expect(() => autoTolerance('median', 1000, 1.0)).toThrow()
  })
})

describe('sampleStats', () => {
  test('computes correct stats for known data', () => {
    // 1, 2, 3, 4, 5 → mean=3, variance=2.5
    const data = [1, 2, 3, 4, 5]
    const stats = sampleStats(data)
    expect(stats.mean).toBeCloseTo(3.0, 10)
    expect(stats.variance).toBeCloseTo(2.5, 10)
  })

  test('handles large float32 arrays', () => {
    const data = new Float32Array(10000)
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random()
    }
    const stats = sampleStats(data)
    expect(stats.mean).toBeGreaterThan(0.4)
    expect(stats.mean).toBeLessThan(0.6)
    expect(stats.variance).toBeGreaterThan(0.05)
    expect(stats.variance).toBeLessThan(0.15)
  })
})
