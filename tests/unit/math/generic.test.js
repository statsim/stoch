import * as tf from '@tensorflow/tfjs'
import { log1mexp, logAddExp, softplusInverse } from '../../../src/math/generic'

describe('math/generic', () => {
  describe('log1mexp', () => {
    test('log(1 - exp(-1)) ≈ -0.4587', () => {
      const result = log1mexp(-1)
      expect(result.dataSync()[0]).toBeCloseTo(Math.log(1 - Math.exp(-1)), 4)
      result.dispose()
    })

    test('log(1 - exp(-5)) ≈ -0.00674', () => {
      const result = log1mexp(-5)
      expect(result.dataSync()[0]).toBeCloseTo(Math.log(1 - Math.exp(-5)), 4)
      result.dispose()
    })

    test('works with tensor input', () => {
      const x = tf.tensor([-0.1, -1.0, -5.0])
      const result = log1mexp(x)
      const data = result.dataSync()
      expect(data[0]).toBeCloseTo(Math.log(1 - Math.exp(-0.1)), 3)
      expect(data[1]).toBeCloseTo(Math.log(1 - Math.exp(-1.0)), 3)
      expect(data[2]).toBeCloseTo(Math.log(1 - Math.exp(-5.0)), 3)
      result.dispose()
      x.dispose()
    })
  })

  describe('logAddExp', () => {
    test('log(exp(0) + exp(0)) = log(2)', () => {
      const result = logAddExp(0, 0)
      expect(result.dataSync()[0]).toBeCloseTo(Math.LN2, 5)
      result.dispose()
    })

    test('log(exp(10) + exp(10)) = 10 + log(2)', () => {
      const result = logAddExp(10, 10)
      expect(result.dataSync()[0]).toBeCloseTo(10 + Math.LN2, 4)
      result.dispose()
    })

    test('numerically stable for large values', () => {
      const result = logAddExp(100, 100)
      expect(result.dataSync()[0]).toBeCloseTo(100 + Math.LN2, 3)
      result.dispose()
    })

    test('works with tensor inputs', () => {
      const a = tf.tensor([0, 10])
      const b = tf.tensor([0, 10])
      const result = logAddExp(a, b)
      const data = result.dataSync()
      expect(data[0]).toBeCloseTo(Math.LN2, 5)
      expect(data[1]).toBeCloseTo(10 + Math.LN2, 4)
      result.dispose()
      a.dispose()
      b.dispose()
    })
  })

  describe('softplusInverse', () => {
    test('inverse of softplus for moderate values', () => {
      // softplus(2) = log(1 + exp(2)) ≈ 2.1269
      const sp = Math.log(1 + Math.exp(2))
      const result = softplusInverse(sp)
      expect(result.dataSync()[0]).toBeCloseTo(2.0, 3)
      result.dispose()
    })

    test('for large x, softplusInverse(x) ≈ x', () => {
      const result = softplusInverse(50)
      expect(result.dataSync()[0]).toBeCloseTo(50, 3)
      result.dispose()
    })

    test('works with tensor input', () => {
      const x = tf.tensor([0.693, 2.127, 50])
      const result = softplusInverse(x)
      const data = result.dataSync()
      // softplus(0) ≈ 0.693, so softplusInverse(0.693) ≈ 0
      expect(data[0]).toBeCloseTo(0, 0)
      result.dispose()
      x.dispose()
    })
  })
})
