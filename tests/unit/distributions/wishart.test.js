import * as tf from '@tensorflow/tfjs'
import { Wishart } from '../../../src/distributions/wishart'
import { expectClose } from '../../helpers/tolerance'

describe('Wishart distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Wishart({ df: 5, scaleTril: [[1, 0], [0, 1]] })
      expect(d.df.dataSync()[0]).toBe(5)
      expect(d.eventShape).toEqual([2, 2])
      d.dispose()
    })

    test('throws for df <= d-1', () => {
      expect(() => new Wishart({ df: 1, scaleTril: [[1, 0], [0, 1]] })).toThrow()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Wishart({ df: 5, scaleTril: [[1, 0], [0, 1]] })
      const s = d.sample([10])
      expect(s.shape).toEqual([10, 2, 2])
      s.dispose()
      d.dispose()
    })

    test('samples are symmetric', () => {
      const d = new Wishart({ df: 5, scaleTril: [[1, 0], [0, 1]] })
      const s = d.sample([5])
      const data = s.dataSync()
      for (let i = 0; i < 5; i++) {
        const off = i * 4
        // [0,1] should equal [1,0]
        expectClose(data[off + 1], data[off + 2], { atol: 1e-4 })
      }
      s.dispose()
      d.dispose()
    })

    test('samples are positive definite (positive diagonal)', () => {
      const d = new Wishart({ df: 5, scaleTril: [[1, 0], [0, 1]] })
      const s = d.sample([10])
      const data = s.dataSync()
      for (let i = 0; i < 10; i++) {
        const off = i * 4
        expect(data[off]).toBeGreaterThan(0)     // [0,0]
        expect(data[off + 3]).toBeGreaterThan(0)  // [1,1]
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean = df * V', () => {
      const d = new Wishart({ df: 5, scaleTril: [[2, 0], [0, 3]] })
      const m = d.mean()
      // V = L*Lᵀ = [[4,0],[0,9]], E[X] = 5*V = [[20,0],[0,45]]
      const data = m.dataSync()
      expectClose(data[0], 20, { atol: 1e-3 })
      expectClose(data[1], 0, { atol: 1e-3 })
      expectClose(data[2], 0, { atol: 1e-3 })
      expectClose(data[3], 45, { atol: 1e-3 })
      m.dispose()
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb is finite for valid input', () => {
      const d = new Wishart({ df: 5, scaleTril: [[1, 0], [0, 1]] })
      // A valid 2x2 positive definite matrix
      const X = tf.tensor([[2, 0.5], [0.5, 1]])
      const lp = d.logProb(X)
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose()
      X.dispose()
      d.dispose()
    })

    test('logProb is higher near mean', () => {
      const d = new Wishart({ df: 10, scaleTril: [[1, 0], [0, 1]] })
      // Mean = 10*I = [[10,0],[0,10]]
      const near = tf.tensor([[10, 0], [0, 10]])
      const far = tf.tensor([[1, 0], [0, 1]])
      const lpNear = d.logProb(near).dataSync()[0]
      const lpFar = d.logProb(far).dataSync()[0]
      expect(lpNear).toBeGreaterThan(lpFar)
      near.dispose(); far.dispose()
      d.dispose()
    })
  })
})
