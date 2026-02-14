import * as tf from '@tensorflow/tfjs'
import { MultivariateNormalTriL } from '../../../src/distributions/mvn_tril'
import { expectClose } from '../../helpers/tolerance'

describe('MultivariateNormalTriL distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[1, 0], [0, 1]]
      })
      expect(d.eventShape).toEqual([2])
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('standard MVN at origin', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[1, 0], [0, 1]]
      })
      const lp = d.logProb([0, 0])
      // log p(0,0) = -0.5*2*log(2π) = -log(2π)
      expectClose(lp.dataSync()[0], -Math.log(2 * Math.PI), { atol: 1e-4 })
      lp.dispose()
      d.dispose()
    })

    test('non-identity scale', () => {
      // L = [[2, 0], [1, 1]], Σ = L*Lᵀ = [[4, 2], [2, 2]]
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[2, 0], [1, 1]]
      })
      const lp = d.logProb([0, 0])
      // logdet(Σ) = 2*log(2) + 2*log(1) = 2*log(2)
      // logp = -0.5*2*log(2π) - 0.5*2*log(2) - 0 = -log(2π) - log(2)
      expectClose(lp.dataSync()[0], -Math.log(2 * Math.PI) - Math.log(2), { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })

    test('logProb decreases away from mean', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[1, 0], [0, 1]]
      })
      const lpCenter = d.logProb([0, 0]).dataSync()[0]
      const lpAway = d.logProb([1, 1]).dataSync()[0]
      expect(lpCenter).toBeGreaterThan(lpAway)
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0, 0],
        scaleTril: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      })
      const s = d.sample([100])
      expect(s.shape).toEqual([100, 3])
      s.dispose()
      d.dispose()
    })

    test('sample mean near loc', () => {
      const d = new MultivariateNormalTriL({
        loc: [1, -1],
        scaleTril: [[1, 0], [0, 1]]
      })
      const s = d.sample([5000])
      const mean = tf.mean(s, 0)
      const data = mean.dataSync()
      expectClose(data[0], 1, { atol: 0.1 })
      expectClose(data[1], -1, { atol: 0.1 })
      s.dispose()
      mean.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/entropy', () => {
    test('mean = loc', () => {
      const d = new MultivariateNormalTriL({
        loc: [1, 2],
        scaleTril: [[1, 0], [0.5, 0.866]]
      })
      const m = d.mean()
      expect(m.dataSync()[0]).toBeCloseTo(1, 5)
      expect(m.dataSync()[1]).toBeCloseTo(2, 5)
      m.dispose()
      d.dispose()
    })

    test('variance of standard MVN = [1, 1]', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[1, 0], [0, 1]]
      })
      const v = d.variance()
      expect(v.dataSync()[0]).toBeCloseTo(1, 4)
      expect(v.dataSync()[1]).toBeCloseTo(1, 4)
      v.dispose()
      d.dispose()
    })

    test('entropy of standard 2D MVN', () => {
      const d = new MultivariateNormalTriL({
        loc: [0, 0],
        scaleTril: [[1, 0], [0, 1]]
      })
      // H = 0.5*d*(1 + log(2π)) + sum(log|diag(L)|)
      // = 0.5*2*(1 + log(2π)) + 0 = 1 + log(2π) ≈ 2.8379
      const h = d.entropy()
      expectClose(h.dataSync()[0], 1 + Math.log(2 * Math.PI), { atol: 1e-4 })
      h.dispose()
      d.dispose()
    })
  })
})
