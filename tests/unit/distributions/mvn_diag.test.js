import * as tf from '@tensorflow/tfjs'
import { MultivariateNormalDiag } from '../../../src/distributions/mvn_diag'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('MultivariateNormalDiag', () => {
  describe('constructor and properties', () => {
    test('basic construction', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [1, 1] })
      expect(d.name).toBe('MultivariateNormalDiag')
      expect(d.batchShape).toEqual([])
      expect(d.eventShape).toEqual([2])
      d.dispose()
    })

    test('loc and scaleDiag accessors', () => {
      const d = new MultivariateNormalDiag({ loc: [1, 2, 3], scaleDiag: [0.5, 1, 2] })
      expect(Array.from(d.loc.dataSync())).toEqual([1, 2, 3])
      expect(Array.from(d.scaleDiag.dataSync())).toEqual([0.5, 1, 2])
      d.dispose()
    })

    test('3-dimensional MVN', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0, 0], scaleDiag: [1, 1, 1] })
      expect(d.eventShape).toEqual([3])
      expect(d.batchShape).toEqual([])
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb at mean is scalar', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [1, 1] })
      const lp = d.logProb(tf.tensor([0, 0]))
      expect(lp.shape).toEqual([])

      // Sum of two standard Normal logProbs at 0
      const logProbAtMean = -0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], 2 * logProbAtMean, { atol: 1e-4 })

      lp.dispose()
      d.dispose()
    })

    test('logProb matches independent Normal sum', () => {
      const d = new MultivariateNormalDiag({ loc: [1, 2], scaleDiag: [0.5, 2] })
      const x = tf.tensor([1.5, 3])
      const lp = d.logProb(x)

      // Manual calculation
      const lp1 = -0.5 * Math.log(2 * Math.PI) - Math.log(0.5) - 0.5 * ((1.5 - 1) / 0.5) ** 2
      const lp2 = -0.5 * Math.log(2 * Math.PI) - Math.log(2) - 0.5 * ((3 - 2) / 2) ** 2
      expectClose(lp.dataSync()[0], lp1 + lp2, { atol: 1e-4 })

      lp.dispose()
      x.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('sample shape', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0, 0], scaleDiag: [1, 1, 1] })
      const s = d.sample([100])
      expect(s.shape).toEqual([100, 3])
      s.dispose()
      d.dispose()
    })

    test('sample statistics', () => {
      const loc = [1, -2, 3]
      const scaleDiag = [0.5, 1, 2]
      const d = new MultivariateNormalDiag({ loc, scaleDiag })
      const s = d.sample([50000])
      const data = s.dataSync()

      // Check each dimension independently
      for (let dim = 0; dim < 3; dim++) {
        const dimData = []
        for (let i = 0; i < 50000; i++) {
          dimData.push(data[i * 3 + dim])
        }
        const stats = sampleStats(dimData)
        expectClose(stats.mean, loc[dim], { atol: 0.05 })
        expectClose(stats.variance, scaleDiag[dim] ** 2, { atol: 0.2 })
      }

      s.dispose()
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy of 2D standard MVN', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [1, 1] })
      const h = d.entropy()
      // Sum of individual entropies: 2 * 0.5 * log(2πe)
      const expected = 2 * 0.5 * Math.log(2 * Math.PI * Math.E)
      expectClose(h.dataSync()[0], expected, { atol: 1e-4 })
      h.dispose()
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean returns loc', () => {
      const d = new MultivariateNormalDiag({ loc: [1, 2, 3], scaleDiag: [1, 1, 1] })
      const m = d.mean()
      expect(Array.from(m.dataSync())).toEqual([1, 2, 3])
      m.dispose()
      d.dispose()
    })

    test('variance returns scaleDiag^2', () => {
      const d = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [2, 3] })
      const v = d.variance()
      const data = v.dataSync()
      expect(data[0]).toBeCloseTo(4, 4)
      expect(data[1]).toBeCloseTo(9, 4)
      v.dispose()
      d.dispose()
    })
  })

  describe('dispose', () => {
    test('frees memory', () => {
      const before = tf.memory().numTensors
      const d = new MultivariateNormalDiag({ loc: [0, 0], scaleDiag: [1, 1] })
      expect(tf.memory().numTensors).toBeGreaterThan(before)
      d.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
