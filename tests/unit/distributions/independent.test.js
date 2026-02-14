import * as tf from '@tensorflow/tfjs'
import { Normal } from '../../../src/distributions/normal'
import { Bernoulli } from '../../../src/distributions/bernoulli'
import { Independent } from '../../../src/distributions/independent'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('Independent distribution', () => {
  describe('batchShape and eventShape', () => {
    test('reinterprets 1 batch dim', () => {
      const base = new Normal({ loc: [0, 0, 0], scale: 1 })
      expect(base.batchShape).toEqual([3])
      expect(base.eventShape).toEqual([])

      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })
      expect(ind.batchShape).toEqual([])
      expect(ind.eventShape).toEqual([3])
      ind.dispose()
    })

    test('reinterprets with higher-rank batch', () => {
      const base = new Normal({ loc: [[0, 1], [2, 3]], scale: 1 })
      expect(base.batchShape).toEqual([2, 2])

      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })
      expect(ind.batchShape).toEqual([2])
      expect(ind.eventShape).toEqual([2])
      ind.dispose()
    })

    test('reinterpret 0 dims is no-op', () => {
      const base = new Normal({ loc: [0, 1], scale: 1 })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 0 })
      expect(ind.batchShape).toEqual([2])
      expect(ind.eventShape).toEqual([])
      ind.dispose()
    })
  })

  describe('logProb', () => {
    test('sums logProb over reinterpreted dims', () => {
      const base = new Normal({ loc: [0, 0], scale: [1, 1] })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      // logProb should be scalar (sum of two independent Normal logProbs)
      const lp = ind.logProb(tf.tensor([0, 0]))
      expect(lp.shape).toEqual([])

      // Should equal sum of individual logProbs
      const baseLp = base.logProb(tf.tensor([0, 0]))
      const sumBaseLp = baseLp.dataSync().reduce((a, b) => a + b, 0)
      expectClose(lp.dataSync()[0], sumBaseLp, { atol: 1e-5 })

      lp.dispose()
      baseLp.dispose()
      ind.dispose()
    })

    test('batched logProb', () => {
      // 3 independent 2-dim Normal distributions
      const base = new Normal({ loc: [[0, 0], [1, 1], [2, 2]], scale: 1 })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })
      expect(ind.batchShape).toEqual([3])

      const values = tf.tensor([[0, 0], [1, 1], [2, 2]])
      const lp = ind.logProb(values)
      expect(lp.shape).toEqual([3])

      // Each should be sum of two Normal(mean, 1) logProbs at mean
      const logProbAtMean = -0.5 * Math.log(2 * Math.PI)
      const data = lp.dataSync()
      for (let i = 0; i < 3; i++) {
        expectClose(data[i], 2 * logProbAtMean, { atol: 1e-4 })
      }

      lp.dispose()
      values.dispose()
      ind.dispose()
    })
  })

  describe('sample', () => {
    test('sample shape with reinterpreted dims', () => {
      const base = new Normal({ loc: [0, 0, 0], scale: 1 })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const s = ind.sample([100])
      expect(s.shape).toEqual([100, 3])
      s.dispose()
      ind.dispose()
    })

    test('sample shape with batched reinterpreted', () => {
      const base = new Normal({ loc: [[0, 1], [2, 3]], scale: 1 })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const s = ind.sample([50])
      expect(s.shape).toEqual([50, 2, 2])
      s.dispose()
      ind.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy sums over reinterpreted dims', () => {
      const base = new Normal({ loc: [0, 0], scale: [1, 2] })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const h = ind.entropy()
      expect(h.shape).toEqual([])

      // Sum of individual entropies
      const baseH = base.entropy()
      const sumH = baseH.dataSync().reduce((a, b) => a + b, 0)
      expectClose(h.dataSync()[0], sumH, { atol: 1e-4 })

      h.dispose()
      baseH.dispose()
      ind.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean is same as base', () => {
      const base = new Normal({ loc: [1, 2, 3], scale: 1 })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const m = ind.mean()
      expect(Array.from(m.dataSync())).toEqual([1, 2, 3])
      m.dispose()
      ind.dispose()
    })

    test('variance is same as base', () => {
      const base = new Normal({ loc: 0, scale: [1, 2, 3] })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const v = ind.variance()
      const data = v.dataSync()
      expect(data[0]).toBeCloseTo(1, 4)
      expect(data[1]).toBeCloseTo(4, 4)
      expect(data[2]).toBeCloseTo(9, 4)
      v.dispose()
      ind.dispose()
    })
  })

  describe('with Bernoulli (discrete)', () => {
    test('logProb sums for independent coin flips', () => {
      const base = new Bernoulli({ probs: [0.3, 0.7, 0.5] })
      const ind = new Independent({ distribution: base, reinterpretedBatchNdims: 1 })

      const lp = ind.logProb(tf.tensor([1, 1, 0]))
      expect(lp.shape).toEqual([])

      // log(0.3) + log(0.7) + log(0.5)
      const expected = Math.log(0.3) + Math.log(0.7) + Math.log(0.5)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })

      lp.dispose()
      ind.dispose()
    })
  })

  describe('dispose', () => {
    test('disposes base distribution', () => {
      const before = tf.memory().numTensors
      const ind = new Independent({
        distribution: new Normal({ loc: [0, 0], scale: [1, 1] }),
        reinterpretedBatchNdims: 1
      })
      expect(tf.memory().numTensors).toBeGreaterThan(before)
      ind.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
