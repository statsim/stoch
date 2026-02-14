import * as tf from '@tensorflow/tfjs'
import { MixtureSameFamily } from '../../../src/distributions/mixture_same_family'
import { Normal } from '../../../src/distributions/normal'
import { Categorical } from '../../../src/distributions/categorical'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('MixtureSameFamily distribution', () => {
  describe('constructor', () => {
    test('basic construction', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.3, 0.7] }),
        componentDist: new Normal({ loc: [-1, 1], scale: [0.5, 0.5] })
      })
      expect(d.name).toBe('MixtureSameFamily')
      expect(d.numComponents).toBe(2)
      d.dispose()
    })

    test('batch and event shapes', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.3, 0.3, 0.4] }),
        componentDist: new Normal({ loc: [-1, 0, 1], scale: [0.5, 0.5, 0.5] })
      })
      // componentDist batchShape = [3], mixture eats last dim
      expect(d.batchShape).toEqual([])
      expect(d.eventShape).toEqual([])
      d.dispose()
    })

    test('custom name', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [0, 0], scale: [1, 1] }),
        name: 'MyMixture'
      })
      expect(d.name).toBe('MyMixture')
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb for two-component mixture', () => {
      // p(x) = 0.5 * N(x; -2, 0.5) + 0.5 * N(x; 2, 0.5)
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-2, 2], scale: [0.5, 0.5] })
      })

      // At x=2, the second component dominates
      const lp = d.logProb(tf.scalar(2))
      // p(2) ≈ 0.5 * N(2; -2, 0.5) + 0.5 * N(2; 2, 0.5)
      // N(2; 2, 0.5) = 1/(0.5*sqrt(2π)) ≈ 0.7979
      // N(2; -2, 0.5) ≈ 0 (far from mean)
      // p(2) ≈ 0.5 * 0.7979 ≈ 0.3989
      const normalAtMean = 1 / (0.5 * Math.sqrt(2 * Math.PI))
      const expected = Math.log(0.5 * normalAtMean)
      expectClose(lp.dataSync()[0], expected, { atol: 0.01 })

      lp.dispose()
      d.dispose()
    })

    test('logProb at midpoint of equal-weight mixture', () => {
      // p(x) = 0.5 * N(x; -1, 1) + 0.5 * N(x; 1, 1)
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-1, 1], scale: [1, 1] })
      })

      // At x=0: both components contribute equally
      const lp = d.logProb(tf.scalar(0))
      // p(0) = 0.5 * N(0; -1, 1) + 0.5 * N(0; 1, 1)
      // N(0; -1, 1) = N(0; 1, 1) = exp(-0.5) / sqrt(2π)
      const nAt1 = Math.exp(-0.5) / Math.sqrt(2 * Math.PI)
      const expected = Math.log(2 * 0.5 * nAt1)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })

      lp.dispose()
      d.dispose()
    })

    test('logProb with logits parameterization', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ logits: [0, 0, 0] }),
        componentDist: new Normal({ loc: [-2, 0, 2], scale: [0.5, 0.5, 0.5] })
      })

      // At x=0, the middle component dominates
      const lp = d.logProb(tf.scalar(0))
      // p(0) = (1/3) * N(0; -2, 0.5) + (1/3) * N(0; 0, 0.5) + (1/3) * N(0; 2, 0.5)
      const n0 = Math.exp(-0) / (0.5 * Math.sqrt(2 * Math.PI))
      const n2 = Math.exp(-0.5 * 4 / 0.25) / (0.5 * Math.sqrt(2 * Math.PI))
      const expected = Math.log((1 / 3) * n2 + (1 / 3) * n0 + (1 / 3) * n2)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })

      lp.dispose()
      d.dispose()
    })

    test('logProb with tensor input', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-1, 1], scale: [1, 1] })
      })
      const lp = d.logProb(tf.tensor([-2, -1, 0, 1, 2]))
      expect(lp.shape).toEqual([5])
      lp.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('sample shape', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-1, 1], scale: [0.5, 0.5] })
      })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample statistics for bimodal mixture', () => {
      // p(x) = 0.5 * N(x; -3, 1) + 0.5 * N(x; 3, 1)
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-3, 3], scale: [1, 1] })
      })
      const s = d.sample([50000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      // Mean should be 0 (symmetric mixture)
      expectClose(stats.mean, 0, { atol: 0.15 })
      // Variance = 0.5*(1 + 9) + 0.5*(1 + 9) - 0 = 10
      expectClose(stats.variance, 10, { atol: 0.5 })

      s.dispose()
      d.dispose()
    })

    test('sample statistics for asymmetric mixture', () => {
      // p(x) = 0.3 * N(x; 0, 1) + 0.7 * N(x; 5, 1)
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.3, 0.7] }),
        componentDist: new Normal({ loc: [0, 5], scale: [1, 1] })
      })
      const s = d.sample([50000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      // Mean = 0.3*0 + 0.7*5 = 3.5
      expectClose(stats.mean, 3.5, { atol: 0.15 })

      s.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean of symmetric mixture', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-2, 2], scale: [1, 1] })
      })
      const m = d.mean()
      expectClose(m.dataSync()[0], 0, { atol: 1e-5 })
      m.dispose()
      d.dispose()
    })

    test('mean of weighted mixture', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.3, 0.7] }),
        componentDist: new Normal({ loc: [0, 10], scale: [1, 1] })
      })
      const m = d.mean()
      // E[X] = 0.3*0 + 0.7*10 = 7
      expectClose(m.dataSync()[0], 7, { atol: 1e-3 })
      m.dispose()
      d.dispose()
    })
  })

  describe('variance', () => {
    test('variance of symmetric mixture', () => {
      // p(x) = 0.5 * N(x; -2, 1) + 0.5 * N(x; 2, 1)
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-2, 2], scale: [1, 1] })
      })
      const v = d.variance()
      // Var = 0.5*(1 + 4) + 0.5*(1 + 4) - 0² = 5
      expectClose(v.dataSync()[0], 5, { atol: 1e-3 })
      v.dispose()
      d.dispose()
    })

    test('variance reduces to component variance for single component', () => {
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [1] }),
        componentDist: new Normal({ loc: [3], scale: [2] })
      })
      const v = d.variance()
      expectClose(v.dataSync()[0], 4, { atol: 1e-3 })
      v.dispose()
      d.dispose()
    })
  })

  describe('dispose', () => {
    test('disposes mixture and component distributions', () => {
      const before = tf.memory().numTensors
      const d = new MixtureSameFamily({
        mixtureDist: new Categorical({ probs: [0.5, 0.5] }),
        componentDist: new Normal({ loc: [-1, 1], scale: [1, 1] })
      })
      expect(tf.memory().numTensors).toBeGreaterThan(before)
      d.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
