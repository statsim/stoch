import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Normal } from '../../../src/distributions/normal'
import { expectClose, expectTensorClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Normal distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new Normal()
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new Normal({ loc: 5, scale: 2 })
      expect(d.loc.dataSync()[0]).toBe(5)
      expect(d.scale.dataSync()[0]).toBe(2)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new Normal({ scale: 0 })).toThrow('must be positive')
      expect(() => new Normal({ scale: -1 })).toThrow('must be positive')
    })

    test('batched params', () => {
      const d = new Normal({ loc: [0, 1, 2], scale: 1 })
      expect(d.batchShape).toEqual([3])
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('standard normal at 0', () => {
      const d = new Normal()
      const lp = d.logProb(0)
      // log(1/sqrt(2π)) = -0.5 * log(2π)
      expectClose(lp.dataSync()[0], -0.5 * Math.log(2 * Math.PI), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('logProb for N(5, 0.5) at x=5', () => {
      const d = new Normal({ loc: 5, scale: 0.5 })
      const lp = d.logProb(5)
      // At the mean: -0.5 * log(2π) - log(0.5)
      const expected = -0.5 * Math.log(2 * Math.PI) - Math.log(0.5)
      expectClose(lp.dataSync()[0], expected, { rtol: 1e-4 })
      lp.dispose()
      d.dispose()
    })

    test('logProb with tensor input', () => {
      const d = new Normal()
      const lp = d.logProb(tf.tensor([-1, 0, 1]))
      expect(lp.shape).toEqual([3])
      // Symmetry: logProb(-1) == logProb(1) for standard normal
      const data = lp.dataSync()
      expectClose(data[0], data[2], { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('prob', () => {
    test('standard normal at 0', () => {
      const d = new Normal()
      const p = d.prob(0)
      expectClose(p.dataSync()[0], 1 / Math.sqrt(2 * Math.PI), { rtol: 1e-4 })
      p.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('standard normal CDF at 0 = 0.5', () => {
      const d = new Normal()
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF at +/- 1.96', () => {
      const d = new Normal()
      const c = d.cdf(1.96)
      expectClose(c.dataSync()[0], 0.975, { atol: 1e-3 })
      c.dispose()
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard normal entropy', () => {
      const d = new Normal()
      const h = d.entropy()
      // H = 0.5 * log(2πe) ≈ 1.4189
      expectClose(h.dataSync()[0], 0.5 * Math.log(2 * Math.PI * Math.E), { rtol: 1e-4 })
      h.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/stddev/mode', () => {
    test('mean equals loc', () => {
      const d = new Normal({ loc: 3, scale: 2 })
      expectClose(d.mean().dataSync()[0], 3, { atol: 1e-6 })
      d.dispose()
    })

    test('variance equals scale^2', () => {
      const d = new Normal({ loc: 0, scale: 3 })
      expectClose(d.variance().dataSync()[0], 9, { atol: 1e-4 })
      d.dispose()
    })

    test('stddev equals scale', () => {
      const d = new Normal({ loc: 0, scale: 2.5 })
      expectClose(d.stddev().dataSync()[0], 2.5, { atol: 1e-6 })
      d.dispose()
    })

    test('mode equals loc', () => {
      const d = new Normal({ loc: 7, scale: 1 })
      expectClose(d.mode().dataSync()[0], 7, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Normal()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('batched sample shape', () => {
      const d = new Normal({ loc: [0, 1, 2], scale: 1 })
      const s = d.sample([50])
      expect(s.shape).toEqual([50, 3])
      s.dispose()
      d.dispose()
    })

    test('sample statistics match theoretical values', () => {
      const d = new Normal({ loc: 5, scale: 2 })
      const s = d.sample([100000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      const tolMean = autoTolerance('mean', 100000, 4)
      const tolVar = autoTolerance('variance', 100000, 4, 48) // kurtosis of normal = 3, m4 = 3*σ⁴ = 48

      expectClose(stats.mean, 5, { atol: tolMean })
      expectClose(stats.variance, 4, { atol: tolVar })

      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/normal.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Normal({ loc: tc.params.loc, scale: tc.params.scale })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()

        for (let i = 0; i < tc.points.length; i++) {
          expectClose(data[i], tc.expected.log_prob[i], { rtol: 1e-3, atol: 1e-3 })
        }

        lp.dispose()
        points.dispose()
        d.dispose()
      }
    })

    test('cdf matches scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Normal({ loc: tc.params.loc, scale: tc.params.scale })
        const points = tf.tensor(tc.points)
        const c = d.cdf(points)
        const data = c.dataSync()

        for (let i = 0; i < tc.points.length; i++) {
          expectClose(data[i], tc.expected.cdf[i], { rtol: 1e-3, atol: 1e-3 })
        }

        c.dispose()
        points.dispose()
        d.dispose()
      }
    })

    test('mean/variance match scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Normal({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-4, atol: 1e-5 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-3, atol: 1e-5 })
        d.dispose()
      }
    })

    test('entropy matches scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Normal({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.expected.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })

  test('dispose frees memory', () => {
    // Warm up tf.js by forcing a tensor operation first
    const warmup = tf.scalar(0)
    warmup.dispose()

    const before = tf.memory().numTensors
    const d = new Normal({ loc: 0, scale: 1 })
    const created = tf.memory().numTensors - before
    d.dispose()
    const afterDispose = tf.memory().numTensors
    expect(afterDispose).toBeLessThanOrEqual(before)
  })
})
