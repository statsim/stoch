import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Binomial } from '../../../src/distributions/binomial'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Binomial distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      expect(d.totalCount.dataSync()[0]).toBe(10)
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-6 })
      d.dispose()
    })

    test('logits param', () => {
      const d = new Binomial({ totalCount: 10, logits: 0 })
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-5 })
      d.dispose()
    })

    test('throws without probs or logits', () => {
      expect(() => new Binomial({ totalCount: 10 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('Binomial(10, 0.5) at k=5', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      const lp = d.logProb(5)
      // C(10,5) * 0.5^10 = 252/1024 ≈ 0.2461
      expectClose(lp.dataSync()[0], Math.log(252 / 1024), { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })

    test('boundary: k=0', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      const lp = d.logProb(0)
      // (1-p)^n = 0.5^10
      expectClose(lp.dataSync()[0], 10 * Math.log(0.5), { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean = n*p', () => {
      const d = new Binomial({ totalCount: 20, probs: 0.3 })
      expectClose(d.mean().dataSync()[0], 6, { atol: 1e-4 })
      d.dispose()
    })

    test('variance = n*p*(1-p)', () => {
      const d = new Binomial({ totalCount: 20, probs: 0.3 })
      expectClose(d.variance().dataSync()[0], 4.2, { atol: 1e-3 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('samples in [0, n]', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
        expect(data[i]).toBeLessThanOrEqual(10)
      }
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new Binomial({ totalCount: 10, probs: 0.5 })
      const s = d.sample([10000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 10000, 2.5)
      expectClose(stats.mean, 5, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/binomial.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Binomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const lp = d.logProb(tc.points[i])
          expectClose(lp.dataSync()[0], tc.expected.log_prob[i], { rtol: 1e-2, atol: 1e-2 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('cdf matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Binomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          if (tc.expected.cdf[i] < 1e-10) continue
          const c = d.cdf(tc.points[i])
          expectClose(c.dataSync()[0], tc.expected.cdf[i], { rtol: 1e-2, atol: 1e-2 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Binomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-3 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-2 })
        d.dispose()
      }
    })
  })
})
