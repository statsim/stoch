import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { NegativeBinomial } from '../../../src/distributions/negative_binomial'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('NegativeBinomial distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      expect(d.totalCount.dataSync()[0]).toBe(5)
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-6 })
      d.dispose()
    })

    test('logits param', () => {
      const d = new NegativeBinomial({ totalCount: 5, logits: 0 })
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-5 })
      d.dispose()
    })

    test('throws without probs or logits', () => {
      expect(() => new NegativeBinomial({ totalCount: 5 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('NegBinomial(5, 0.5) at k=0', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      const lp = d.logProb(0)
      // P(0) = p^r = 0.5^5 = 0.03125
      expectClose(lp.dataSync()[0], Math.log(0.03125), { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean = r*(1-p)/p', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      expectClose(d.mean().dataSync()[0], 5, { atol: 1e-4 })
      d.dispose()
    })

    test('variance = r*(1-p)/p^2', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      expectClose(d.variance().dataSync()[0], 10, { atol: 1e-3 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples non-negative', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
      }
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new NegativeBinomial({ totalCount: 5, probs: 0.5 })
      const s = d.sample([10000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 10000, 10)
      expectClose(stats.mean, 5, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/negativeBinomial.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new NegativeBinomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const lp = d.logProb(tc.points[i])
          expectClose(lp.dataSync()[0], tc.expected.log_prob[i], { rtol: 1e-2, atol: 1e-2 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new NegativeBinomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-2, atol: 1e-2 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-1 })
        d.dispose()
      }
    })
  })
})
