import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Geometric } from '../../../src/distributions/geometric'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Geometric distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Geometric({ probs: 0.5 })
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-6 })
      d.dispose()
    })

    test('logits param', () => {
      const d = new Geometric({ logits: 0 })
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-5 })
      d.dispose()
    })

    test('throws without probs or logits', () => {
      expect(() => new Geometric({})).toThrow()
    })
  })

  describe('logProb', () => {
    test('Geometric(0.5) at k=0', () => {
      const d = new Geometric({ probs: 0.5 })
      const lp = d.logProb(0)
      // P(0) = p = 0.5
      expectClose(lp.dataSync()[0], Math.log(0.5), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('Geometric(0.5) at k=1', () => {
      const d = new Geometric({ probs: 0.5 })
      const lp = d.logProb(1)
      // P(1) = p*(1-p) = 0.25
      expectClose(lp.dataSync()[0], Math.log(0.25), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at k=0', () => {
      const d = new Geometric({ probs: 0.5 })
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF at k=-1 is 0', () => {
      const d = new Geometric({ probs: 0.5 })
      const c = d.cdf(-1)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = (1-p)/p', () => {
      const d = new Geometric({ probs: 0.5 })
      expectClose(d.mean().dataSync()[0], 1, { atol: 1e-5 })
      d.dispose()
    })

    test('variance = (1-p)/p^2', () => {
      const d = new Geometric({ probs: 0.5 })
      expectClose(d.variance().dataSync()[0], 2, { atol: 1e-4 })
      d.dispose()
    })

    test('mode = 0', () => {
      const d = new Geometric({ probs: 0.3 })
      expectClose(d.mode().dataSync()[0], 0, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('Geometric(0.5) entropy', () => {
      const d = new Geometric({ probs: 0.5 })
      // H = [-(1-p)log(1-p) - p*log(p)] / p
      const expected = (-0.5 * Math.log(0.5) - 0.5 * Math.log(0.5)) / 0.5
      expectClose(d.entropy().dataSync()[0], expected, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Geometric({ probs: 0.5 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples non-negative integers', () => {
      const d = new Geometric({ probs: 0.5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
        expect(data[i] % 1).toBe(0)
      }
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new Geometric({ probs: 0.5 })
      const s = d.sample([20000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 20000, 2)
      expectClose(stats.mean, 1, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/geometric.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Geometric({ probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const lp = d.logProb(tc.points[i])
          expectClose(lp.dataSync()[0], tc.expected.log_prob[i], { rtol: 1e-3, atol: 1e-3 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('cdf matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Geometric({ probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const c = d.cdf(tc.points[i])
          expectClose(c.dataSync()[0], tc.expected.cdf[i], { rtol: 1e-3, atol: 1e-3 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Geometric({ probs: tc.params.probs })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-3 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-2 })
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Geometric({ probs: tc.params.probs })
        expectClose(d.entropy().dataSync()[0], tc.expected.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
