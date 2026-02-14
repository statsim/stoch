import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { HalfNormal } from '../../../src/distributions/half_normal'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('HalfNormal distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new HalfNormal()
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new HalfNormal({ scale: 3 })
      expect(d.scale.dataSync()[0]).toBe(3)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new HalfNormal({ scale: 0 })).toThrow()
      expect(() => new HalfNormal({ scale: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('at 0', () => {
      const d = new HalfNormal()
      const lp = d.logProb(0)
      // log(sqrt(2/pi)) = 0.5*log(2/pi) ~ -0.2258
      expectClose(lp.dataSync()[0], 0.5 * Math.log(2 / Math.PI), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('-Infinity for negative values', () => {
      const d = new HalfNormal()
      const lp = d.logProb(-1)
      expect(lp.dataSync()[0]).toBe(-Infinity)
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at 0 = 0', () => {
      const d = new HalfNormal()
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF for negative = 0', () => {
      const d = new HalfNormal()
      const c = d.cdf(-1)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = scale*sqrt(2/pi)', () => {
      const d = new HalfNormal({ scale: 2 })
      expectClose(d.mean().dataSync()[0], 2 * Math.sqrt(2 / Math.PI), { atol: 1e-5 })
      d.dispose()
    })

    test('variance = scale^2*(1-2/pi)', () => {
      const d = new HalfNormal({ scale: 2 })
      expectClose(d.variance().dataSync()[0], 4 * (1 - 2 / Math.PI), { atol: 1e-4 })
      d.dispose()
    })

    test('mode = 0', () => {
      const d = new HalfNormal({ scale: 3 })
      expectClose(d.mode().dataSync()[0], 0, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard HalfNormal entropy', () => {
      const d = new HalfNormal()
      expectClose(d.entropy().dataSync()[0], 0.5 * Math.log(Math.PI / 2) + 0.5, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new HalfNormal()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples non-negative', () => {
      const d = new HalfNormal()
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
      }
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new HalfNormal({ scale: 2 })
      const s = d.sample([50000])
      const stats = sampleStats(s.dataSync())
      const expected = 2 * Math.sqrt(2 / Math.PI)
      const variance = 4 * (1 - 2 / Math.PI)
      const tol = autoTolerance('mean', 50000, variance)
      expectClose(stats.mean, expected, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/halfNormal.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfNormal({ scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const lp = d.logProb(pt.x)
          expectClose(lp.dataSync()[0], pt.logProb, { rtol: 1e-4, atol: 1e-4 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('cdf matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfNormal({ scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-3, atol: 1e-4 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfNormal({ scale: tc.params.scale })
        expectClose(d.mean().dataSync()[0], tc.mean, { rtol: 1e-4, atol: 1e-5 })
        expectClose(d.variance().dataSync()[0], tc.variance, { rtol: 1e-3, atol: 1e-4 })
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfNormal({ scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-4, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
