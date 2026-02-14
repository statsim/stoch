import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Weibull } from '../../../src/distributions/weibull'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Weibull distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Weibull({ concentration: 2, scale: 1 })
      expect(d.concentration.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('throws for non-positive params', () => {
      expect(() => new Weibull({ concentration: 0, scale: 1 })).toThrow()
      expect(() => new Weibull({ concentration: 1, scale: 0 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('-Infinity for negative x', () => {
      const d = new Weibull({ concentration: 2, scale: 1 })
      const lp = d.logProb(-1)
      expect(lp.dataSync()[0]).toBe(-Infinity)
      lp.dispose()
      d.dispose()
    })

    test('exponential special case (k=1)', () => {
      const d = new Weibull({ concentration: 1, scale: 1 })
      const lp = d.logProb(1)
      // Weibull(1,1) = Exponential(1), logpdf(1) = -1
      expectClose(lp.dataSync()[0], -1, { atol: 1e-4 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at 0 = 0', () => {
      const d = new Weibull({ concentration: 2, scale: 1 })
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Weibull({ concentration: 2, scale: 1 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples non-negative', () => {
      const d = new Weibull({ concentration: 2, scale: 1 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/weibull.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Weibull({ concentration: tc.params.concentration, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          if (pt.logProb === null || pt.logProb < -30) continue
          const lp = d.logProb(pt.x)
          expectClose(lp.dataSync()[0], pt.logProb, { rtol: 1e-2, atol: 1e-2 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('cdf matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Weibull({ concentration: tc.params.concentration, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-3, atol: 1e-3 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Weibull({ concentration: tc.params.concentration, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
