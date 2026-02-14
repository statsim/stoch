import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { InverseGamma } from '../../../src/distributions/inverse_gamma'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('InverseGamma distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new InverseGamma({ concentration: 2, scale: 1 })
      expect(d.concentration.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('throws for non-positive params', () => {
      expect(() => new InverseGamma({ concentration: 0, scale: 1 })).toThrow()
      expect(() => new InverseGamma({ concentration: 2, scale: 0 })).toThrow()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = scale/(a-1) for a > 1', () => {
      const d = new InverseGamma({ concentration: 2, scale: 1 })
      expectClose(d.mean().dataSync()[0], 1, { atol: 1e-4 })
      d.dispose()
    })

    test('mean is NaN for a <= 1', () => {
      const d = new InverseGamma({ concentration: 1, scale: 0.5 })
      expect(d.mean().dataSync()[0]).toBeNaN()
      d.dispose()
    })

    test('variance = scale^2/((a-1)^2*(a-2)) for a > 2', () => {
      const d = new InverseGamma({ concentration: 5, scale: 2 })
      // 4 / (16 * 3) = 4/48 = 0.08333
      expectClose(d.variance().dataSync()[0], 4 / 48, { atol: 1e-3 })
      d.dispose()
    })

    test('mode = scale/(a+1)', () => {
      const d = new InverseGamma({ concentration: 2, scale: 1 })
      expectClose(d.mode().dataSync()[0], 1 / 3, { atol: 1e-4 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new InverseGamma({ concentration: 2, scale: 1 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples positive', () => {
      const d = new InverseGamma({ concentration: 2, scale: 1 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/inverseGamma.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new InverseGamma({ concentration: tc.params.concentration, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const lp = d.logProb(pt.x)
          expectClose(lp.dataSync()[0], pt.logProb, { rtol: 1e-3, atol: 1e-3 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('cdf matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new InverseGamma({ concentration: tc.params.concentration, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          if (pt.cdf < 1e-10) continue
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-2, atol: 1e-3 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new InverseGamma({ concentration: tc.params.concentration, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
