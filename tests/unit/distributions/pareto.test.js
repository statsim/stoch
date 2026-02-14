import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Pareto } from '../../../src/distributions/pareto'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Pareto distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Pareto({ concentration: 2, scale: 1 })
      expect(d.concentration.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('throws for non-positive params', () => {
      expect(() => new Pareto({ concentration: 0, scale: 1 })).toThrow()
      expect(() => new Pareto({ concentration: 2, scale: 0 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('-Infinity below scale', () => {
      const d = new Pareto({ concentration: 2, scale: 1 })
      const lp = d.logProb(0.5)
      expect(lp.dataSync()[0]).toBe(-Infinity)
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = alpha*scale/(alpha-1) for alpha > 1', () => {
      const d = new Pareto({ concentration: 2, scale: 1 })
      expectClose(d.mean().dataSync()[0], 2, { atol: 1e-4 })
      d.dispose()
    })

    test('mode = scale', () => {
      const d = new Pareto({ concentration: 2, scale: 3 })
      expectClose(d.mode().dataSync()[0], 3, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Pareto({ concentration: 2, scale: 1 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples >= scale', () => {
      const d = new Pareto({ concentration: 2, scale: 1 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(1)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/pareto.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Pareto({ concentration: tc.params.concentration, scale: tc.params.scale })
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
        const d = new Pareto({ concentration: tc.params.concentration, scale: tc.params.scale })
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
        const d = new Pareto({ concentration: tc.params.concentration, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
