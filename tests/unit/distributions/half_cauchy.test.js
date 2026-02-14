import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { HalfCauchy } from '../../../src/distributions/half_cauchy'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('HalfCauchy distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new HalfCauchy()
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new HalfCauchy({ scale: 3 })
      expect(d.scale.dataSync()[0]).toBe(3)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new HalfCauchy({ scale: 0 })).toThrow()
      expect(() => new HalfCauchy({ scale: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('at 0', () => {
      const d = new HalfCauchy()
      const lp = d.logProb(0)
      // log(2/pi) ~ -0.4516
      expectClose(lp.dataSync()[0], Math.log(2 / Math.PI), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('-Infinity for negative values', () => {
      const d = new HalfCauchy()
      const lp = d.logProb(-1)
      expect(lp.dataSync()[0]).toBe(-Infinity)
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at 0 = 0', () => {
      const d = new HalfCauchy()
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF at scale = 2/pi*atan(1) = 0.5', () => {
      const d = new HalfCauchy()
      const c = d.cdf(1)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF for negative = 0', () => {
      const d = new HalfCauchy()
      const c = d.cdf(-1)
      expectClose(c.dataSync()[0], 0, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean is NaN', () => {
      const d = new HalfCauchy()
      expect(d.mean().dataSync()[0]).toBeNaN()
      d.dispose()
    })

    test('variance is NaN', () => {
      const d = new HalfCauchy()
      expect(d.variance().dataSync()[0]).toBeNaN()
      d.dispose()
    })

    test('mode = 0', () => {
      const d = new HalfCauchy({ scale: 3 })
      expectClose(d.mode().dataSync()[0], 0, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard HalfCauchy entropy', () => {
      const d = new HalfCauchy()
      expectClose(d.entropy().dataSync()[0], Math.log(2 * Math.PI), { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new HalfCauchy()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples non-negative', () => {
      const d = new HalfCauchy()
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
      }
      s.dispose()
      d.dispose()
    })

    test('median near scale', () => {
      const d = new HalfCauchy({ scale: 2 })
      const s = d.sample([10000])
      const data = Array.from(s.dataSync()).sort((a, b) => a - b)
      const median = data[Math.floor(data.length / 2)]
      expectClose(median, 2, { atol: 0.2 })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/halfCauchy.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfCauchy({ scale: tc.params.scale })
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
        const d = new HalfCauchy({ scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-3, atol: 1e-4 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new HalfCauchy({ scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-4, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
