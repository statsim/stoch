import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { TruncatedNormal } from '../../../src/distributions/truncated_normal'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('TruncatedNormal distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      expect(d.low.dataSync()[0]).toBe(-2)
      expect(d.high.dataSync()[0]).toBe(2)
      d.dispose()
    })

    test('throws for low >= high', () => {
      expect(() => new TruncatedNormal({ loc: 0, scale: 1, low: 2, high: 1 })).toThrow()
    })

    test('throws for non-positive scale', () => {
      expect(() => new TruncatedNormal({ loc: 0, scale: 0, low: -1, high: 1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('-Infinity outside bounds', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -1, high: 1 })
      expect(d.logProb(-2).dataSync()[0]).toBe(-Infinity)
      expect(d.logProb(2).dataSync()[0]).toBe(-Infinity)
      d.dispose()
    })

    test('higher density at loc than at boundaries', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      const lpCenter = d.logProb(0).dataSync()[0]
      const lpEdge = d.logProb(1.5).dataSync()[0]
      expect(lpCenter).toBeGreaterThan(lpEdge)
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at low = 0', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      const c = d.cdf(-2)
      expectClose(c.dataSync()[0], 0, { atol: 1e-4 })
      c.dispose()
      d.dispose()
    })

    test('CDF at high = 1', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      const c = d.cdf(2)
      expectClose(c.dataSync()[0], 1, { atol: 1e-4 })
      c.dispose()
      d.dispose()
    })

    test('CDF at loc near 0.5 for symmetric bounds', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-4 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mode', () => {
    test('mode = loc when loc is within bounds', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      expectClose(d.mode().dataSync()[0], 0, { atol: 1e-5 })
      d.dispose()
    })

    test('mode clamped to low when loc < low', () => {
      const d = new TruncatedNormal({ loc: -5, scale: 1, low: 0, high: 10 })
      expectClose(d.mode().dataSync()[0], 0, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -2, high: 2 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('all samples within bounds', () => {
      const d = new TruncatedNormal({ loc: 0, scale: 1, low: -1, high: 1 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(-1)
        expect(data[i]).toBeLessThanOrEqual(1)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/truncatedNormal.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new TruncatedNormal({
          loc: tc.params.loc, scale: tc.params.scale,
          low: tc.params.low, high: tc.params.high
        })
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
        const d = new TruncatedNormal({
          loc: tc.params.loc, scale: tc.params.scale,
          low: tc.params.low, high: tc.params.high
        })
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
        const d = new TruncatedNormal({
          loc: tc.params.loc, scale: tc.params.scale,
          low: tc.params.low, high: tc.params.high
        })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-2, atol: 1e-2 })
        d.dispose()
      }
    })
  })
})
