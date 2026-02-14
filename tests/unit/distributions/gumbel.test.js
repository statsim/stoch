import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Gumbel } from '../../../src/distributions/gumbel'
import { EULER_MASCHERONI } from '../../../src/math/numeric'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Gumbel distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new Gumbel()
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new Gumbel({ loc: 2, scale: 3 })
      expect(d.loc.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(3)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new Gumbel({ scale: 0 })).toThrow()
      expect(() => new Gumbel({ scale: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('at loc', () => {
      const d = new Gumbel()
      const lp = d.logProb(0)
      // z=0: -0 - exp(0) - log(1) = -1
      expectClose(lp.dataSync()[0], -1, { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at loc', () => {
      const d = new Gumbel()
      const c = d.cdf(0)
      // exp(-exp(0)) = exp(-1) = 1/e
      expectClose(c.dataSync()[0], Math.exp(-1), { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = loc + scale*gamma', () => {
      const d = new Gumbel()
      expectClose(d.mean().dataSync()[0], EULER_MASCHERONI, { atol: 1e-5 })
      d.dispose()
    })

    test('variance = pi^2*scale^2/6', () => {
      const d = new Gumbel({ scale: 2 })
      expectClose(d.variance().dataSync()[0], Math.PI * Math.PI * 4 / 6, { atol: 1e-3 })
      d.dispose()
    })

    test('mode equals loc', () => {
      const d = new Gumbel({ loc: 5 })
      expectClose(d.mode().dataSync()[0], 5, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard Gumbel entropy', () => {
      const d = new Gumbel()
      // H = 1 + gamma
      expectClose(d.entropy().dataSync()[0], 1 + EULER_MASCHERONI, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Gumbel()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new Gumbel({ loc: 2, scale: 1 })
      const s = d.sample([50000])
      const stats = sampleStats(s.dataSync())
      const expected = 2 + EULER_MASCHERONI
      const tol = autoTolerance('mean', 50000, Math.PI * Math.PI / 6)
      expectClose(stats.mean, expected, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/gumbel.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Gumbel({ loc: tc.params.loc, scale: tc.params.scale })
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
        const d = new Gumbel({ loc: tc.params.loc, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          if (pt.cdf < 1e-30) continue // skip extreme tail
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
        const d = new Gumbel({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.mean().dataSync()[0], tc.mean, { rtol: 1e-4, atol: 1e-4 })
        expectClose(d.variance().dataSync()[0], tc.variance, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Gumbel({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-4, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
