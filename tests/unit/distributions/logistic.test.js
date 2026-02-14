import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Logistic } from '../../../src/distributions/logistic'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Logistic distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new Logistic()
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new Logistic({ loc: 2, scale: 0.5 })
      expect(d.loc.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(0.5)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new Logistic({ scale: 0 })).toThrow()
      expect(() => new Logistic({ scale: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('at loc', () => {
      const d = new Logistic()
      const lp = d.logProb(0)
      // -log(scale) - 2*log(2) = -2*log(2)
      expectClose(lp.dataSync()[0], -2 * Math.log(2), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('symmetry', () => {
      const d = new Logistic()
      const lp = d.logProb(tf.tensor([-1, 1]))
      const data = lp.dataSync()
      expectClose(data[0], data[1], { atol: 1e-6 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at loc = 0.5', () => {
      const d = new Logistic()
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })

    test('CDF uses sigmoid', () => {
      const d = new Logistic({ loc: 2, scale: 3 })
      const c = d.cdf(5)
      // sigmoid((5-2)/3) = sigmoid(1)
      expectClose(c.dataSync()[0], 1 / (1 + Math.exp(-1)), { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean equals loc', () => {
      const d = new Logistic({ loc: 3 })
      expectClose(d.mean().dataSync()[0], 3, { atol: 1e-6 })
      d.dispose()
    })

    test('variance = (pi*scale)^2/3', () => {
      const d = new Logistic({ scale: 2 })
      expectClose(d.variance().dataSync()[0], Math.PI * Math.PI * 4 / 3, { atol: 1e-3 })
      d.dispose()
    })

    test('mode equals loc', () => {
      const d = new Logistic({ loc: 5 })
      expectClose(d.mode().dataSync()[0], 5, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard Logistic entropy', () => {
      const d = new Logistic()
      expectClose(d.entropy().dataSync()[0], 2, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Logistic()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new Logistic({ loc: 1, scale: 2 })
      const s = d.sample([50000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 50000, Math.PI * Math.PI * 4 / 3)
      expectClose(stats.mean, 1, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/logistic.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Logistic({ loc: tc.params.loc, scale: tc.params.scale })
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
        const d = new Logistic({ loc: tc.params.loc, scale: tc.params.scale })
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
        const d = new Logistic({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.mean().dataSync()[0], tc.mean, { rtol: 1e-4, atol: 1e-5 })
        expectClose(d.variance().dataSync()[0], tc.variance, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Logistic({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-4, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
