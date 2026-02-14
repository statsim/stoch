import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Cauchy } from '../../../src/distributions/cauchy'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Cauchy distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new Cauchy()
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('custom params', () => {
      const d = new Cauchy({ loc: 2, scale: 3 })
      expect(d.loc.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(3)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new Cauchy({ scale: 0 })).toThrow()
      expect(() => new Cauchy({ scale: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('standard Cauchy at 0', () => {
      const d = new Cauchy()
      const lp = d.logProb(0)
      // log(1/(π*1)) = -log(π)
      expectClose(lp.dataSync()[0], -Math.log(Math.PI), { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })

    test('symmetry', () => {
      const d = new Cauchy()
      const lp = d.logProb(tf.tensor([-2, 2]))
      const data = lp.dataSync()
      expectClose(data[0], data[1], { atol: 1e-6 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('cdf', () => {
    test('CDF at loc = 0.5', () => {
      const d = new Cauchy()
      const c = d.cdf(0)
      expectClose(c.dataSync()[0], 0.5, { atol: 1e-5 })
      c.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean is NaN', () => {
      const d = new Cauchy()
      expect(d.mean().dataSync()[0]).toBeNaN()
      d.dispose()
    })

    test('variance is NaN', () => {
      const d = new Cauchy()
      expect(d.variance().dataSync()[0]).toBeNaN()
      d.dispose()
    })

    test('mode equals loc', () => {
      const d = new Cauchy({ loc: 5 })
      expectClose(d.mode().dataSync()[0], 5, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('standard Cauchy entropy', () => {
      const d = new Cauchy()
      expectClose(d.entropy().dataSync()[0], Math.log(4 * Math.PI), { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Cauchy()
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('median of samples near loc', () => {
      const d = new Cauchy({ loc: 3, scale: 1 })
      const s = d.sample([10000])
      const data = Array.from(s.dataSync()).sort((a, b) => a - b)
      const median = data[Math.floor(data.length / 2)]
      expectClose(median, 3, { atol: 0.1 })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/cauchy.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Cauchy({ loc: tc.params.loc, scale: tc.params.scale })
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
        const d = new Cauchy({ loc: tc.params.loc, scale: tc.params.scale })
        for (const pt of tc.test_points) {
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-4, atol: 1e-4 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Cauchy({ loc: tc.params.loc, scale: tc.params.scale })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-4, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
