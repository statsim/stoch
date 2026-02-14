import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { VonMises } from '../../../src/distributions/von_mises'
import { besselI0, besselI1 } from '../../../src/math/special'
import { expectClose } from '../../helpers/tolerance'

describe('VonMises distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new VonMises({ loc: 0, concentration: 1 })
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.concentration.dataSync()[0]).toBe(1)
      d.dispose()
    })

    test('throws for negative concentration', () => {
      expect(() => new VonMises({ loc: 0, concentration: -1 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('maximum at loc', () => {
      const d = new VonMises({ loc: 0, concentration: 5 })
      const lpAtLoc = d.logProb(0).dataSync()[0]
      const lpAway = d.logProb(1).dataSync()[0]
      expect(lpAtLoc).toBeGreaterThan(lpAway)
      d.dispose()
    })

    test('symmetric around loc', () => {
      const d = new VonMises({ loc: 0, concentration: 2 })
      const lp = d.logProb(tf.tensor([-1, 1]))
      const data = lp.dataSync()
      expectClose(data[0], data[1], { atol: 1e-5 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance/mode', () => {
    test('mean = loc', () => {
      const d = new VonMises({ loc: 1.5, concentration: 3 })
      expectClose(d.mean().dataSync()[0], 1.5, { atol: 1e-5 })
      d.dispose()
    })

    test('circular variance = 1 - I1/I0', () => {
      const kappa = 2
      const expected = 1 - besselI1(kappa) / besselI0(kappa)
      const d = new VonMises({ loc: 0, concentration: kappa })
      expectClose(d.variance().dataSync()[0], expected, { atol: 1e-5 })
      d.dispose()
    })

    test('mode = loc', () => {
      const d = new VonMises({ loc: 2, concentration: 5 })
      expectClose(d.mode().dataSync()[0], 2, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new VonMises({ loc: 0, concentration: 1 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('samples in [-pi, pi]', () => {
      const d = new VonMises({ loc: 0, concentration: 5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(-Math.PI - 0.01)
        expect(data[i]).toBeLessThanOrEqual(Math.PI + 0.01)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/vonMises.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new VonMises({ loc: tc.params.loc, concentration: tc.params.concentration })
        for (const pt of tc.test_points) {
          const lp = d.logProb(pt.x)
          expectClose(lp.dataSync()[0], pt.logProb, { rtol: 1e-3, atol: 1e-3 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new VonMises({ loc: tc.params.loc, concentration: tc.params.concentration })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
