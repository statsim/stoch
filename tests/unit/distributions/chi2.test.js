import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Chi2 } from '../../../src/distributions/chi2'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Chi2 distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new Chi2()
      expectClose(d.df.dataSync()[0], 1, { atol: 1e-6 })
      d.dispose()
    })

    test('custom df', () => {
      const d = new Chi2({ df: 5 })
      expectClose(d.df.dataSync()[0], 5, { atol: 1e-6 })
      d.dispose()
    })

    test('is a Gamma distribution', () => {
      const d = new Chi2({ df: 4 })
      // Chi2(4) = Gamma(2, 0.5)
      expectClose(d.concentration.dataSync()[0], 2, { atol: 1e-6 })
      expectClose(d.rate.dataSync()[0], 0.5, { atol: 1e-6 })
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean = df', () => {
      const d = new Chi2({ df: 5 })
      expectClose(d.mean().dataSync()[0], 5, { atol: 1e-4 })
      d.dispose()
    })

    test('variance = 2*df', () => {
      const d = new Chi2({ df: 5 })
      expectClose(d.variance().dataSync()[0], 10, { atol: 1e-3 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Chi2({ df: 3 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample mean converges', () => {
      const d = new Chi2({ df: 5 })
      const s = d.sample([50000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 50000, 10)
      expectClose(stats.mean, 5, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/chi2.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Chi2({ df: tc.params.df })
        for (const pt of tc.test_points) {
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
        const d = new Chi2({ df: tc.params.df })
        for (const pt of tc.test_points) {
          if (pt.cdf < 1e-6) continue // skip extreme tail
          const c = d.cdf(pt.x)
          expectClose(c.dataSync()[0], pt.cdf, { rtol: 1e-2, atol: 1e-2 })
          c.dispose()
        }
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Chi2({ df: tc.params.df })
        expectClose(d.mean().dataSync()[0], tc.mean, { rtol: 1e-3, atol: 1e-3 })
        expectClose(d.variance().dataSync()[0], tc.variance, { rtol: 1e-2, atol: 1e-2 })
        d.dispose()
      }
    })

    test('entropy matches scipy', () => {
      if (!refData) return
      for (const tc of refData) {
        const d = new Chi2({ df: tc.params.df })
        expectClose(d.entropy().dataSync()[0], tc.entropy, { rtol: 1e-2, atol: 1e-2 })
        d.dispose()
      }
    })
  })
})
