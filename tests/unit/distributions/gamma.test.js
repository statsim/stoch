import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Gamma } from '../../../src/distributions/gamma'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Gamma distribution', () => {
  test('mean = concentration / rate', () => {
    const d = new Gamma({ concentration: 2, rate: 0.5 })
    expectClose(d.mean().dataSync()[0], 4, { atol: 1e-4 })
    d.dispose()
  })

  test('variance = concentration / rate^2', () => {
    const d = new Gamma({ concentration: 2, rate: 0.5 })
    expectClose(d.variance().dataSync()[0], 8, { atol: 1e-3 })
    d.dispose()
  })

  test('throws for non-positive params', () => {
    expect(() => new Gamma({ concentration: 0, rate: 1 })).toThrow()
    expect(() => new Gamma({ concentration: 1, rate: 0 })).toThrow()
  })

  test('logProb at mean', () => {
    const d = new Gamma({ concentration: 1, rate: 1 })
    // Gamma(1,1) = Exponential(1), logpdf(1) = -1
    const lp = d.logProb(1)
    expectClose(lp.dataSync()[0], -1, { atol: 1e-4 })
    lp.dispose()
    d.dispose()
  })

  test('mode for concentration >= 1', () => {
    const d = new Gamma({ concentration: 5, rate: 2 })
    expectClose(d.mode().dataSync()[0], 2, { atol: 1e-4 }) // (5-1)/2
    d.dispose()
  })

  test('mode for concentration < 1 is 0', () => {
    const d = new Gamma({ concentration: 0.5, rate: 1 })
    expectClose(d.mode().dataSync()[0], 0, { atol: 1e-4 })
    d.dispose()
  })

  test('sample mean converges', () => {
    const d = new Gamma({ concentration: 2, rate: 0.5 })
    const s = d.sample([50000])
    const stats = sampleStats(s.dataSync())
    const tol = autoTolerance('mean', 50000, 8)
    expectClose(stats.mean, 4, { atol: tol })
    s.dispose()
    d.dispose()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/gamma.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Gamma({ concentration: tc.params.concentration, rate: tc.params.rate })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()
        for (let i = 0; i < tc.points.length; i++) {
          if (tc.expected.log_prob[i] !== null) {
            expectClose(data[i], tc.expected.log_prob[i], { rtol: 1e-2, atol: 1e-2 })
          }
        }
        lp.dispose()
        points.dispose()
        d.dispose()
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Gamma({ concentration: tc.params.concentration, rate: tc.params.rate })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-4 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
