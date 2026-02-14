import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Exponential } from '../../../src/distributions/exponential'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Exponential distribution', () => {
  test('mean = 1/rate', () => {
    const d = new Exponential({ rate: 2 })
    expectClose(d.mean().dataSync()[0], 0.5, { atol: 1e-5 })
    d.dispose()
  })

  test('variance = 1/rate^2', () => {
    const d = new Exponential({ rate: 2 })
    expectClose(d.variance().dataSync()[0], 0.25, { atol: 1e-4 })
    d.dispose()
  })

  test('logProb at 0 = log(rate)', () => {
    const d = new Exponential({ rate: 2 })
    expectClose(d.logProb(0).dataSync()[0], Math.log(2), { atol: 1e-4 })
    d.dispose()
  })

  test('CDF(1) for rate=1', () => {
    const d = new Exponential({ rate: 1 })
    expectClose(d.cdf(1).dataSync()[0], 1 - Math.exp(-1), { atol: 1e-4 })
    d.dispose()
  })

  test('entropy = 1 - log(rate)', () => {
    const d = new Exponential({ rate: 2 })
    expectClose(d.entropy().dataSync()[0], 1 - Math.log(2), { atol: 1e-4 })
    d.dispose()
  })

  test('samples are non-negative', () => {
    const d = new Exponential({ rate: 1 })
    const s = d.sample([10000])
    const data = s.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThanOrEqual(0)
    }
    s.dispose()
    d.dispose()
  })

  test('sample mean converges', () => {
    const d = new Exponential({ rate: 2 })
    const s = d.sample([100000])
    const stats = sampleStats(s.dataSync())
    const tol = autoTolerance('mean', 100000, 0.25)
    expectClose(stats.mean, 0.5, { atol: tol })
    s.dispose()
    d.dispose()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/exponential.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Exponential({ rate: tc.params.rate })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()
        for (let i = 0; i < tc.points.length; i++) {
          if (tc.expected.log_prob[i] !== null) {
            expectClose(data[i], tc.expected.log_prob[i], { rtol: 1e-3, atol: 1e-2 })
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
        const d = new Exponential({ rate: tc.params.rate })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-4 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
