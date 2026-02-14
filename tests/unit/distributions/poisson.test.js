import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Poisson } from '../../../src/distributions/poisson'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Poisson distribution', () => {
  test('mean = rate', () => {
    const d = new Poisson({ rate: 5 })
    expectClose(d.mean().dataSync()[0], 5, { atol: 1e-5 })
    d.dispose()
  })

  test('variance = rate', () => {
    const d = new Poisson({ rate: 5 })
    expectClose(d.variance().dataSync()[0], 5, { atol: 1e-4 })
    d.dispose()
  })

  test('logProb P(0) for rate=1 = -1', () => {
    const d = new Poisson({ rate: 1 })
    // P(0) = exp(-1), logP(0) = -1
    expectClose(d.logProb(0).dataSync()[0], -1, { atol: 1e-4 })
    d.dispose()
  })

  test('logProb P(1) for rate=1', () => {
    const d = new Poisson({ rate: 1 })
    // P(1) = 1 * exp(-1) / 1! = exp(-1), logP(1) = -1
    expectClose(d.logProb(1).dataSync()[0], -1, { atol: 1e-4 })
    d.dispose()
  })

  test('mode = floor(rate)', () => {
    const d = new Poisson({ rate: 5.7 })
    expectClose(d.mode().dataSync()[0], 5, { atol: 1e-4 })
    d.dispose()
  })

  test('samples are non-negative integers', () => {
    const d = new Poisson({ rate: 3 })
    const s = d.sample([5000])
    const data = s.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThanOrEqual(0)
      expect(data[i] % 1).toBeCloseTo(0, 10) // integer
    }
    s.dispose()
    d.dispose()
  })

  test('sample mean converges', () => {
    const d = new Poisson({ rate: 5 })
    const s = d.sample([50000])
    const stats = sampleStats(s.dataSync())
    const tol = autoTolerance('mean', 50000, 5)
    expectClose(stats.mean, 5, { atol: tol })
    s.dispose()
    d.dispose()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/poisson.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Poisson({ rate: tc.params.rate })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()
        for (let i = 0; i < tc.points.length; i++) {
          if (tc.expected.log_prob[i] !== null) {
            expectClose(data[i], tc.expected.log_prob[i], { rtol: 1e-2, atol: 5e-2 })
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
        const d = new Poisson({ rate: tc.params.rate })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-4 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-3, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
