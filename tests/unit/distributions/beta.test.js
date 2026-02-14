import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Beta } from '../../../src/distributions/beta'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Beta distribution', () => {
  test('mean for Beta(2, 5)', () => {
    const d = new Beta({ concentration1: 2, concentration0: 5 })
    expectClose(d.mean().dataSync()[0], 2 / 7, { rtol: 1e-4 })
    d.dispose()
  })

  test('variance for Beta(2, 5)', () => {
    const d = new Beta({ concentration1: 2, concentration0: 5 })
    const expected = (2 * 5) / (49 * 8) // a*b / ((a+b)^2 * (a+b+1))
    expectClose(d.variance().dataSync()[0], expected, { rtol: 1e-3 })
    d.dispose()
  })

  test('Beta(1,1) is Uniform', () => {
    const d = new Beta({ concentration1: 1, concentration0: 1 })
    const lp = d.logProb(0.5)
    expectClose(lp.dataSync()[0], 0, { atol: 1e-4 }) // log(1) = 0
    lp.dispose()
    d.dispose()
  })

  test('throws for non-positive concentrations', () => {
    expect(() => new Beta({ concentration1: 0, concentration0: 1 })).toThrow()
    expect(() => new Beta({ concentration1: 1, concentration0: -1 })).toThrow()
  })

  test('samples in [0, 1]', () => {
    const d = new Beta({ concentration1: 2, concentration0: 5 })
    const s = d.sample([10000])
    const data = s.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThanOrEqual(0)
      expect(data[i]).toBeLessThanOrEqual(1)
    }
    s.dispose()
    d.dispose()
  })

  test('sample mean converges', () => {
    const d = new Beta({ concentration1: 2, concentration0: 5 })
    const s = d.sample([100000])
    const stats = sampleStats(s.dataSync())
    const expectedMean = 2 / 7
    const expectedVar = (2 * 5) / (49 * 8)
    const tol = autoTolerance('mean', 100000, expectedVar)
    expectClose(stats.mean, expectedMean, { atol: tol })
    s.dispose()
    d.dispose()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/beta.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Beta({
          concentration1: tc.params.concentration1,
          concentration0: tc.params.concentration0
        })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()
        for (let i = 0; i < tc.points.length; i++) {
          if (tc.expected.log_prob[i] !== null && isFinite(tc.expected.log_prob[i])) {
            expectClose(data[i], tc.expected.log_prob[i], { rtol: 5e-2, atol: 1e-1 })
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
        const d = new Beta({
          concentration1: tc.params.concentration1,
          concentration0: tc.params.concentration0
        })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-3, atol: 1e-4 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-2, atol: 1e-4 })
        d.dispose()
      }
    })
  })
})
