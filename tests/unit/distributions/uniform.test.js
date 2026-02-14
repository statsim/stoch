import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Uniform } from '../../../src/distributions/uniform'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Uniform distribution', () => {
  test('default params [0,1]', () => {
    const d = new Uniform()
    expectClose(d.mean().dataSync()[0], 0.5, { atol: 1e-6 })
    d.dispose()
  })

  test('custom params', () => {
    const d = new Uniform({ low: 2, high: 8 })
    expectClose(d.mean().dataSync()[0], 5, { atol: 1e-5 })
    expectClose(d.variance().dataSync()[0], 3, { atol: 1e-4 })
    d.dispose()
  })

  test('throws for low >= high', () => {
    expect(() => new Uniform({ low: 5, high: 5 })).toThrow()
    expect(() => new Uniform({ low: 5, high: 3 })).toThrow()
  })

  test('logProb inside support', () => {
    const d = new Uniform({ low: 0, high: 1 })
    const lp = d.logProb(0.5)
    expectClose(lp.dataSync()[0], 0, { atol: 1e-5 }) // log(1/(1-0)) = 0
    lp.dispose()
    d.dispose()
  })

  test('logProb outside support is -Infinity', () => {
    const d = new Uniform({ low: 0, high: 1 })
    const lp = d.logProb(-1)
    expect(lp.dataSync()[0]).toBe(-Infinity)
    lp.dispose()
    d.dispose()
  })

  test('CDF', () => {
    const d = new Uniform({ low: 0, high: 10 })
    expectClose(d.cdf(5).dataSync()[0], 0.5, { atol: 1e-5 })
    expectClose(d.cdf(0).dataSync()[0], 0.0, { atol: 1e-5 })
    expectClose(d.cdf(10).dataSync()[0], 1.0, { atol: 1e-5 })
    d.dispose()
  })

  test('entropy', () => {
    const d = new Uniform({ low: 0, high: 10 })
    expectClose(d.entropy().dataSync()[0], Math.log(10), { rtol: 1e-4 })
    d.dispose()
  })

  test('sample in range', () => {
    const d = new Uniform({ low: 2, high: 5 })
    const s = d.sample([10000])
    const data = s.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThanOrEqual(2)
      expect(data[i]).toBeLessThanOrEqual(5)
    }
    s.dispose()
    d.dispose()
  })

  test('sample mean', () => {
    const d = new Uniform({ low: 0, high: 10 })
    const s = d.sample([100000])
    const stats = sampleStats(s.dataSync())
    const tol = autoTolerance('mean', 100000, 100 / 12)
    expectClose(stats.mean, 5, { atol: tol })
    s.dispose()
    d.dispose()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/uniform.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('mean/variance match scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Uniform({ low: tc.params.low, high: tc.params.high })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-4, atol: 1e-5 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-3, atol: 1e-5 })
        d.dispose()
      }
    })
  })
})
