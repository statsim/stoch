import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Categorical } from '../../../src/distributions/categorical'
import { expectClose } from '../../helpers/tolerance'

describe('Categorical distribution', () => {
  test('uniform probs', () => {
    const d = new Categorical({ probs: [0.25, 0.25, 0.25, 0.25] })
    expect(d.numCategories).toBe(4)
    d.dispose()
  })

  test('mean for uniform', () => {
    const d = new Categorical({ probs: [0.25, 0.25, 0.25, 0.25] })
    // E[X] = 0*0.25 + 1*0.25 + 2*0.25 + 3*0.25 = 1.5
    expectClose(d.mean().dataSync()[0], 1.5, { atol: 1e-4 })
    d.dispose()
  })

  test('entropy for uniform over 4 categories', () => {
    const d = new Categorical({ probs: [0.25, 0.25, 0.25, 0.25] })
    expectClose(d.entropy().dataSync()[0], Math.log(4), { rtol: 1e-4 })
    d.dispose()
  })

  test('logProb', () => {
    const d = new Categorical({ probs: [0.1, 0.2, 0.3, 0.4] })
    const lp0 = d.logProb(0)
    const lp1 = d.logProb(1)
    expectClose(lp0.dataSync()[0], Math.log(0.1), { rtol: 1e-3 })
    expectClose(lp1.dataSync()[0], Math.log(0.2), { rtol: 1e-3 })
    lp0.dispose()
    lp1.dispose()
    d.dispose()
  })

  test('mode', () => {
    const d = new Categorical({ probs: [0.1, 0.2, 0.3, 0.4] })
    expectClose(d.mode().dataSync()[0], 3, { atol: 1e-5 })
    d.dispose()
  })

  test('logits param', () => {
    // logits [0, 0, 0, 0] → probs [0.25, 0.25, 0.25, 0.25]
    const d = new Categorical({ logits: [0, 0, 0, 0] })
    const p = d.probs
    const data = p.dataSync()
    for (let i = 0; i < 4; i++) {
      expectClose(data[i], 0.25, { atol: 1e-5 })
    }
    p.dispose()
    d.dispose()
  })

  test('sample returns valid categories', () => {
    const d = new Categorical({ probs: [0.1, 0.2, 0.3, 0.4] })
    const s = d.sample([1000])
    const data = s.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThanOrEqual(0)
      expect(data[i]).toBeLessThan(4)
      expect(data[i] % 1).toBeCloseTo(0, 10) // integer
    }
    s.dispose()
    d.dispose()
  })

  test('throws when both probs and logits given', () => {
    expect(() => new Categorical({ probs: [0.5, 0.5], logits: [0, 0] })).toThrow()
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/categorical.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches reference', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Categorical({ probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const lp = d.logProb(tc.points[i])
          expectClose(lp.dataSync()[0], tc.expected.log_prob[i], { rtol: 1e-3, atol: 1e-3 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('entropy matches reference', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Categorical({ probs: tc.params.probs })
        expectClose(d.entropy().dataSync()[0], tc.expected.entropy, { rtol: 1e-3, atol: 1e-3 })
        d.dispose()
      }
    })
  })
})
