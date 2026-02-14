import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Multinomial } from '../../../src/distributions/multinomial'
import { expectClose, expectTensorClose } from '../../helpers/tolerance'

describe('Multinomial distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      expect(d.totalCount.dataSync()[0]).toBe(10)
      expect(d.numCategories).toBe(4)
      d.dispose()
    })

    test('logits param', () => {
      const d = new Multinomial({ totalCount: 10, logits: [0, 0, 0, 0] })
      const p = d.probs.dataSync()
      expectClose(p[0], 0.25, { atol: 1e-5 })
      d.dispose()
    })

    test('throws without probs or logits', () => {
      expect(() => new Multinomial({ totalCount: 10 })).toThrow()
    })
  })

  describe('logProb', () => {
    test('uniform multinomial', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      const lp = d.logProb(tf.tensor([3, 3, 2, 2]))
      // Compare to known value
      expect(lp.shape).toEqual([])
      lp.dispose()
      d.dispose()
    })

    test('all in one category', () => {
      const d = new Multinomial({ totalCount: 5, probs: [0.5, 0.3, 0.2] })
      const lp = d.logProb(tf.tensor([5, 0, 0]))
      // P = 0.5^5 = 0.03125
      expectClose(lp.dataSync()[0], Math.log(0.03125), { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean = n*p', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      const m = d.mean()
      expect(m.shape).toEqual([4])
      const data = m.dataSync()
      for (let i = 0; i < 4; i++) {
        expectClose(data[i], 2.5, { atol: 1e-4 })
      }
      m.dispose()
      d.dispose()
    })

    test('variance = n*p*(1-p)', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      const v = d.variance()
      expect(v.shape).toEqual([4])
      const data = v.dataSync()
      for (let i = 0; i < 4; i++) {
        expectClose(data[i], 1.875, { atol: 1e-3 })
      }
      v.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('shape is correct', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      const s = d.sample([100])
      expect(s.shape).toEqual([100, 4])
      s.dispose()
      d.dispose()
    })

    test('samples sum to totalCount', () => {
      const d = new Multinomial({ totalCount: 10, probs: [0.25, 0.25, 0.25, 0.25] })
      const s = d.sample([50])
      const sums = s.sum(-1)
      const data = sums.dataSync()
      for (let i = 0; i < data.length; i++) {
        expectClose(data[i], 10, { atol: 1e-5 })
      }
      sums.dispose()
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/multinomial.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Multinomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        for (let i = 0; i < tc.points.length; i++) {
          const lp = d.logProb(tf.tensor(tc.points[i]))
          expectClose(lp.dataSync()[0], tc.expected.log_prob[i], { rtol: 1e-2, atol: 1e-2 })
          lp.dispose()
        }
        d.dispose()
      }
    })

    test('mean matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Multinomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        const m = d.mean()
        const data = m.dataSync()
        for (let i = 0; i < tc.expected.mean.length; i++) {
          expectClose(data[i], tc.expected.mean[i], { rtol: 1e-3, atol: 1e-3 })
        }
        m.dispose()
        d.dispose()
      }
    })

    test('variance matches scipy', () => {
      if (!refData) return
      for (const tc of refData.test_cases) {
        const d = new Multinomial({ totalCount: tc.params.totalCount, probs: tc.params.probs })
        const v = d.variance()
        const data = v.dataSync()
        for (let i = 0; i < tc.expected.variance.length; i++) {
          expectClose(data[i], tc.expected.variance[i], { rtol: 1e-2, atol: 1e-2 })
        }
        v.dispose()
        d.dispose()
      }
    })
  })
})
