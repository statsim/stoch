import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Bernoulli } from '../../../src/distributions/bernoulli'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('Bernoulli distribution', () => {
  describe('constructor', () => {
    test('probs param', () => {
      const d = new Bernoulli({ probs: 0.5 })
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-6 })
      d.dispose()
    })

    test('logits param', () => {
      const d = new Bernoulli({ logits: 0 })
      // logit(0.5) = 0
      expectClose(d.probs.dataSync()[0], 0.5, { atol: 1e-5 })
      d.dispose()
    })

    test('throws when both probs and logits given', () => {
      expect(() => new Bernoulli({ probs: 0.5, logits: 0 })).toThrow('Exactly one')
    })

    test('throws when neither given', () => {
      expect(() => new Bernoulli({})).toThrow('Exactly one')
    })
  })

  describe('logProb', () => {
    test('logP(1) for p=0.5', () => {
      const d = new Bernoulli({ probs: 0.5 })
      const lp = d.logProb(1)
      expectClose(lp.dataSync()[0], Math.log(0.5), { rtol: 1e-4 })
      lp.dispose()
      d.dispose()
    })

    test('logP(0) for p=0.5', () => {
      const d = new Bernoulli({ probs: 0.5 })
      const lp = d.logProb(0)
      expectClose(lp.dataSync()[0], Math.log(0.5), { rtol: 1e-4 })
      lp.dispose()
      d.dispose()
    })

    test('logP(1) for p=0.9', () => {
      const d = new Bernoulli({ probs: 0.9 })
      const lp = d.logProb(1)
      expectClose(lp.dataSync()[0], Math.log(0.9), { rtol: 1e-3 })
      lp.dispose()
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean equals probs', () => {
      const d = new Bernoulli({ probs: 0.3 })
      expectClose(d.mean().dataSync()[0], 0.3, { atol: 1e-5 })
      d.dispose()
    })

    test('variance = p(1-p)', () => {
      const d = new Bernoulli({ probs: 0.3 })
      expectClose(d.variance().dataSync()[0], 0.3 * 0.7, { atol: 1e-5 })
      d.dispose()
    })
  })

  describe('sample', () => {
    test('samples are 0 or 1', () => {
      const d = new Bernoulli({ probs: 0.5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i] === 0 || data[i] === 1).toBe(true)
      }
      s.dispose()
      d.dispose()
    })

    test('sample mean close to probs', () => {
      const d = new Bernoulli({ probs: 0.7 })
      const s = d.sample([50000])
      const stats = sampleStats(s.dataSync())
      const tol = autoTolerance('mean', 50000, 0.7 * 0.3)
      expectClose(stats.mean, 0.7, { atol: tol })
      s.dispose()
      d.dispose()
    })
  })

  describe('reference data', () => {
    let refData
    const refPath = path.join(__dirname, '../../reference-data/bernoulli.json')

    beforeAll(() => {
      if (fs.existsSync(refPath)) {
        refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
      }
    })

    test('logProb matches scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Bernoulli({ probs: tc.params.probs })
        const points = tf.tensor(tc.points)
        const lp = d.logProb(points)
        const data = lp.dataSync()

        for (let i = 0; i < tc.points.length; i++) {
          expectClose(data[i], tc.expected.log_prob[i], { rtol: 1e-3, atol: 1e-3 })
        }

        lp.dispose()
        points.dispose()
        d.dispose()
      }
    })

    test('mean/variance match scipy reference', () => {
      if (!refData) return

      for (const tc of refData.test_cases) {
        const d = new Bernoulli({ probs: tc.params.probs })
        expectClose(d.mean().dataSync()[0], tc.expected.mean, { rtol: 1e-4, atol: 1e-5 })
        expectClose(d.variance().dataSync()[0], tc.expected.variance, { rtol: 1e-3, atol: 1e-5 })
        d.dispose()
      }
    })
  })
})
