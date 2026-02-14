import * as tf from '@tensorflow/tfjs'
import { RelaxedBernoulli } from '../../../src/distributions/relaxed_bernoulli'
import { expectClose } from '../../helpers/tolerance'

describe('RelaxedBernoulli distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new RelaxedBernoulli({ temperature: 0.5, probs: 0.7 })
      expect(d.temperature.dataSync()[0]).toBe(0.5)
      d.dispose()
    })
  })

  describe('sample', () => {
    test('samples in (0, 1)', () => {
      const d = new RelaxedBernoulli({ temperature: 0.5, probs: 0.5 })
      const s = d.sample([100])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
        expect(data[i]).toBeLessThan(1)
      }
      s.dispose()
      d.dispose()
    })

    test('low temperature concentrates near 0 or 1', () => {
      const d = new RelaxedBernoulli({ temperature: 0.01, probs: 0.5 })
      const s = d.sample([100])
      const data = s.dataSync()
      let nearEdge = 0
      for (let i = 0; i < data.length; i++) {
        if (data[i] < 0.1 || data[i] > 0.9) nearEdge++
      }
      // Most samples should be near 0 or 1 with low temperature
      expect(nearEdge).toBeGreaterThan(80)
      s.dispose()
      d.dispose()
    })

    test('sample mean near probs', () => {
      const d = new RelaxedBernoulli({ temperature: 1, probs: 0.7 })
      const s = d.sample([5000])
      const mean = tf.mean(s).dataSync()[0]
      expectClose(mean, 0.7, { atol: 0.1 })
      s.dispose()
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb is finite for valid inputs', () => {
      const d = new RelaxedBernoulli({ temperature: 1, probs: 0.5 })
      const lp = d.logProb(tf.tensor([0.3, 0.5, 0.7]))
      const data = lp.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(isFinite(data[i])).toBe(true)
      }
      lp.dispose()
      d.dispose()
    })

    test('logProb peaks near probs with low temperature', () => {
      const d = new RelaxedBernoulli({ temperature: 0.1, probs: 0.8 })
      const lpNear = d.logProb(0.8).dataSync()[0]
      const lpFar = d.logProb(0.2).dataSync()[0]
      expect(lpNear).toBeGreaterThan(lpFar)
      d.dispose()
    })
  })
})
