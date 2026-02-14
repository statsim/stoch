import * as tf from '@tensorflow/tfjs'
import { OneHotCategorical } from '../../../src/distributions/one_hot_categorical'
import { expectClose } from '../../helpers/tolerance'

describe('OneHotCategorical distribution', () => {
  describe('constructor', () => {
    test('basic params with probs', () => {
      const d = new OneHotCategorical({ probs: [0.2, 0.3, 0.5] })
      expect(d.numCategories).toBe(3)
      expect(d.eventShape).toEqual([3])
      d.dispose()
    })

    test('basic params with logits', () => {
      const d = new OneHotCategorical({ logits: [0, 0, 0] })
      expect(d.numCategories).toBe(3)
      d.dispose()
    })
  })

  describe('sample', () => {
    test('samples are one-hot', () => {
      const d = new OneHotCategorical({ probs: [0.2, 0.3, 0.5] })
      const s = d.sample([10])
      expect(s.shape).toEqual([10, 3])
      const data = s.dataSync()
      for (let i = 0; i < 10; i++) {
        const row = [data[i * 3], data[i * 3 + 1], data[i * 3 + 2]]
        // Exactly one 1, rest are 0
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
        expect(row.filter(x => x === 1).length).toBe(1)
      }
      s.dispose()
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb of one-hot vectors', () => {
      const d = new OneHotCategorical({ probs: [0.2, 0.3, 0.5] })
      const lp0 = d.logProb(tf.tensor([1, 0, 0]))
      const lp1 = d.logProb(tf.tensor([0, 1, 0]))
      const lp2 = d.logProb(tf.tensor([0, 0, 1]))
      expectClose(lp0.dataSync()[0], Math.log(0.2), { atol: 1e-4 })
      expectClose(lp1.dataSync()[0], Math.log(0.3), { atol: 1e-4 })
      expectClose(lp2.dataSync()[0], Math.log(0.5), { atol: 1e-4 })
      lp0.dispose(); lp1.dispose(); lp2.dispose()
      d.dispose()
    })
  })

  describe('mean/entropy/mode', () => {
    test('mean = probs', () => {
      const d = new OneHotCategorical({ probs: [0.2, 0.3, 0.5] })
      const m = d.mean()
      expect(m.shape).toEqual([3])
      expectClose(m.dataSync()[0], 0.2, { atol: 1e-5 })
      expectClose(m.dataSync()[1], 0.3, { atol: 1e-5 })
      expectClose(m.dataSync()[2], 0.5, { atol: 1e-5 })
      m.dispose()
      d.dispose()
    })

    test('mode is one-hot at max prob', () => {
      const d = new OneHotCategorical({ probs: [0.1, 0.6, 0.3] })
      const m = d.mode()
      expect(m.dataSync()[0]).toBe(0)
      expect(m.dataSync()[1]).toBe(1)
      expect(m.dataSync()[2]).toBe(0)
      m.dispose()
      d.dispose()
    })

    test('entropy of uniform', () => {
      const d = new OneHotCategorical({ probs: [0.25, 0.25, 0.25, 0.25] })
      const h = d.entropy()
      expectClose(h.dataSync()[0], Math.log(4), { atol: 1e-4 })
      h.dispose()
      d.dispose()
    })
  })
})
