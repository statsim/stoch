import * as tf from '@tensorflow/tfjs'
import { RelaxedOneHotCategorical } from '../../../src/distributions/relaxed_one_hot_categorical'
import { expectClose } from '../../helpers/tolerance'

describe('RelaxedOneHotCategorical distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new RelaxedOneHotCategorical({ temperature: 0.5, probs: [0.2, 0.3, 0.5] })
      expect(d.temperature.dataSync()[0]).toBe(0.5)
      expect(d.numCategories).toBe(3)
      expect(d.eventShape).toEqual([3])
      d.dispose()
    })
  })

  describe('sample', () => {
    test('samples are on the simplex', () => {
      const d = new RelaxedOneHotCategorical({ temperature: 0.5, probs: [0.2, 0.3, 0.5] })
      const s = d.sample([10])
      expect(s.shape).toEqual([10, 3])
      const data = s.dataSync()
      for (let i = 0; i < 10; i++) {
        const sum = data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2]
        expectClose(sum, 1, { atol: 1e-4 })
        for (let j = 0; j < 3; j++) {
          expect(data[i * 3 + j]).toBeGreaterThan(0)
        }
      }
      s.dispose()
      d.dispose()
    })

    test('low temperature concentrates near one-hot', () => {
      const d = new RelaxedOneHotCategorical({ temperature: 0.01, probs: [0.1, 0.2, 0.7] })
      const s = d.sample([100])
      const data = s.dataSync()
      let nearOneHot = 0
      for (let i = 0; i < 100; i++) {
        const max = Math.max(data[i * 3], data[i * 3 + 1], data[i * 3 + 2])
        if (max > 0.8) nearOneHot++
      }
      // Most samples should be concentrated near one-hot
      expect(nearOneHot).toBeGreaterThan(50)
      s.dispose()
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb is finite for valid simplex input', () => {
      const d = new RelaxedOneHotCategorical({ temperature: 1, probs: [0.3, 0.3, 0.4] })
      const y = tf.tensor([0.3, 0.3, 0.4])
      const lp = d.logProb(y)
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose(); y.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean = probs', () => {
      const d = new RelaxedOneHotCategorical({ temperature: 0.5, probs: [0.2, 0.3, 0.5] })
      const m = d.mean()
      expect(m.shape).toEqual([3])
      expectClose(m.dataSync()[0], 0.2, { atol: 1e-5 })
      expectClose(m.dataSync()[1], 0.3, { atol: 1e-5 })
      expectClose(m.dataSync()[2], 0.5, { atol: 1e-5 })
      m.dispose()
      d.dispose()
    })
  })
})
