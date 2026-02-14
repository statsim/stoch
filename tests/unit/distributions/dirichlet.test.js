import * as tf from '@tensorflow/tfjs'
import { Dirichlet } from '../../../src/distributions/dirichlet'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('Dirichlet distribution', () => {
  describe('constructor', () => {
    test('basic construction', () => {
      const d = new Dirichlet({ concentration: [1, 1, 1] })
      expect(d.name).toBe('Dirichlet')
      expect(d.eventShape).toEqual([3])
      expect(d.batchShape).toEqual([])
      d.dispose()
    })

    test('2D Dirichlet', () => {
      const d = new Dirichlet({ concentration: [2, 3] })
      expect(d.eventShape).toEqual([2])
      d.dispose()
    })

    test('throws for non-positive concentration', () => {
      expect(() => new Dirichlet({ concentration: [1, 0, 1] })).toThrow('must be positive')
      expect(() => new Dirichlet({ concentration: [1, -1, 1] })).toThrow('must be positive')
    })
  })

  describe('logProb', () => {
    test('uniform Dirichlet (all α=1) is constant', () => {
      // Dir(1,1,1): logpdf = -logB(1,1,1) = -log(Γ(1)³/Γ(3)) = log(2) = 0.693
      const d = new Dirichlet({ concentration: [1, 1, 1] })
      const lp1 = d.logProb(tf.tensor([1/3, 1/3, 1/3]))
      const lp2 = d.logProb(tf.tensor([0.2, 0.3, 0.5]))

      // Should be constant for all valid points
      expectClose(lp1.dataSync()[0], lp2.dataSync()[0], { atol: 1e-4 })
      // Value should be log(2!)
      expectClose(lp1.dataSync()[0], Math.log(2), { atol: 1e-3 })

      lp1.dispose()
      lp2.dispose()
      d.dispose()
    })

    test('logProb for Dir(2, 3)', () => {
      const d = new Dirichlet({ concentration: [2, 3] })
      const x = [0.3, 0.7]
      const lp = d.logProb(tf.tensor(x))

      // logpdf = (2-1)*log(0.3) + (3-1)*log(0.7) - logB(2,3)
      // logB(2,3) = logΓ(2)+logΓ(3)-logΓ(5) = 0+log(2)-log(24) = log(2)-log(24)
      const logBeta = Math.log(2) - Math.log(24) // logΓ(2) + logΓ(3) - logΓ(5) = 0 + ln2 - ln24
      const expected = Math.log(0.3) + 2 * Math.log(0.7) - logBeta
      expectClose(lp.dataSync()[0], expected, { atol: 1e-3 })

      lp.dispose()
      d.dispose()
    })

    test('logProb output is scalar', () => {
      const d = new Dirichlet({ concentration: [1, 1, 1] })
      const lp = d.logProb(tf.tensor([0.2, 0.3, 0.5]))
      expect(lp.shape).toEqual([])
      lp.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('sample shape', () => {
      const d = new Dirichlet({ concentration: [1, 1, 1] })
      const s = d.sample([100])
      expect(s.shape).toEqual([100, 3])
      s.dispose()
      d.dispose()
    })

    test('samples sum to 1', () => {
      const d = new Dirichlet({ concentration: [2, 3, 5] })
      const s = d.sample([100])
      const sums = tf.sum(s, -1)
      const data = sums.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeCloseTo(1, 3)
      }
      s.dispose()
      sums.dispose()
      d.dispose()
    })

    test('samples are positive', () => {
      const d = new Dirichlet({ concentration: [2, 3, 5] })
      const s = d.sample([100])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
      }
      s.dispose()
      d.dispose()
    })

    test('sample means match theoretical', () => {
      const alpha = [2, 3, 5]
      const alpha0 = 10
      const d = new Dirichlet({ concentration: alpha })
      const s = d.sample([20000])
      const data = s.dataSync()

      for (let k = 0; k < 3; k++) {
        const kData = []
        for (let i = 0; i < 20000; i++) {
          kData.push(data[i * 3 + k])
        }
        const stats = sampleStats(kData)
        expectClose(stats.mean, alpha[k] / alpha0, { atol: 0.02 })
      }

      s.dispose()
      d.dispose()
    })

    test('sample with small concentration (alpha < 1)', () => {
      // Test Ahrens-Dieter trick for alpha < 1
      const d = new Dirichlet({ concentration: [0.1, 0.2, 0.7] })
      const s = d.sample([100])
      const sums = tf.sum(s, -1)
      const data = sums.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeCloseTo(1, 2)
      }
      s.dispose()
      sums.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean = concentration / sum(concentration)', () => {
      const d = new Dirichlet({ concentration: [2, 3, 5] })
      const m = d.mean()
      expect(m.shape).toEqual([3])
      const data = m.dataSync()
      expect(data[0]).toBeCloseTo(0.2, 4)
      expect(data[1]).toBeCloseTo(0.3, 4)
      expect(data[2]).toBeCloseTo(0.5, 4)
      m.dispose()
      d.dispose()
    })
  })

  describe('variance', () => {
    test('variance formula', () => {
      const alpha = [2, 3, 5]
      const alpha0 = 10
      const d = new Dirichlet({ concentration: alpha })
      const v = d.variance()
      const data = v.dataSync()
      for (let k = 0; k < 3; k++) {
        const expected = alpha[k] * (alpha0 - alpha[k]) / (alpha0 * alpha0 * (alpha0 + 1))
        expectClose(data[k], expected, { atol: 1e-4 })
      }
      v.dispose()
      d.dispose()
    })
  })

  describe('mode', () => {
    test('mode for α_k > 1', () => {
      const d = new Dirichlet({ concentration: [3, 4, 5] })
      const mode = d.mode()
      const data = mode.dataSync()
      // mode_k = (α_k - 1) / (α₀ - K) = (α_k - 1) / (12 - 3) = (α_k - 1) / 9
      expect(data[0]).toBeCloseTo(2 / 9, 4)
      expect(data[1]).toBeCloseTo(3 / 9, 4)
      expect(data[2]).toBeCloseTo(4 / 9, 4)
      mode.dispose()
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy of uniform Dirichlet', () => {
      const d = new Dirichlet({ concentration: [1, 1, 1] })
      const h = d.entropy()
      // For Dir(1,1,1): H = logB(1,1,1) + 0 + 0 = -log(2)
      expectClose(h.dataSync()[0], -Math.log(2), { atol: 1e-3 })
      h.dispose()
      d.dispose()
    })
  })

  test('dispose frees memory', () => {
    const before = tf.memory().numTensors
    const d = new Dirichlet({ concentration: [2, 3, 5] })
    expect(tf.memory().numTensors).toBeGreaterThan(before)
    d.dispose()
    expect(tf.memory().numTensors).toBe(before)
  })
})
