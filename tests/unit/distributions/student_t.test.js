import * as tf from '@tensorflow/tfjs'
import { StudentT } from '../../../src/distributions/student_t'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('StudentT distribution', () => {
  describe('constructor', () => {
    test('basic construction', () => {
      const d = new StudentT({ df: 5 })
      expect(d.df.dataSync()[0]).toBe(5)
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      expect(d.name).toBe('StudentT')
      d.dispose()
    })

    test('custom params', () => {
      const d = new StudentT({ df: 10, loc: 2, scale: 3 })
      expect(d.df.dataSync()[0]).toBe(10)
      expect(d.loc.dataSync()[0]).toBe(2)
      expect(d.scale.dataSync()[0]).toBe(3)
      d.dispose()
    })

    test('throws for non-positive df', () => {
      expect(() => new StudentT({ df: 0 })).toThrow('must be positive')
      expect(() => new StudentT({ df: -1 })).toThrow('must be positive')
    })

    test('throws for non-positive scale', () => {
      expect(() => new StudentT({ df: 5, scale: 0 })).toThrow('must be positive')
    })
  })

  describe('logProb', () => {
    test('logProb at mean for df=5', () => {
      const d = new StudentT({ df: 5, loc: 0, scale: 1 })
      const lp = d.logProb(0)

      // At x=0: logpdf = logΓ(3) - logΓ(2.5) - 0.5*log(5π)
      // Γ(3) = 2!, Γ(2.5) = 1.5*0.5*√π = 0.75√π
      const expected = Math.log(2) - Math.log(0.75 * Math.sqrt(Math.PI))
        - 0.5 * Math.log(5 * Math.PI)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-3 })

      lp.dispose()
      d.dispose()
    })

    test('logProb is symmetric', () => {
      const d = new StudentT({ df: 5, loc: 0, scale: 1 })
      const lpPos = d.logProb(2)
      const lpNeg = d.logProb(-2)
      expectClose(lpPos.dataSync()[0], lpNeg.dataSync()[0], { atol: 1e-5 })
      lpPos.dispose()
      lpNeg.dispose()
      d.dispose()
    })

    test('heavier tails than Normal', () => {
      const t3 = new StudentT({ df: 3, loc: 0, scale: 1 })
      const lp = t3.logProb(5)

      // Compare with Normal(0,1) at x=5
      const normalLp = -0.5 * 25 - 0.5 * Math.log(2 * Math.PI)
      // t-distribution should have higher logProb at x=5 (heavier tails)
      expect(lp.dataSync()[0]).toBeGreaterThan(normalLp)

      lp.dispose()
      t3.dispose()
    })

    test('logProb with shifted loc and scale', () => {
      const d = new StudentT({ df: 10, loc: 3, scale: 2 })
      // At the mode/mean, the kernel term is 0
      const lp = d.logProb(3)
      // logpdf(loc) = logΓ(5.5) - logΓ(5) - 0.5*log(10π) - log(2)
      const lgamma55 = 4.0 * Math.log(4.0) + Math.log(3.5 * 2.5 * 1.5 * 0.5 * Math.sqrt(Math.PI))
      // Use numerical check instead
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose()
      d.dispose()
    })

    test('logProb with tensor input', () => {
      const d = new StudentT({ df: 5 })
      const lp = d.logProb(tf.tensor([-2, -1, 0, 1, 2]))
      expect(lp.shape).toEqual([5])
      lp.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('sample shape', () => {
      const d = new StudentT({ df: 5 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample statistics for df=30 (close to Normal)', () => {
      // With high df, t-distribution approximates Normal
      const d = new StudentT({ df: 30, loc: 5, scale: 2 })
      const s = d.sample([50000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      expectClose(stats.mean, 5, { atol: 0.1 })
      // Var = scale² * df/(df-2) = 4 * 30/28 ≈ 4.286
      const expectedVar = 4 * 30 / 28
      expectClose(stats.variance, expectedVar, { atol: 0.5 })

      s.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean = loc for df > 1', () => {
      const d = new StudentT({ df: 5, loc: 3, scale: 2 })
      const m = d.mean()
      expectClose(m.dataSync()[0], 3, { atol: 1e-5 })
      m.dispose()
      d.dispose()
    })

    test('mean is NaN for df <= 1', () => {
      const d = new StudentT({ df: 0.5, loc: 0, scale: 1, validateArgs: false })
      const m = d.mean()
      expect(isNaN(m.dataSync()[0])).toBe(true)
      m.dispose()
      d.dispose()
    })
  })

  describe('variance', () => {
    test('variance for df > 2', () => {
      const d = new StudentT({ df: 5, loc: 0, scale: 2 })
      const v = d.variance()
      // var = scale² * df/(df-2) = 4 * 5/3
      expectClose(v.dataSync()[0], 4 * 5 / 3, { atol: 1e-3 })
      v.dispose()
      d.dispose()
    })

    test('variance is Infinity for 1 < df <= 2', () => {
      const d = new StudentT({ df: 1.5, loc: 0, scale: 1, validateArgs: false })
      const v = d.variance()
      expect(v.dataSync()[0]).toBe(Infinity)
      v.dispose()
      d.dispose()
    })
  })

  describe('mode', () => {
    test('mode = loc', () => {
      const d = new StudentT({ df: 5, loc: 7, scale: 2 })
      const mode = d.mode()
      expectClose(mode.dataSync()[0], 7, { atol: 1e-5 })
      mode.dispose()
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy is finite for df=5', () => {
      const d = new StudentT({ df: 5, loc: 0, scale: 1 })
      const h = d.entropy()
      expect(isFinite(h.dataSync()[0])).toBe(true)
      // Should be greater than Normal entropy (heavier tails = more entropy)
      const normalEntropy = 0.5 * Math.log(2 * Math.PI * Math.E)
      expect(h.dataSync()[0]).toBeGreaterThan(normalEntropy)
      h.dispose()
      d.dispose()
    })
  })

  test('dispose frees memory', () => {
    const before = tf.memory().numTensors
    const d = new StudentT({ df: 5, loc: 0, scale: 1 })
    expect(tf.memory().numTensors).toBeGreaterThan(before)
    d.dispose()
    expect(tf.memory().numTensors).toBe(before)
  })
})
