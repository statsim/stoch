import * as tf from '@tensorflow/tfjs'
import { LogNormal } from '../../../src/distributions/log_normal'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('LogNormal distribution', () => {
  describe('constructor', () => {
    test('default params', () => {
      const d = new LogNormal()
      expect(d.loc.dataSync()[0]).toBe(0)
      expect(d.scale.dataSync()[0]).toBe(1)
      expect(d.name).toBe('LogNormal')
      d.dispose()
    })

    test('custom params', () => {
      const d = new LogNormal({ loc: 1, scale: 0.5 })
      expect(d.loc.dataSync()[0]).toBe(1)
      expect(d.scale.dataSync()[0]).toBe(0.5)
      d.dispose()
    })

    test('throws for non-positive scale', () => {
      expect(() => new LogNormal({ scale: 0 })).toThrow('must be positive')
    })
  })

  describe('logProb', () => {
    test('logProb at specific points', () => {
      const d = new LogNormal({ loc: 0, scale: 1 })
      const points = [0.5, 1, 2, 5]
      const lp = d.logProb(tf.tensor(points))
      const data = lp.dataSync()

      for (let i = 0; i < points.length; i++) {
        const y = points[i]
        const logY = Math.log(y)
        // logpdf(y; μ, σ) = -0.5*((logY-μ)/σ)² - logY - log(σ) - 0.5*log(2π)
        const expected = -0.5 * logY * logY - logY - 0.5 * Math.log(2 * Math.PI)
        expectClose(data[i], expected, { atol: 1e-4 })
      }

      lp.dispose()
      d.dispose()
    })

    test('logProb with non-standard params', () => {
      const mu = 2, sigma = 0.5
      const d = new LogNormal({ loc: mu, scale: sigma })
      const y = 5
      const lp = d.logProb(y)

      const logY = Math.log(y)
      const expected = -0.5 * ((logY - mu) / sigma) ** 2 - logY
        - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })

      lp.dispose()
      d.dispose()
    })
  })

  describe('sample', () => {
    test('all samples positive', () => {
      const d = new LogNormal({ loc: 0, scale: 1 })
      const s = d.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
      }
      s.dispose()
      d.dispose()
    })

    test('sample shape', () => {
      const d = new LogNormal({ loc: 0, scale: 1 })
      const s = d.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      d.dispose()
    })

    test('sample statistics', () => {
      const mu = 0, sigma = 0.5
      const d = new LogNormal({ loc: mu, scale: sigma })
      const s = d.sample([50000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      const expectedMean = Math.exp(mu + sigma * sigma / 2)
      const expectedVar = (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma)

      expectClose(stats.mean, expectedMean, { atol: 0.05 })
      expectClose(stats.variance, expectedVar, { atol: 0.05 })

      s.dispose()
      d.dispose()
    })
  })

  describe('mean', () => {
    test('mean = exp(μ + σ²/2)', () => {
      const d = new LogNormal({ loc: 0, scale: 1 })
      const m = d.mean()
      expectClose(m.dataSync()[0], Math.exp(0.5), { atol: 1e-4 })
      m.dispose()
      d.dispose()
    })

    test('mean with custom params', () => {
      const mu = 1, sigma = 2
      const d = new LogNormal({ loc: mu, scale: sigma })
      const m = d.mean()
      expectClose(m.dataSync()[0], Math.exp(mu + sigma * sigma / 2), { atol: 1e-3 })
      m.dispose()
      d.dispose()
    })
  })

  describe('variance', () => {
    test('variance formula', () => {
      const mu = 0, sigma = 1
      const d = new LogNormal({ loc: mu, scale: sigma })
      const v = d.variance()
      const expected = (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma)
      expectClose(v.dataSync()[0], expected, { atol: 1e-3 })
      v.dispose()
      d.dispose()
    })
  })

  describe('mode', () => {
    test('mode = exp(μ - σ²)', () => {
      const mu = 1, sigma = 0.5
      const d = new LogNormal({ loc: mu, scale: sigma })
      const mode = d.mode()
      expectClose(mode.dataSync()[0], Math.exp(mu - sigma * sigma), { atol: 1e-4 })
      mode.dispose()
      d.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy formula', () => {
      const mu = 0, sigma = 1
      const d = new LogNormal({ loc: mu, scale: sigma })
      const h = d.entropy()
      // H = μ + 0.5 + log(σ) + 0.5*log(2π)
      const expected = mu + 0.5 + Math.log(sigma) + 0.5 * Math.log(2 * Math.PI)
      expectClose(h.dataSync()[0], expected, { atol: 1e-4 })
      h.dispose()
      d.dispose()
    })
  })

  describe('batched', () => {
    test('batched params', () => {
      const d = new LogNormal({ loc: [0, 1, 2], scale: 1 })
      expect(d.batchShape).toEqual([3])
      const s = d.sample([50])
      expect(s.shape).toEqual([50, 3])
      s.dispose()
      d.dispose()
    })
  })

  test('dispose frees memory', () => {
    const before = tf.memory().numTensors
    const d = new LogNormal({ loc: 0, scale: 1 })
    expect(tf.memory().numTensors).toBeGreaterThan(before)
    d.dispose()
    expect(tf.memory().numTensors).toBe(before)
  })
})
