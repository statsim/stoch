import * as tf from '@tensorflow/tfjs'
import { Normal } from '../../../src/distributions/normal'
import { Uniform } from '../../../src/distributions/uniform'
import { TransformedDistribution } from '../../../src/distributions/transformed_distribution'
import { Exp } from '../../../src/bijectors/exp'
import { Shift } from '../../../src/bijectors/shift'
import { Scale } from '../../../src/bijectors/scale'
import { Chain } from '../../../src/bijectors/chain'
import { Softplus } from '../../../src/bijectors/softplus'
import { expectClose, sampleStats, autoTolerance } from '../../helpers/tolerance'

describe('TransformedDistribution', () => {
  describe('Normal + Exp = LogNormal', () => {
    test('logProb matches analytical LogNormal', () => {
      // LogNormal(mu=0, sigma=1): logpdf(y) = -log(y) - 0.5*log(2π) - 0.5*(log(y))²
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Exp()
      })

      const points = [0.5, 1, 2, 5]
      const lp = dist.logProb(tf.tensor(points))
      const data = lp.dataSync()

      for (let i = 0; i < points.length; i++) {
        const y = points[i]
        const logY = Math.log(y)
        // logpdf = Normal.logProb(log(y)) + ILDJ(y)
        // = -0.5*(log(y))² - 0.5*log(2π) - log(y)
        const expected = -0.5 * logY * logY - 0.5 * Math.log(2 * Math.PI) - logY
        expectClose(data[i], expected, { atol: 1e-4 })
      }

      lp.dispose()
      dist.dispose()
    })

    test('logProb with non-standard params', () => {
      // LogNormal(mu=1, sigma=0.5)
      const mu = 1, sigma = 0.5
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: mu, scale: sigma }),
        bijector: new Exp()
      })

      const y = 2
      const lp = dist.logProb(y)
      const logY = Math.log(y)
      // logpdf = -0.5*((logY-mu)/sigma)² - log(sigma) - 0.5*log(2π) - log(y)
      const expected = -0.5 * ((logY - mu) / sigma) ** 2 - Math.log(sigma)
        - 0.5 * Math.log(2 * Math.PI) - logY
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })

      lp.dispose()
      dist.dispose()
    })
  })

  describe('sample', () => {
    test('samples are in correct domain', () => {
      // LogNormal: all samples should be positive
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Exp()
      })
      const s = dist.sample([1000])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
      }
      s.dispose()
      dist.dispose()
    })

    test('sample shape is correct', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Exp()
      })
      const s = dist.sample([100])
      expect(s.shape).toEqual([100])
      s.dispose()
      dist.dispose()
    })

    test('batched sample shape', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: [0, 1], scale: 1 }),
        bijector: new Exp()
      })
      expect(dist.batchShape).toEqual([2])
      const s = dist.sample([50])
      expect(s.shape).toEqual([50, 2])
      s.dispose()
      dist.dispose()
    })

    test('LogNormal sample statistics', () => {
      const mu = 0, sigma = 1
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: mu, scale: sigma }),
        bijector: new Exp()
      })
      const s = dist.sample([50000])
      const data = s.dataSync()
      const stats = sampleStats(data)

      // LogNormal mean = exp(mu + sigma²/2) = exp(0.5) ≈ 1.6487
      const expectedMean = Math.exp(mu + sigma * sigma / 2)
      // LogNormal variance = (exp(sigma²) - 1) * exp(2*mu + sigma²)
      const expectedVar = (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma)

      expectClose(stats.mean, expectedMean, { atol: 0.1 })
      expectClose(stats.variance, expectedVar, { atol: 1.0 })

      s.dispose()
      dist.dispose()
    })
  })

  describe('affine transform', () => {
    test('Shift+Scale transform of Normal', () => {
      // Y = 2*X + 3 where X ~ Normal(0, 1) ⟹ Y ~ Normal(3, 2)
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Chain({ bijectors: [new Shift({ shift: 3 }), new Scale({ scale: 2 })] })
      })

      // logProb at mean should match Normal(3, 2)
      const ref = new Normal({ loc: 3, scale: 2 })
      const lpTransformed = dist.logProb(3)
      const lpRef = ref.logProb(3)
      expectClose(lpTransformed.dataSync()[0], lpRef.dataSync()[0], { atol: 1e-3 })

      lpTransformed.dispose()
      lpRef.dispose()
      dist.dispose()
      ref.dispose()
    })
  })

  describe('properties', () => {
    test('name defaults to TransformedBaseName', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal(),
        bijector: new Exp()
      })
      expect(dist.name).toBe('TransformedNormal')
      dist.dispose()
    })

    test('custom name', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal(),
        bijector: new Exp(),
        name: 'LogNormal'
      })
      expect(dist.name).toBe('LogNormal')
      dist.dispose()
    })

    test('batchShape from base distribution', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: [0, 1, 2], scale: 1 }),
        bijector: new Exp()
      })
      expect(dist.batchShape).toEqual([3])
      dist.dispose()
    })

    test('eventShape from base distribution', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Exp()
      })
      expect(dist.eventShape).toEqual([])
      dist.dispose()
    })
  })

  describe('entropy', () => {
    test('entropy with constant-Jacobian bijector', () => {
      // Y = 2X + 3 where X ~ Normal(0, 1)
      // H(Y) = H(X) + log|det(J)| = H(Normal(0,1)) + log(2)
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Scale({ scale: 2 })
      })
      const h = dist.entropy()
      const normalEntropy = 0.5 * Math.log(2 * Math.PI * Math.E)
      expectClose(h.dataSync()[0], normalEntropy + Math.log(2), { atol: 1e-4 })
      h.dispose()
      dist.dispose()
    })

    test('entropy throws for non-constant Jacobian', () => {
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Exp()
      })
      expect(() => dist.entropy()).toThrow('non-constant Jacobian')
      dist.dispose()
    })
  })

  describe('dispose', () => {
    test('disposes both distribution and bijector', () => {
      const before = tf.memory().numTensors
      const dist = new TransformedDistribution({
        distribution: new Normal({ loc: 0, scale: 1 }),
        bijector: new Scale({ scale: 2 })
      })
      expect(tf.memory().numTensors).toBeGreaterThan(before)
      dist.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
