import * as tf from '@tensorflow/tfjs'
import { JointDistributionNamed } from '../../../src/distributions/joint_distribution_named'
import { Normal } from '../../../src/distributions/normal'
import { LogNormal } from '../../../src/distributions/log_normal'
import { Bernoulli } from '../../../src/distributions/bernoulli'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('JointDistributionNamed', () => {
  describe('constructor', () => {
    test('basic construction with explicit deps', () => {
      const model = new JointDistributionNamed({
        mu: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        x: { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 1 }) }
      })
      expect(model.name).toBe('JointDistributionNamed')
      expect(model.variableNames).toEqual(['mu', 'x'])
    })

    test('construction with arg parsing', () => {
      const model = new JointDistributionNamed({
        mu: () => new Normal({ loc: 0, scale: 1 }),
        x: ({ mu }) => new Normal({ loc: mu, scale: 1 })
      })
      expect(model.variableNames).toEqual(['mu', 'x'])
    })

    test('custom name', () => {
      const model = new JointDistributionNamed({
        x: () => new Normal({ loc: 0, scale: 1 })
      }, { name: 'MyModel' })
      expect(model.name).toBe('MyModel')
    })

    test('topological sort with multiple deps', () => {
      const model = new JointDistributionNamed({
        c: { deps: ['a', 'b'], fn: ({ a, b }) => new Normal({ loc: tf.add(a, b), scale: 1 }) },
        a: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        b: { deps: ['a'], fn: ({ a }) => new Normal({ loc: a, scale: 1 }) }
      })
      // Should be sorted as: a, b, c (respecting dependencies)
      const names = model.variableNames
      expect(names.indexOf('a')).toBeLessThan(names.indexOf('b'))
      expect(names.indexOf('b')).toBeLessThan(names.indexOf('c'))
    })

    test('throws on circular dependency', () => {
      expect(() => new JointDistributionNamed({
        a: { deps: ['b'], fn: ({ b }) => new Normal({ loc: b, scale: 1 }) },
        b: { deps: ['a'], fn: ({ a }) => new Normal({ loc: a, scale: 1 }) }
      })).toThrow('Circular dependency')
    })

    test('throws on unknown dependency', () => {
      expect(() => new JointDistributionNamed({
        x: { deps: ['missing'], fn: ({ missing }) => new Normal({ loc: missing, scale: 1 }) }
      })).toThrow("Unknown dependency 'missing'")
    })

    test('throws on invalid model spec', () => {
      expect(() => new JointDistributionNamed({
        x: 42
      })).toThrow('Invalid model spec')
    })
  })

  describe('arg parsing', () => {
    test('parses arrow function with destructuring', () => {
      const model = new JointDistributionNamed({
        a: () => new Normal({ loc: 0, scale: 1 }),
        b: ({ a }) => new Normal({ loc: a, scale: 1 }),
        c: ({ a, b }) => new Normal({ loc: tf.add(a, b), scale: 1 })
      })
      const names = model.variableNames
      expect(names.indexOf('a')).toBeLessThan(names.indexOf('b'))
      expect(names.indexOf('a')).toBeLessThan(names.indexOf('c'))
      expect(names.indexOf('b')).toBeLessThan(names.indexOf('c'))
    })

    test('no-arg function has no deps', () => {
      const model = new JointDistributionNamed({
        root: () => new Normal({ loc: 0, scale: 1 })
      })
      expect(model.variableNames).toEqual(['root'])
    })

    test('mixed explicit and arg-parsed specs', () => {
      const model = new JointDistributionNamed({
        a: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        b: ({ a }) => new Normal({ loc: a, scale: 1 })
      })
      expect(model.variableNames).toEqual(['a', 'b'])
    })
  })

  describe('sample', () => {
    test('sample returns dict of tensors', () => {
      const model = new JointDistributionNamed({
        mu: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        x: { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 1 }) }
      })
      const s = model.sample()
      expect(s).toHaveProperty('mu')
      expect(s).toHaveProperty('x')
      expect(s.mu instanceof tf.Tensor).toBe(true)
      expect(s.x instanceof tf.Tensor).toBe(true)
      expect(s.mu.shape).toEqual([])
      expect(s.x.shape).toEqual([])
      s.mu.dispose()
      s.x.dispose()
    })

    test('sample with shape', () => {
      const model = new JointDistributionNamed({
        mu: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        x: { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 1 }) }
      })
      const s = model.sample([10])
      expect(s.mu.shape).toEqual([10])
      expect(s.x.shape).toEqual([10])
      s.mu.dispose()
      s.x.dispose()
    })

    test('downstream variables conditioned on upstream', () => {
      // mu is always exactly 5 (very tight prior)
      // x | mu ~ Normal(mu, 0.01)
      // So x should be very close to 5
      const model = new JointDistributionNamed({
        mu: { deps: [], fn: () => new Normal({ loc: 5, scale: 0.001 }) },
        x: { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 0.001 }) }
      })
      const s = model.sample([100])
      const xData = s.x.dataSync()
      const stats = sampleStats(xData)
      expectClose(stats.mean, 5, { atol: 0.05 })
      s.mu.dispose()
      s.x.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb returns scalar', () => {
      const model = new JointDistributionNamed({
        x: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) }
      })
      const lp = model.logProb({ x: 0 })
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose()
    })

    test('logProb sums component logProbs', () => {
      // Two independent normals: logProb should be sum
      const model = new JointDistributionNamed({
        a: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        b: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) }
      })

      const lp = model.logProb({ a: 0, b: 0 })

      // Each Normal(0,1).logProb(0) = -0.5*log(2π)
      const normalLp0 = -0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], 2 * normalLp0, { atol: 1e-4 })
      lp.dispose()
    })

    test('logProb with dependent variables', () => {
      // mu ~ Normal(0, 1), x | mu ~ Normal(mu, 1)
      const model = new JointDistributionNamed({
        mu: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        x: { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 1 }) }
      })

      // logProb({mu: 0, x: 0}) = logN(0; 0, 1) + logN(0; 0, 1)
      const lp = model.logProb({ mu: 0, x: 0 })
      const normalLp0 = -0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], 2 * normalLp0, { atol: 1e-4 })
      lp.dispose()

      // logProb({mu: 2, x: 3}) = logN(2; 0, 1) + logN(3; 2, 1)
      const lp2 = model.logProb({ mu: 2, x: 3 })
      const expectedMu = -0.5 * 4 - 0.5 * Math.log(2 * Math.PI)
      const expectedX = -0.5 * 1 - 0.5 * Math.log(2 * Math.PI)
      expectClose(lp2.dataSync()[0], expectedMu + expectedX, { atol: 1e-4 })
      lp2.dispose()
    })

    test('logProb with tensor values', () => {
      const model = new JointDistributionNamed({
        x: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) }
      })
      const lp = model.logProb({ x: tf.scalar(1.5) })
      const expected = -0.5 * 1.5 * 1.5 - 0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-4 })
      lp.dispose()
    })
  })

  describe('logProbParts', () => {
    test('returns per-component logProbs', () => {
      const model = new JointDistributionNamed({
        a: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        b: { deps: [], fn: () => new Normal({ loc: 3, scale: 2 }) }
      })

      const parts = model.logProbParts({ a: 0, b: 3 })
      expect(parts).toHaveProperty('a')
      expect(parts).toHaveProperty('b')

      // N(0; 0, 1) at 0
      const lpA = -0.5 * Math.log(2 * Math.PI)
      expectClose(parts.a.dataSync()[0], lpA, { atol: 1e-4 })

      // N(3; 3, 2) at 3
      const lpB = -0.5 * Math.log(2 * Math.PI) - Math.log(2)
      expectClose(parts.b.dataSync()[0], lpB, { atol: 1e-4 })

      parts.a.dispose()
      parts.b.dispose()
    })
  })

  describe('Bayesian linear regression model', () => {
    test('sample and logProb on 3-variable model', () => {
      const model = new JointDistributionNamed({
        w: { deps: [], fn: () => new Normal({ loc: 0, scale: 10 }) },
        sigma: { deps: [], fn: () => new LogNormal({ loc: 0, scale: 1 }) },
        y: { deps: ['w', 'sigma'], fn: ({ w, sigma }) => new Normal({ loc: w, scale: sigma }) }
      })

      // Sample
      const s = model.sample()
      expect(s).toHaveProperty('w')
      expect(s).toHaveProperty('sigma')
      expect(s).toHaveProperty('y')
      expect(s.sigma.dataSync()[0]).toBeGreaterThan(0) // LogNormal is positive

      // LogProb
      const lp = model.logProb(s)
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)

      lp.dispose()
      s.w.dispose()
      s.sigma.dispose()
      s.y.dispose()
    })
  })

  describe('memory management', () => {
    test('sample does not leak tensors from temporary distributions', () => {
      const model = new JointDistributionNamed({
        x: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        y: { deps: ['x'], fn: ({ x }) => new Normal({ loc: x, scale: 1 }) }
      })

      const before = tf.memory().numTensors
      const s = model.sample()
      // Only 2 new tensors (the samples)
      expect(tf.memory().numTensors).toBe(before + 2)
      s.x.dispose()
      s.y.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })

    test('logProb does not leak tensors', () => {
      const model = new JointDistributionNamed({
        x: { deps: [], fn: () => new Normal({ loc: 0, scale: 1 }) },
        y: { deps: ['x'], fn: ({ x }) => new Normal({ loc: x, scale: 1 }) }
      })

      const before = tf.memory().numTensors
      const lp = model.logProb({ x: 0, y: 0 })
      // Only 1 new tensor (the scalar logProb)
      expect(tf.memory().numTensors).toBe(before + 1)
      lp.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
