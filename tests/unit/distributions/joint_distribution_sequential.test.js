import * as tf from '@tensorflow/tfjs'
import { JointDistributionSequential } from '../../../src/distributions/joint_distribution_sequential'
import { Normal } from '../../../src/distributions/normal'
import { LogNormal } from '../../../src/distributions/log_normal'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('JointDistributionSequential', () => {
  describe('constructor', () => {
    test('basic construction', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (mu) => new Normal({ loc: mu, scale: 1 })
      ])
      expect(model.name).toBe('JointDistributionSequential')
      expect(model.numVariables).toBe(2)
    })

    test('custom name', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 })
      ], { name: 'MyModel' })
      expect(model.name).toBe('MyModel')
    })

    test('throws if not given array', () => {
      expect(() => new JointDistributionSequential({})).toThrow('expects an array')
    })
  })

  describe('sample', () => {
    test('sample returns array of tensors', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (mu) => new Normal({ loc: mu, scale: 1 })
      ])
      const s = model.sample()
      expect(Array.isArray(s)).toBe(true)
      expect(s.length).toBe(2)
      expect(s[0] instanceof tf.Tensor).toBe(true)
      expect(s[1] instanceof tf.Tensor).toBe(true)
      expect(s[0].shape).toEqual([])
      expect(s[1].shape).toEqual([])
      s[0].dispose()
      s[1].dispose()
    })

    test('sample with shape', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (mu) => new Normal({ loc: mu, scale: 1 })
      ])
      const s = model.sample([10])
      expect(s[0].shape).toEqual([10])
      expect(s[1].shape).toEqual([10])
      s[0].dispose()
      s[1].dispose()
    })

    test('reverse arg order (TFP convention)', () => {
      // sigma is [1], mu is [0]
      // y receives (sigma, mu) in that order
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 5, scale: 0.001 }),     // [0] mu ≈ 5
        () => new LogNormal({ loc: -2, scale: 0.001 }), // [1] sigma ≈ exp(-2)
        (sigma, mu) => new Normal({ loc: mu, scale: sigma }) // [2] y ≈ mu ≈ 5
      ])
      const s = model.sample([100])
      const yData = s[2].dataSync()
      const stats = sampleStats(yData)
      expectClose(stats.mean, 5, { atol: 0.1 })
      s[0].dispose()
      s[1].dispose()
      s[2].dispose()
    })
  })

  describe('logProb', () => {
    test('logProb returns scalar', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 })
      ])
      const lp = model.logProb([0])
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose()
    })

    test('logProb sums component logProbs', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        () => new Normal({ loc: 0, scale: 1 })
      ])
      const lp = model.logProb([0, 0])
      const normalLp0 = -0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], 2 * normalLp0, { atol: 1e-4 })
      lp.dispose()
    })

    test('logProb with dependencies', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (mu) => new Normal({ loc: mu, scale: 1 })
      ])
      // logProb([0, 0]) = logN(0;0,1) + logN(0;0,1)
      const lp = model.logProb([0, 0])
      const normalLp0 = -0.5 * Math.log(2 * Math.PI)
      expectClose(lp.dataSync()[0], 2 * normalLp0, { atol: 1e-4 })
      lp.dispose()
    })

    test('logProb with reverse arg order', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),     // [0] mu
        () => new Normal({ loc: 0, scale: 1 }),     // [1] sigma_raw
        (sigma_raw, mu) => new Normal({ loc: mu, scale: tf.abs(sigma_raw) })
      ])
      const lp = model.logProb([0, 1, 0])
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)
      lp.dispose()
    })
  })

  describe('logProbParts', () => {
    test('returns per-component logProbs', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        () => new Normal({ loc: 3, scale: 2 })
      ])
      const parts = model.logProbParts([0, 3])
      expect(parts.length).toBe(2)

      const lpA = -0.5 * Math.log(2 * Math.PI)
      expectClose(parts[0].dataSync()[0], lpA, { atol: 1e-4 })

      const lpB = -0.5 * Math.log(2 * Math.PI) - Math.log(2)
      expectClose(parts[1].dataSync()[0], lpB, { atol: 1e-4 })

      parts[0].dispose()
      parts[1].dispose()
    })
  })

  describe('3-variable model', () => {
    test('sample and logProb', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 10 }),
        () => new LogNormal({ loc: 0, scale: 1 }),
        (sigma, w) => new Normal({ loc: w, scale: sigma })
      ])

      const s = model.sample()
      expect(s.length).toBe(3)
      expect(s[1].dataSync()[0]).toBeGreaterThan(0) // LogNormal is positive

      const lp = model.logProb(s)
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)

      lp.dispose()
      s.forEach(t => t.dispose())
    })
  })

  describe('memory management', () => {
    test('sample does not leak tensors', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (x) => new Normal({ loc: x, scale: 1 })
      ])

      const before = tf.memory().numTensors
      const s = model.sample()
      expect(tf.memory().numTensors).toBe(before + 2)
      s[0].dispose()
      s[1].dispose()
      expect(tf.memory().numTensors).toBe(before)
    })

    test('logProb does not leak tensors', () => {
      const model = new JointDistributionSequential([
        () => new Normal({ loc: 0, scale: 1 }),
        (x) => new Normal({ loc: x, scale: 1 })
      ])

      const before = tf.memory().numTensors
      const lp = model.logProb([0, 0])
      expect(tf.memory().numTensors).toBe(before + 1)
      lp.dispose()
      expect(tf.memory().numTensors).toBe(before)
    })
  })
})
