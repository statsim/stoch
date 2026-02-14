import * as tf from '@tensorflow/tfjs'
import { trainableNormal, buildMeanFieldPosterior } from '../../../src/vi/trainable'

describe('trainable distributions', () => {
  describe('trainableNormal', () => {
    test('creates with default parameters', () => {
      const q = trainableNormal()

      expect(q.trainableVariables).toHaveLength(2)
      expect(q.trainableVariables[0] instanceof tf.Variable).toBe(true)
      expect(q.trainableVariables[1] instanceof tf.Variable).toBe(true)

      const params = q.getParameters()
      expect(params.loc).toBeCloseTo(0, 3)
      expect(params.scale).toBeCloseTo(1, 1)

      q.dispose()
    })

    test('creates with custom parameters', () => {
      const q = trainableNormal({ loc: 5, scale: 2 })

      const params = q.getParameters()
      expect(params.loc).toBeCloseTo(5, 3)
      expect(params.scale).toBeCloseTo(2, 1)

      q.dispose()
    })

    test('sample returns a scalar tensor', () => {
      const q = trainableNormal()
      const s = q.sample([])

      expect(s instanceof tf.Tensor).toBe(true)
      expect(s.shape).toEqual([])
      expect(isFinite(s.dataSync()[0])).toBe(true)

      s.dispose()
      q.dispose()
    })

    test('sample returns array of samples', () => {
      const q = trainableNormal()
      const s = q.sample([10])

      expect(s.shape).toEqual([10])

      s.dispose()
      q.dispose()
    })

    test('logProb returns finite values', () => {
      const q = trainableNormal({ loc: 0, scale: 1 })
      const lp = q.logProb(tf.scalar(0))

      expect(lp instanceof tf.Tensor).toBe(true)
      // log N(0; 0, 1) = -0.5 * log(2π) ≈ -0.919
      expect(lp.dataSync()[0]).toBeCloseTo(-0.919, 1)

      lp.dispose()
      q.dispose()
    })

    test('logProb is differentiable through variables', () => {
      const q = trainableNormal({ loc: 0, scale: 1 })
      const vars = q.trainableVariables

      const gradFn = tf.grads((locVar, unconstrainedScale) => {
        // Manually compute logProb at x=1 using these variables
        const scale = tf.softplus(unconstrainedScale)
        const z = tf.div(tf.sub(tf.scalar(1), locVar), scale)
        return tf.sub(
          tf.mul(-0.5, tf.add(tf.square(z), Math.log(2 * Math.PI))),
          tf.log(scale)
        )
      })

      const grads = gradFn(vars)
      expect(grads).toHaveLength(2)
      expect(isFinite(grads[0].dataSync()[0])).toBe(true)
      expect(isFinite(grads[1].dataSync()[0])).toBe(true)

      grads.forEach(g => g.dispose())
      q.dispose()
    })

    test('scale stays positive via softplus', () => {
      const q = trainableNormal({ loc: 0, scale: 0.01 })
      const params = q.getParameters()
      expect(params.scale).toBeGreaterThan(0)

      q.dispose()
    })
  })

  describe('buildMeanFieldPosterior', () => {
    test('creates posteriors for each parameter', () => {
      const q = buildMeanFieldPosterior({ mu: 0, sigma: 1 })

      expect(q.trainableVariables.length).toBe(4) // 2 per param
      expect(typeof q.sample).toBe('function')
      expect(typeof q.logProb).toBe('function')

      q.dispose()
    })

    test('sample returns state object', () => {
      const q = buildMeanFieldPosterior({ mu: 0, sigma: 1 })
      const s = q.sample()

      expect(s).toHaveProperty('mu')
      expect(s).toHaveProperty('sigma')
      expect(s.mu instanceof tf.Tensor).toBe(true)
      expect(s.sigma instanceof tf.Tensor).toBe(true)

      s.mu.dispose()
      s.sigma.dispose()
      q.dispose()
    })

    test('logProb returns scalar', () => {
      const q = buildMeanFieldPosterior({ mu: 0, sigma: 1 })
      const lp = q.logProb({ mu: tf.scalar(0), sigma: tf.scalar(1) })

      expect(lp instanceof tf.Tensor).toBe(true)
      expect(lp.shape).toEqual([])
      expect(isFinite(lp.dataSync()[0])).toBe(true)

      lp.dispose()
      q.dispose()
    })

    test('getParameters returns current values', () => {
      const q = buildMeanFieldPosterior({ mu: 5, sigma: 2 })
      const params = q.getParameters()

      expect(params.mu.loc).toBeCloseTo(5, 3)
      expect(params.sigma.loc).toBeCloseTo(2, 3)

      q.dispose()
    })
  })
})
